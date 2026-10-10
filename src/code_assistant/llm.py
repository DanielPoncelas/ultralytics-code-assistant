"""OpenRouter chat-completions client with ordered model fallback and streaming.

Free models are shared across all OpenRouter users and can be rate-limited upstream (429),
temporarily unavailable, or return an empty completion (reasoning models that spend the
token budget thinking). The client therefore tries an ordered list of models and reports
which one actually answered, plus every failed attempt.
"""

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from code_assistant.config import Settings

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}

# Account-level problem: trying another model cannot help. (402 "insufficient credits" is
# not fatal: a paid model fails but the free models in the chain still answer.)
FATAL_STATUS = {401: "invalid API key"}


class LLMError(RuntimeError):
    """No model produced an answer."""


@dataclass(frozen=True)
class Attempt:
    model: str
    error: str


@dataclass(frozen=True)
class Completion:
    text: str
    model: str  # model that produced the answer, as reported by OpenRouter
    latency_s: float
    failed_attempts: tuple[Attempt, ...] = ()
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # "length" means the token budget ran out. Reasoning models spend part of the budget
    # thinking before they answer, so this can happen even for short visible answers.
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class _ModelFailure(Exception):
    """This model failed before producing text; the next one should be tried."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable  # transient (429, 5xx, network) vs. model behaviour


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        models: Sequence[str],
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        timeout_s: float = 90.0,
        max_retries: int = 0,
        retry_backoff_s: float = 2.0,
        first_token_timeout_s: float | None = None,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required")
        self.models = tuple(models)
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Retries of the *same* model on transient errors before falling back. Interactive use
        # keeps 0 (falling back is faster than waiting); evaluation uses retries so that a
        # momentary rate limit isn't scored as a model failure.
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.first_token_timeout_s = first_token_timeout_s  # streaming only
        self._sleep = sleep
        self._http = http_client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self._http.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                # Optional OpenRouter app attribution headers.
                "HTTP-Referer": "https://github.com/DanielPoncelas/ultralytics-code-assistant",
                "X-Title": "Ultralytics Code Assistant",
            }
        )

    @classmethod
    def from_settings(
        cls,
        settings: "Settings",
        models: Sequence[str] | None = None,
        max_retries: int = 0,
        interactive: bool = False,
    ) -> "OpenRouterClient":
        timeout = settings.llm_interactive_timeout_s if interactive else settings.llm_timeout_s
        return cls(
            max_retries=max_retries,
            first_token_timeout_s=settings.llm_interactive_timeout_s if interactive else None,
            api_key=settings.openrouter_api_key.get_secret_value(),
            models=models or settings.llm_models,
            base_url=settings.openrouter_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_s=timeout,
        )

    def close(self) -> None:
        self._http.close()

    # -- public API --------------------------------------------------------------------

    def complete(self, messages: list[Message]) -> Completion:
        """Return the first non-empty answer, trying models in order."""
        failures: list[Attempt] = []
        for model in self.models:
            for retry in range(self.max_retries + 1):
                started = time.perf_counter()
                try:
                    data = self._post(model, messages)
                    try:
                        text = (data["choices"][0]["message"].get("content") or "").strip()
                    except (KeyError, IndexError, TypeError, AttributeError) as exc:
                        raise _ModelFailure(f"malformed response: {str(data)[:100]}") from exc
                    if not text:
                        raise _ModelFailure("empty completion", retryable=False)
                except _ModelFailure as exc:
                    failures.append(Attempt(model, str(exc)))
                    if exc.retryable and retry < self.max_retries:
                        self._sleep(self.retry_backoff_s * 2**retry)
                        continue
                    break
                usage = data.get("usage") or {}
                return Completion(
                    text=text,
                    model=data.get("model", model),
                    latency_s=time.perf_counter() - started,
                    failed_attempts=tuple(failures),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    finish_reason=data["choices"][0].get("finish_reason"),
                )
        raise LLMError(_summary(failures))

    def stream(self, messages: list[Message]) -> Iterator[str | Completion]:
        """Yield text deltas as they arrive, then a final `Completion`.

        Falls back to the next model only while nothing has been yielded yet: once text is
        on the user's screen, switching models would splice two different answers.
        """
        failures: list[Attempt] = []
        for model in self.models:
            started = time.perf_counter()
            parts: list[str] = []
            shown = False  # True once real text reached the caller: no fallback after that
            reported_model, usage, finish_reason = model, {}, None
            deadline = started + self.first_token_timeout_s if self.first_token_timeout_s else None
            try:
                for event in self._stream_events(model, messages):
                    if not shown and deadline and time.perf_counter() > deadline:
                        raise _ModelFailure("timed out waiting for the first token")
                    if event is None:  # keep-alive
                        continue
                    reported_model = event.get("model") or reported_model
                    usage = event.get("usage") or usage
                    choice = (event.get("choices") or [None])[0] or {}
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = (choice.get("delta") or {}).get("content")
                    if not isinstance(delta, str) or not delta:
                        continue
                    parts.append(delta)
                    if shown:
                        yield delta
                    elif "".join(parts).strip():  # hold back leading whitespace
                        shown = True
                        yield "".join(parts)
                if not shown:
                    raise _ModelFailure("empty completion", retryable=False)
            except _ModelFailure as exc:
                if shown:
                    raise LLMError(f"{model} failed mid-answer: {exc}") from exc
                failures.append(Attempt(model, str(exc)))
                continue
            yield Completion(
                text="".join(parts).strip(),
                model=reported_model,
                latency_s=time.perf_counter() - started,
                failed_attempts=tuple(failures),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                finish_reason=finish_reason,
            )
            return
        raise LLMError(_summary(failures))

    # -- HTTP --------------------------------------------------------------------------

    def _payload(self, model: str, messages: list[Message], stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if stream:
            payload["usage"] = {"include": True}  # token counts in the final chunk
        return payload

    def _post(self, model: str, messages: list[Message]) -> dict[str, Any]:
        try:
            response = self._http.post(
                "/chat/completions", json=self._payload(model, messages, stream=False)
            )
        except httpx.HTTPError as exc:
            raise _ModelFailure(f"network error: {exc!r}") from exc
        _check_status(response)
        try:
            data = response.json()
        except ValueError as exc:  # e.g. an HTML error page from a proxy
            raise _ModelFailure(f"malformed response: {response.text[:100]}") from exc
        if not isinstance(data, dict):
            raise _ModelFailure(f"malformed response: {str(data)[:100]}")
        if "error" in data:  # OpenRouter can report provider errors with HTTP 200
            raise _ModelFailure(f"provider error: {data['error']}")
        return data

    def _stream_events(
        self, model: str, messages: list[Message]
    ) -> Iterator[dict[str, Any] | None]:
        """Parse OpenRouter's server-sent events into JSON chunks (None for keep-alives).

        Keep-alives are surfaced so the caller can enforce a first-token deadline: httpx's
        timeout only limits the gap between bytes, and OpenRouter sends keep-alive comments
        while a model is queued or reasoning, so a slow model would never time out.
        """
        try:
            with self._http.stream(
                "POST", "/chat/completions", json=self._payload(model, messages, stream=True)
            ) as response:
                if response.status_code != 200:
                    response.read()
                    _check_status(response)
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        yield None  # blank line or keep-alive (": OPENROUTER PROCESSING")
                        continue
                    data = line.removeprefix("data: ")
                    if data == "[DONE]":
                        return
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise _ModelFailure(f"malformed stream chunk: {data[:100]}") from exc
                    if not isinstance(event, dict):
                        raise _ModelFailure(f"malformed stream chunk: {data[:100]}")
                    if "error" in event:
                        raise _ModelFailure(f"provider error: {event['error']}")
                    yield event
        except httpx.HTTPError as exc:
            raise _ModelFailure(f"network error: {exc!r}") from exc


def _check_status(response: httpx.Response) -> None:
    if response.status_code == 200:
        return
    detail = response.text[:300]
    if response.status_code in FATAL_STATUS:
        raise LLMError(
            f"OpenRouter returned {response.status_code} ({FATAL_STATUS[response.status_code]}): "
            f"{detail}"
        )
    # 408/429/5xx are transient; other 4xx (bad request, no endpoint, moderation) are not.
    transient = response.status_code in (408, 429) or response.status_code >= 500
    raise _ModelFailure(f"HTTP {response.status_code}: {detail}", retryable=transient)


def _summary(failures: list[Attempt]) -> str:
    lines = "\n".join(f"  - {a.model}: {a.error}" for a in failures)
    return f"All models failed:\n{lines}"
