import json

import httpx
import pytest

from code_assistant.llm import Completion, LLMError, OpenRouterClient

MESSAGES = [{"role": "user", "content": "hi"}]


def make_client(handler, models=("a/model", "b/model")) -> OpenRouterClient:
    http = httpx.Client(
        base_url="https://openrouter.test/api/v1", transport=httpx.MockTransport(handler)
    )
    return OpenRouterClient("sk-test", models, http_client=http)


def completion_body(text: str, model: str) -> dict:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def sse(*events: dict) -> bytes:
    lines = [": OPENROUTER PROCESSING", ""]
    lines += [f"data: {json.dumps(e)}\n" for e in events]
    lines.append("data: [DONE]\n")
    return "\n".join(lines).encode()


def chunk(text: str, model: str = "a/model") -> dict:
    return {"model": model, "choices": [{"delta": {"content": text}}]}


def test_complete_returns_first_model_answer_and_sends_expected_request():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion_body("Hello [1]", "a/model"))

    result = make_client(handler).complete(MESSAGES)

    assert result.text == "Hello [1]"
    assert result.model == "a/model"
    assert result.failed_attempts == ()
    assert (result.prompt_tokens, result.completion_tokens) == (10, 5)
    body = json.loads(seen[0].content)
    assert body["model"] == "a/model" and body["stream"] is False and body["temperature"] == 0.0
    assert seen[0].headers["Authorization"] == "Bearer sk-test"


@pytest.mark.parametrize(
    "first_response",
    [
        httpx.Response(429, json={"error": {"message": "rate-limited upstream"}}),
        httpx.Response(503, text="unavailable"),
        httpx.Response(200, json=completion_body("   ", "a/model")),  # empty completion
        httpx.Response(200, json={"error": {"message": "provider failed"}}),  # error with 200
    ],
)
def test_complete_falls_back_to_next_model(first_response):
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["model"] == "a/model":
            return first_response
        return httpx.Response(200, json=completion_body("from b", "b/model"))

    result = make_client(handler).complete(MESSAGES)

    assert result.text == "from b"
    assert result.model == "b/model"
    assert [a.model for a in result.failed_attempts] == ["a/model"]


def test_complete_falls_back_on_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["model"] == "a/model":
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=completion_body("ok", "b/model"))

    assert make_client(handler).complete(MESSAGES).model == "b/model"


def test_complete_raises_with_all_attempts_when_every_model_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="busy")

    with pytest.raises(LLMError, match="a/model: HTTP 429.*\n.*b/model: HTTP 429"):
        make_client(handler).complete(MESSAGES)


@pytest.mark.parametrize("status", [401, 402])
def test_account_errors_stop_immediately(status):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, text="nope")

    with pytest.raises(LLMError, match=str(status)):
        make_client(handler).complete(MESSAGES)
    assert len(calls) == 1  # no point trying other models


def test_stream_yields_deltas_then_completion():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            content=sse(
                chunk("Hel"),
                chunk("lo"),
                {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}},
            ),
        )

    events = list(make_client(handler).stream(MESSAGES))

    assert events[:2] == ["Hel", "lo"]
    final = events[-1]
    assert isinstance(final, Completion)
    assert (final.text, final.model, final.prompt_tokens) == ("Hello", "a/model", 7)


def test_stream_falls_back_if_first_model_fails_before_any_text():
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["model"] == "a/model":
            return httpx.Response(200, content=sse())  # empty stream
        return httpx.Response(200, content=sse(chunk("ok", "b/model")))

    events = list(make_client(handler).stream(MESSAGES))

    assert events[0] == "ok"
    assert events[-1].model == "b/model"
    assert events[-1].failed_attempts[0].error == "empty completion"


def test_stream_does_not_switch_models_after_text_was_shown():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse(chunk("partial"), {"error": {"message": "boom"}}))

    stream = make_client(handler).stream(MESSAGES)

    assert next(stream) == "partial"
    with pytest.raises(LLMError, match="failed mid-answer"):
        list(stream)


def test_complete_retries_same_model_on_transient_errors_with_backoff():
    responses = iter([httpx.Response(429, text="busy"), httpx.Response(503, text="down")])
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses, httpx.Response(200, json=completion_body("ok", "a/model")))

    http = httpx.Client(base_url="https://t", transport=httpx.MockTransport(handler))
    client = OpenRouterClient("k", ["a/model"], max_retries=3, retry_backoff_s=1.0,
                              http_client=http, sleep=sleeps.append)  # fmt: skip

    result = client.complete(MESSAGES)

    assert result.model == "a/model"
    assert sleeps == [1.0, 2.0]  # exponential backoff
    assert len(result.failed_attempts) == 2


def test_complete_does_not_retry_empty_completions():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=completion_body("", "a/model"))

    http = httpx.Client(base_url="https://t", transport=httpx.MockTransport(handler))
    client = OpenRouterClient(
        "k", ["a/model"], max_retries=3, http_client=http, sleep=lambda s: None
    )

    with pytest.raises(LLMError, match="empty completion"):
        client.complete(MESSAGES)
    assert len(calls) == 1


def test_finish_reason_length_marks_truncated_answers():
    def handler(request: httpx.Request) -> httpx.Response:
        body = completion_body("partial answer", "a/model")
        body["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=body)

    assert make_client(handler).complete(MESSAGES).truncated


def test_stream_captures_finish_reason():
    final = {"model": "a/model", "choices": [{"delta": {}, "finish_reason": "stop"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse(chunk("done"), final))

    completion = list(make_client(handler).stream(MESSAGES))[-1]
    assert completion.finish_reason == "stop" and not completion.truncated
