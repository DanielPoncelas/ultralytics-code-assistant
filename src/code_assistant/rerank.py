"""Reranking with a hosted cross-encoder (OpenRouter's `/rerank` endpoint).

Vector search compares a question and a chunk that were embedded separately. A reranker reads
each (question, code) pair together, which ranks better but is too slow for the whole corpus,
so it only reorders the ~33 candidates returned by the vector and keyword searches.
"""

from dataclasses import replace
from typing import TYPE_CHECKING

import httpx

from code_assistant.retrieval import SearchResult

if TYPE_CHECKING:
    from code_assistant.config import Settings


class OpenRouterReranker:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: float = 10.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._http = http_client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self._http.headers.update({"Authorization": f"Bearer {api_key}"})

    @classmethod
    def from_settings(cls, settings: "Settings") -> "OpenRouterReranker | None":
        """None when `RERANK_MODEL` is empty (reranking disabled)."""
        if not settings.rerank_model:
            return None
        return cls(
            settings.openrouter_api_key.get_secret_value(),
            settings.rerank_model,
            base_url=settings.openrouter_base_url,
        )

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Return `results` most relevant first, each scored by the reranker."""
        if not results:
            return []
        response = self._http.post(
            "/rerank",
            json={
                "model": self.model,
                "query": query,
                "documents": [f"{r.path} :: {r.qualified_name}\n{r.code}" for r in results],
            },
        )
        response.raise_for_status()
        ranked = sorted(response.json()["results"], key=lambda item: -item["relevance_score"])
        return [replace(results[item["index"]], score=item["relevance_score"]) for item in ranked]
