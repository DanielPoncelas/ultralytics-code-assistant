import json

import httpx
import pytest

from code_assistant.rerank import OpenRouterReranker
from code_assistant.retrieval import SearchResult


def result(name: str) -> SearchResult:
    return SearchResult(
        id=name, kind="method", path="p.py", qualified_name=name, code=f"def {name}(): ...",
        start_line=1, end_line=2, github_url="https://x", score=0.9,
    )  # fmt: skip


def reranker(handler) -> OpenRouterReranker:
    http = httpx.Client(base_url="https://t/api/v1", transport=httpx.MockTransport(handler))
    return OpenRouterReranker("sk-test", "r/model", http_client=http)


def test_rerank_orders_by_relevance_and_uses_the_reranker_score():
    sent = {}

    def handler(request):
        sent.update(url=str(request.url), auth=request.headers["Authorization"])
        sent.update(json.loads(request.content))
        scores = [{"index": 0, "relevance_score": 0.01}, {"index": 1, "relevance_score": 0.04}]
        return httpx.Response(200, json={"results": scores})

    ranked = reranker(handler).rerank("nms?", [result("A"), result("B")])

    assert [(r.qualified_name, r.score) for r in ranked] == [("B", 0.04), ("A", 0.01)]
    assert sent["url"] == "https://t/api/v1/rerank" and sent["auth"] == "Bearer sk-test"
    assert sent["model"] == "r/model" and sent["query"] == "nms?"
    assert sent["documents"][1] == "p.py :: B\ndef B(): ..."


def test_rerank_raises_on_http_errors_and_skips_empty_input():
    failing = reranker(lambda request: httpx.Response(429, json={"error": "busy"}))

    with pytest.raises(httpx.HTTPStatusError):
        failing.rerank("q", [result("A")])
    assert failing.rerank("q", []) == []
