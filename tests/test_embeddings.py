import json

import httpx
import numpy as np

from code_assistant import embeddings
from code_assistant.embeddings import LOCAL_MODELS, Embedder


def api_embedder(handler) -> Embedder:
    http = httpx.Client(base_url="https://t/api/v1", transport=httpx.MockTransport(handler))
    return Embedder("qwen/qwen3-embedding-8b", http_client=http)


def test_api_models_are_opt_in_and_never_load_a_local_model():
    assert "BAAI/bge-small-en-v1.5" in LOCAL_MODELS
    assert "qwen/qwen3-embedding-8b" not in LOCAL_MODELS
    assert api_embedder(lambda r: httpx.Response(500)).device == "openrouter"


def test_api_embedder_batches_adds_the_query_prompt_pins_the_provider_and_normalizes(monkeypatch):
    monkeypatch.setattr(embeddings, "API_BATCH_SIZE", 2)
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        data = [{"index": i, "embedding": [3.0, 4.0]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": list(reversed(data))})  # any order

    embedder = api_embedder(handler)
    vectors = embedder.embed_documents(["a", "b", "c"])
    query = embedder.embed_query("q")

    prompt = embedder.spec.query_prompt  # documents are embedded as-is
    assert [r["input"] for r in requests] == [["a", "b"], ["c"], [f"{prompt}q"]]
    assert requests[0]["provider"] == {"order": ["nebius"], "allow_fallbacks": False}
    np.testing.assert_allclose(vectors, [[0.6, 0.8]] * 3)
    np.testing.assert_allclose(query, [0.6, 0.8])


def test_api_embedder_backs_off_on_rate_limits(monkeypatch):
    sleeps = []
    monkeypatch.setattr(embeddings.time, "sleep", sleeps.append)
    responses = iter(
        [
            httpx.Response(429),
            httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]}),
        ]
    )

    assert api_embedder(lambda request: next(responses)).embed_query("q") == [1.0, 0.0]
    assert sleeps == [embeddings.API_BACKOFF_S]


def test_api_failures_become_a_clear_embedding_error():
    import pytest

    from code_assistant.embeddings import EmbeddingError

    with pytest.raises(EmbeddingError, match="needs OpenRouter credits \\(HTTP 402\\)"):
        api_embedder(lambda request: httpx.Response(402)).embed_query("q")
