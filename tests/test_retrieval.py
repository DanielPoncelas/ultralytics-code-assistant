import pytest

from code_assistant.retrieval import (
    CANDIDATES_PER_RESULT,
    IndexMismatchError,
    Retriever,
    build_filter,
    check_index_compatible,
    deduplicate,
)


def make_doc(name: str, score: float, part: int = 1, total_parts: int = 1) -> dict:
    return {
        "_id": f"{name}-{part}",
        "kind": "method",
        "path": "ultralytics/engine/trainer.py",
        "qualified_name": name,
        "code": f"def {name}(): ...",
        "start_line": 1,
        "end_line": 2,
        "github_url": "https://github.com/x",
        "score": score,
        "part": part,
        "total_parts": total_parts,
    }


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2]


class FakeStore:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.calls: list[dict] = []

    def vector_search(self, query_vector, limit, num_candidates, filters=None):
        self.calls.append({"limit": limit, "num_candidates": num_candidates, "filters": filters})
        return self.documents[:limit]


def test_deduplicate_keeps_first_occurrence_in_order():
    items = ["a1", "b1", "a2", "c1", "b2"]

    assert deduplicate(items, key=lambda s: s[0]) == ["a1", "b1", "c1"]


def test_search_collapses_parts_of_the_same_symbol_and_returns_k_results():
    store = FakeStore(
        [
            make_doc("Exporter.__call__", 0.95, part=2, total_parts=3),
            make_doc("Exporter.__call__", 0.93, part=1, total_parts=3),
            make_doc("Exporter.export_onnx", 0.90),
            make_doc("Exporter.export_engine", 0.85),
        ]
    )
    embedder = FakeEmbedder()

    results = Retriever(store, embedder).search("how does export work?", k=2)

    assert [r.qualified_name for r in results] == ["Exporter.__call__", "Exporter.export_onnx"]
    assert results[0].part == 2  # the best-scoring part is kept
    assert embedder.queries == ["how does export work?"]


def test_search_overfetches_and_asks_for_enough_candidates():
    store = FakeStore([make_doc("f", 0.9)])

    Retriever(store, FakeEmbedder()).search("q", k=5)

    (call,) = store.calls
    assert call["limit"] == 15  # 3x overfetch to survive deduplication
    assert call["num_candidates"] == 15 * CANDIDATES_PER_RESULT
    assert call["filters"] is None


def test_search_passes_filters_to_the_store():
    store = FakeStore([])

    Retriever(store, FakeEmbedder()).search("q", k=3, filters={"task": "segment"})

    assert store.calls[0]["filters"] == {"task": {"$eq": "segment"}}


def test_build_filter_combines_fields_and_rejects_unknown_ones():
    assert build_filter(None) is None
    assert build_filter({"folder": "data", "kind": "class"}) == {
        "$and": [{"folder": {"$eq": "data"}}, {"kind": {"$eq": "class"}}]
    }
    with pytest.raises(ValueError, match="Unsupported filter"):
        build_filter({"path": "x"})


def test_index_compatibility_check():
    check_index_compatible("model/a", "model/a")  # no error

    with pytest.raises(IndexMismatchError, match="empty"):
        check_index_compatible(None, "model/a")
    with pytest.raises(IndexMismatchError, match="not comparable"):
        check_index_compatible("model/a", "model/b")
