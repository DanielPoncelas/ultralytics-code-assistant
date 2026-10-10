import numpy as np
import pytest

from code_assistant.evaluation.fact_recall import FactVerdict
from code_assistant.evaluation.search_eval import LocalVectorStore, StrategyResult


class FakeAtlas:
    def text_search(self, query, limit, filters=None):
        return [{"_id": "kw", "query": query, "limit": limit}]


def test_local_store_ranks_saved_vectors_by_cosine_and_delegates_keywords():
    documents = [{"_id": "a"}, {"_id": "b"}, {"_id": "c"}]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    store = LocalVectorStore(documents, vectors, FakeAtlas())

    found = store.vector_search([0.0, 1.0], limit=2, num_candidates=40)

    assert [(d["_id"], round(d["score"], 2)) for d in found] == [("b", 1.0), ("c", 0.8)]
    assert store.text_search("nms", limit=5) == [{"_id": "kw", "query": "nms", "limit": 5}]
    with pytest.raises(NotImplementedError):
        store.vector_search([0.0, 1.0], limit=2, num_candidates=40, filters={"task": "x"})


def test_strategy_row_and_cell_report_facts_only_when_judged():
    result = StrategyResult("Vector only", ranks=[1, None], evidence=[1.0, 0.0],
                            facts=[FactVerdict((True, True)), FactVerdict((True, False))],
                            latencies=[0.1, 0.3])  # fmt: skip

    assert "| 75.0% | 50.0% |" in result.row(k=6)  # Facts@6, All facts@6
    assert result.cell(1) == "– (0%, 1/2 facts)"
    assert StrategyResult("x", [1], [1.0], [None], [0.1]).row(6).count("– | –") == 1
