"""Semantic search over the indexed code chunks."""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from code_assistant.store import FILTER_FIELDS

logger = logging.getLogger(__name__)

# Atlas recommends numCandidates >= 20x limit for results close to an exact search.
CANDIDATES_PER_RESULT = 20
MAX_CANDIDATES = 10_000  # Atlas upper bound
# Split units can return several parts of the same symbol; fetch extra so that k distinct
# symbols remain after deduplication.
OVERFETCH = 3


class IndexMismatchError(RuntimeError):
    """The query embedding model differs from the one used to build the index."""


@dataclass(frozen=True)
class SearchResult:
    id: str
    kind: str
    path: str
    qualified_name: str
    code: str
    start_line: int
    end_line: int
    github_url: str
    score: float
    signature: str | None = None
    docstring: str | None = None
    part: int = 1
    total_parts: int = 1

    @property
    def symbol(self) -> str:
        """Stable reference to the code unit: '<path>::<qualified name>'."""
        return f"{self.path}::{self.qualified_name}"

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> "SearchResult":
        return cls(
            id=doc["_id"],
            kind=doc["kind"],
            path=doc["path"],
            qualified_name=doc["qualified_name"],
            code=doc["code"],
            start_line=doc["start_line"],
            end_line=doc["end_line"],
            github_url=doc["github_url"],
            score=doc["score"],
            signature=doc.get("signature"),
            docstring=doc.get("docstring"),
            part=doc.get("part", 1),
            total_parts=doc.get("total_parts", 1),
        )


def deduplicate[T](items: Iterable[T], key: Callable[[T], str]) -> list[T]:
    """Keep the first (best-ranked) item per key, preserving order.

    Used to collapse several parts of the same split function into one result, so the
    top-k contains k *different* code units.
    """
    seen: set[str] = set()
    unique = []
    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            unique.append(item)
    return unique


def build_filter(filters: dict[str, str] | None) -> dict[str, Any] | None:
    """Translate {'task': 'segment'} into an Atlas $vectorSearch pre-filter."""
    if not filters:
        return None
    unknown = set(filters) - set(FILTER_FIELDS)
    if unknown:
        raise ValueError(f"Unsupported filter field(s) {sorted(unknown)}; use {FILTER_FIELDS}")
    clauses = [{field: {"$eq": value}} for field, value in filters.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def text_search(
        self, query: str, limit: int, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...

    def vector_search(
        self,
        query_vector: list[float],
        limit: int,
        num_candidates: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


class Retriever:
    """Hybrid retrieval: semantic (vectors) + keyword (BM25), merged with Reciprocal Rank Fusion.

    Vectors capture meaning ("stop training early"); keywords catch exact identifiers that an
    embedding dilutes ("nms", "rect", "close_mosaic"). If keyword search fails (e.g. the text
    index doesn't exist yet), results are vector-only.
    """

    def __init__(self, store: VectorStore, embedder: QueryEmbedder, hybrid: bool = False) -> None:
        self.store = store
        self.embedder = embedder
        self.hybrid = hybrid

    def search(
        self, query: str, k: int = 5, filters: dict[str, str] | None = None
    ) -> list[SearchResult]:
        """Return the k most relevant distinct code units for a natural-language query."""
        limit = k * OVERFETCH
        documents = self.store.vector_search(
            self.embedder.embed_query(query),
            limit=limit,
            num_candidates=min(limit * CANDIDATES_PER_RESULT, MAX_CANDIDATES),
            filters=build_filter(filters),
        )
        semantic = deduplicate([SearchResult.from_document(d) for d in documents], key=_symbol)
        if not self.hybrid:
            return semantic[:k]
        try:
            keyword_docs = self.store.text_search(query, limit=limit, filters=filters)
        except Exception as exc:  # degrade to vector-only rather than fail the question
            logger.warning("Keyword search failed, using vector search only: %s", exc)
            return semantic[:k]
        keyword = deduplicate([SearchResult.from_document(d) for d in keyword_docs], key=_symbol)
        return reciprocal_rank_fusion([semantic, keyword])[:k]


def _symbol(result: SearchResult) -> str:
    return result.symbol


RRF_K = 60  # standard constant from Cormack et al. (2009); dampens the weight of top ranks


def reciprocal_rank_fusion(rankings: list[list[SearchResult]]) -> list[SearchResult]:
    """Merge ranked lists: score(d) = sum over lists of 1 / (RRF_K + rank). Ties keep list order.

    Uses ranks only, so the incomparable scales of cosine similarity and BM25 don't matter.
    A result keeps its semantic score (shown in the UI) when it appears in the vector list.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, SearchResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.symbol] = scores.get(result.symbol, 0.0) + 1 / (RRF_K + rank)
            first_seen.setdefault(result.symbol, result)
    order = sorted(scores, key=lambda symbol: -scores[symbol])  # stable: earlier lists win ties
    return [first_seen[symbol] for symbol in order]


def check_index_compatible(indexed_model: str | None, query_model: str) -> None:
    """Fail fast if queries would be embedded with a different model than the index."""
    if indexed_model is None:
        raise IndexMismatchError("The collection is empty. Run `code-assistant index` first.")
    if indexed_model != query_model:
        raise IndexMismatchError(
            f"The index was built with {indexed_model!r} but EMBEDDING_MODEL is {query_model!r}. "
            "Vectors from different models are not comparable: re-run `code-assistant index` "
            "or change EMBEDDING_MODEL."
        )
