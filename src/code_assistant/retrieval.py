"""Semantic search over the indexed code chunks."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from code_assistant.store import FILTER_FIELDS

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
    def vector_search(
        self,
        query_vector: list[float],
        limit: int,
        num_candidates: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


class Retriever:
    def __init__(self, store: VectorStore, embedder: QueryEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(
        self, query: str, k: int = 5, filters: dict[str, str] | None = None
    ) -> list[SearchResult]:
        """Return the k most similar distinct code units for a natural-language query."""
        vector = self.embedder.embed_query(query)
        limit = k * OVERFETCH
        documents = self.store.vector_search(
            vector,
            limit=limit,
            num_candidates=min(limit * CANDIDATES_PER_RESULT, MAX_CANDIDATES),
            filters=build_filter(filters),
        )
        results = [SearchResult.from_document(doc) for doc in documents]
        return deduplicate(results, key=lambda r: r.symbol)[:k]


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
