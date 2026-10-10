"""MongoDB Atlas storage for chunks and their vector search index."""

import time
from dataclasses import dataclass
from itertools import batched
from typing import TYPE_CHECKING, Any

from pymongo import MongoClient, ReplaceOne
from pymongo.operations import SearchIndexModel

if TYPE_CHECKING:
    from code_assistant.config import Settings

EMBEDDING_FIELD = "embedding"
# Metadata fields usable as pre-filters in $vectorSearch (e.g. only search `task: "segment"`).
FILTER_FIELDS = ("folder", "kind", "model_family", "task")

TEXT_INDEX_NAME = "chunks_text_index"
# Keyword index. The "code" analyzer splits identifiers into words, so a question about "nms"
# matches `agnostic_nms`, `NMSModel` or `self.args.nms` (whitespace + wordDelimiterGraph).
TEXT_INDEX_DEFINITION: dict[str, Any] = {
    "analyzers": [
        {
            "name": "code",
            "tokenizer": {"type": "whitespace"},
            "tokenFilters": [
                {
                    "type": "wordDelimiterGraph",
                    "delimiterOptions": {
                        "generateWordParts": True,
                        "splitOnCaseChange": True,
                        "preserveOriginal": True,
                    },
                },
                {"type": "lowercase"},
            ],
        }
    ],
    "mappings": {
        "dynamic": False,
        "fields": {
            "qualified_name": {"type": "string", "analyzer": "code"},
            "code": {"type": "string", "analyzer": "code"},
            "docstring": {"type": "string", "analyzer": "lucene.english"},
            **{field: {"type": "token"} for field in FILTER_FIELDS},
        },
    },
}


@dataclass(frozen=True)
class SyncResult:
    upserted: int
    modified: int
    deleted: int


class ChunkStore:
    def __init__(
        self,
        uri: str,
        database: str,
        collection: str,
        index_name: str,
        text_collection: str | None = None,
        client: MongoClient | None = None,
    ) -> None:
        self.client: MongoClient = client or MongoClient(uri)
        self.collection = self.client[database][collection]
        # Keyword search may use another collection's text index: a second embedding model's
        # collection holds the same chunks, and M0 clusters allow only 3 search indexes.
        self.text_collection = self.client[database][text_collection or collection]
        self.index_name = index_name

    @classmethod
    def from_settings(
        cls, settings: "Settings", model: str | None = None, client: MongoClient | None = None
    ) -> "ChunkStore":
        """The collection of `model` (default EMBEDDING_MODEL); `client` shares a connection."""
        return cls(
            settings.mongodb_uri.get_secret_value(),
            settings.mongodb_database,
            settings.collection_for(model or settings.embedding_model),
            settings.vector_index_name,
            text_collection=settings.mongodb_collection,
            client=client,
        )

    def __enter__(self) -> "ChunkStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    def vector_search(
        self,
        query_vector: list[float],
        limit: int,
        num_candidates: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Approximate nearest-neighbour search. Returns documents (without vectors) + `score`."""
        stage: dict[str, Any] = {
            "index": self.index_name,
            "path": EMBEDDING_FIELD,
            "queryVector": query_vector,
            "numCandidates": num_candidates,
            "limit": limit,
        }
        if filters:
            stage["filter"] = filters
        pipeline = [
            {"$vectorSearch": stage},
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
            {"$unset": EMBEDDING_FIELD},  # 1,024 floats per hit we don't need to transfer
        ]
        return list(self.collection.aggregate(pipeline))

    def indexed_model(self) -> str | None:
        """Embedding model the stored vectors were built with (None if the collection is empty)."""
        doc = self.collection.find_one({}, {"embedding_model": 1})
        return doc["embedding_model"] if doc else None

    def replace_all(self, documents: list[dict[str, Any]], batch_size: int = 200) -> SyncResult:
        """Make the collection contain exactly `documents` (idempotent upsert by `_id`)."""
        upserted = modified = 0
        for batch in batched(documents, batch_size):
            ops = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in batch]
            result = self.collection.bulk_write(ops, ordered=False)
            upserted += result.upserted_count
            modified += result.modified_count
        # Chunks that no longer exist (e.g. a function removed in a newer release).
        ids = [doc["_id"] for doc in documents]
        deleted = self.collection.delete_many({"_id": {"$nin": ids}}).deleted_count
        return SyncResult(upserted=upserted, modified=modified, deleted=deleted)

    def text_search(
        self, query: str, limit: int, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Keyword (BM25) search over symbol names, docstrings and code. Same shape as vectors."""
        compound: dict[str, Any] = {
            "should": [
                {
                    "text": {
                        "query": query,
                        "path": "qualified_name",
                        "score": {"boost": {"value": 3}},
                    }
                },
                {"text": {"query": query, "path": "docstring", "score": {"boost": {"value": 2}}}},
                {"text": {"query": query, "path": "code"}},
            ],
            "minimumShouldMatch": 1,
        }
        if filters:
            compound["filter"] = [{"equals": {"path": f, "value": v}} for f, v in filters.items()]
        pipeline = [
            {"$search": {"index": TEXT_INDEX_NAME, "compound": compound}},
            {"$limit": limit},
            {"$set": {"score": {"$meta": "searchScore"}}},
            {"$unset": EMBEDDING_FIELD},
        ]
        return list(self.text_collection.aggregate(pipeline))

    def ensure_vector_index(self, dimensions: int) -> str:
        return self._ensure_index(
            self.index_name, vector_index_definition(dimensions), "vectorSearch"
        )

    def ensure_text_index(self) -> str:
        return self._ensure_index(TEXT_INDEX_NAME, TEXT_INDEX_DEFINITION, "search")

    def wait_until_queryable(self, dimensions: int, timeout_s: float = 300) -> None:
        self._wait(self.index_name, vector_index_definition(dimensions), timeout_s)

    def wait_until_text_index_ready(self, timeout_s: float = 300) -> None:
        self._wait(TEXT_INDEX_NAME, TEXT_INDEX_DEFINITION, timeout_s)

    def _ensure_index(self, name: str, definition: dict[str, Any], kind: str) -> str:
        """Create the search index, or update it if its definition changed."""
        existing = next(iter(self.collection.list_search_indexes(name)), None)
        if existing is None:
            model = SearchIndexModel(definition=definition, name=name, type=kind)
            self.collection.create_search_index(model)
            return "created"
        if not _contains(existing.get("latestDefinition"), definition):
            self.collection.update_search_index(name, definition)
            return "updated"
        return "unchanged"

    def _wait(
        self, name: str, definition: dict[str, Any], timeout_s: float, poll_s: float = 5
    ) -> None:
        """Atlas builds search indexes asynchronously; block until ours serves `definition`.

        After an update the old definition can still report READY, so the definition is
        checked too; a FAILED build raises immediately instead of waiting for the timeout.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            index = next(iter(self.collection.list_search_indexes(name)), None)
            if index and index.get("status") == "FAILED":
                raise RuntimeError(f"Search index {name!r} build failed: {index}")
            if (
                index
                and index.get("queryable")
                and index.get("status") == "READY"
                and _contains(index.get("latestDefinition"), definition)
            ):
                return
            time.sleep(poll_s)
        raise TimeoutError(f"Search index {name!r} not ready after {timeout_s}s")


def _contains(actual: Any, expected: Any) -> bool:
    """True if `expected` is a subset of `actual` (Atlas may add defaults to a definition)."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            _contains(actual.get(k), v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_contains(a, e) for a, e in zip(actual, expected, strict=True))
        )
    return actual == expected


def vector_index_definition(dimensions: int) -> dict[str, Any]:
    return {
        "fields": [
            {
                "type": "vector",
                "path": EMBEDDING_FIELD,
                "numDimensions": dimensions,
                # Vectors are L2-normalized, so cosine == dot product; cosine is the safe default.
                "similarity": "cosine",
            },
            *({"type": "filter", "path": field} for field in FILTER_FIELDS),
        ]
    }
