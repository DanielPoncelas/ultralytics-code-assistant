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


@dataclass(frozen=True)
class SyncResult:
    upserted: int
    modified: int
    deleted: int


class ChunkStore:
    def __init__(self, uri: str, database: str, collection: str, index_name: str) -> None:
        self._client: MongoClient = MongoClient(uri)
        self.collection = self._client[database][collection]
        self.index_name = index_name

    @classmethod
    def from_settings(cls, settings: "Settings") -> "ChunkStore":
        return cls(
            settings.mongodb_uri.get_secret_value(),
            settings.mongodb_database,
            settings.mongodb_collection,
            settings.vector_index_name,
        )

    def __enter__(self) -> "ChunkStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

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

    def ensure_vector_index(self, dimensions: int) -> str:
        """Create the vector search index, or update it if its definition changed."""
        definition = vector_index_definition(dimensions)
        existing = next(iter(self.collection.list_search_indexes(self.index_name)), None)
        if existing is None:
            model = SearchIndexModel(
                definition=definition, name=self.index_name, type="vectorSearch"
            )
            self.collection.create_search_index(model)
            return "created"
        if existing.get("latestDefinition") != definition:
            self.collection.update_search_index(self.index_name, definition)
            return "updated"
        return "unchanged"

    def wait_until_queryable(
        self, dimensions: int, timeout_s: float = 300, poll_s: float = 5
    ) -> None:
        """Atlas builds search indexes asynchronously; block until ours serves `dimensions`.

        After an update the old definition can still report READY, so the definition is
        checked too; a FAILED build raises immediately instead of waiting for the timeout.
        """
        expected = vector_index_definition(dimensions)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            index = next(iter(self.collection.list_search_indexes(self.index_name)), None)
            if index and index.get("status") == "FAILED":
                raise RuntimeError(f"Vector index {self.index_name!r} build failed: {index}")
            if (
                index
                and index.get("queryable")
                and index.get("status") == "READY"
                and index.get("latestDefinition") == expected
            ):
                return
            time.sleep(poll_s)
        raise TimeoutError(f"Vector index {self.index_name!r} not ready after {timeout_s}s")


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
