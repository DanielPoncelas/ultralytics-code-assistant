"""Indexing pipeline: fetch source -> chunk -> embed -> store in Atlas."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from code_assistant.config import Settings
from code_assistant.embeddings import Embedder
from code_assistant.indexing import embedding_cache
from code_assistant.indexing.chunker import chunk_source
from code_assistant.indexing.models import Chunk
from code_assistant.indexing.source import fetch_source, iter_python_files
from code_assistant.store import EMBEDDING_FIELD, ChunkStore, SyncResult


@dataclass(frozen=True)
class SourceInfo:
    repo_url: str
    tag: str
    commit: str

    def permalink(self, chunk: Chunk) -> str:
        """GitHub link to the exact lines at the pinned commit (stable forever)."""
        return (
            f"{self.repo_url}/blob/{self.commit}/{chunk.path}#L{chunk.start_line}-L{chunk.end_line}"
        )


@dataclass(frozen=True)
class IndexReport:
    files: int
    chunks: int
    dimensions: int
    upserted: int
    modified: int
    deleted: int
    index_status: str
    seconds: float


def path_metadata(path: str) -> dict[str, str | None]:
    """Derive filterable metadata from the file location.

    ultralytics/models/yolo/segment/train.py -> folder=models, model_family=yolo, task=segment
    """
    parts = path.split("/")
    folder = parts[1]
    model_family = parts[2] if folder == "models" and len(parts) > 3 else None
    task = parts[3] if model_family == "yolo" and len(parts) > 4 else None
    return {"folder": folder, "model_family": model_family, "task": task}


def build_document(
    chunk: Chunk, embedding: list[float], source: SourceInfo, model_name: str
) -> dict[str, Any]:
    return {
        "_id": chunk.id,
        "kind": chunk.kind,
        "path": chunk.path,
        "module": chunk.module,
        "qualified_name": chunk.qualified_name,
        "code": chunk.code,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "signature": chunk.signature,
        "docstring": chunk.docstring,
        "decorators": list(chunk.decorators),
        "parent_class": chunk.parent_class,
        "bases": list(chunk.bases),
        "part": chunk.part,
        "total_parts": chunk.total_parts,
        **path_metadata(chunk.path),
        "github_url": source.permalink(chunk),
        "source": {"repo": source.repo_url, "tag": source.tag, "commit": source.commit},
        "embedding_model": model_name,
        EMBEDDING_FIELD: embedding,
    }


def load_chunks(
    settings: Settings, log: Callable[[str], None] = print
) -> tuple[SourceInfo, list[Path], list[Chunk]]:
    """Fetch the pinned source and chunk every file under the configured directories."""
    source = SourceInfo(
        settings.ultralytics_repo_url, settings.ultralytics_tag, settings.ultralytics_commit
    )
    log(f"Fetching {source.repo_url} @ {source.tag} ...")
    checkout = fetch_source(source.repo_url, source.tag, source.commit, settings.cache_dir)
    files = iter_python_files(checkout, settings.source_dirs)
    chunks = [
        chunk
        for path in files
        for chunk in chunk_source(path.read_text(encoding="utf-8"), str(path.relative_to(checkout)))
    ]
    log(f"Chunked {len(files)} files into {len(chunks)} chunks")
    return source, files, chunks


def embed_chunks(
    chunks: list[Chunk],
    model_name: str,
    settings: Settings,
    log: Callable[[str], None] = print,
) -> np.ndarray:
    """Embed chunks with `model_name`, reusing the on-disk cache when it is still valid."""
    embedder = Embedder(model_name, device=settings.embedding_device)
    texts = [c.embedding_text() for c in chunks]
    ids = [c.id for c in chunks]
    path = embedding_cache.cache_path(
        settings.cache_dir, embedder.spec, embedding_cache.fingerprint(texts, embedder.spec)
    )
    vectors = embedding_cache.load(path, ids)
    if vectors is not None:
        log(f"Loaded cached embeddings from {path}")
        return vectors
    log(f"Embedding {len(texts)} chunks with {model_name} on {embedder.device} ...")
    vectors = embedder.embed_documents(texts, show_progress=True)
    embedding_cache.save(path, ids, vectors)
    log(f"Saved embeddings to {path}")
    return vectors


def run_indexing(
    settings: Settings,
    model_name: str | None = None,
    upload: bool = True,
    log: Callable[[str], None] = print,
) -> IndexReport:
    """Full pipeline. With `upload=False` it only computes (and caches) the embeddings."""
    started = time.monotonic()
    model_name = model_name or settings.embedding_model
    source, files, chunks = load_chunks(settings, log)
    vectors = embed_chunks(chunks, model_name, settings, log)

    result, index_status = SyncResult(0, 0, 0), "skipped"
    if upload:
        documents = [
            build_document(chunk, vector.tolist(), source, model_name)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        store = ChunkStore(
            settings.mongodb_uri.get_secret_value(),
            settings.mongodb_database,
            settings.mongodb_collection,
            settings.vector_index_name,
        )
        try:
            log(f"Writing {len(documents)} documents to Atlas ...")
            result = store.replace_all(documents)
            index_status = store.ensure_vector_index(int(vectors.shape[1]))
            log(f"Vector index {index_status}; waiting until it is queryable ...")
            store.wait_until_queryable()
        finally:
            store.close()

    return IndexReport(
        files=len(files),
        chunks=len(chunks),
        dimensions=int(vectors.shape[1]),
        upserted=result.upserted,
        modified=result.modified,
        deleted=result.deleted,
        index_status=index_status,
        seconds=time.monotonic() - started,
    )
