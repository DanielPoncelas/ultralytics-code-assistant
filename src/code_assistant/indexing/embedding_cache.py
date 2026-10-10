"""On-disk cache of chunk embeddings, keyed by model and corpus content.

Embedding is by far the slowest indexing step (minutes on a GPU, much longer on a CPU).
Caching the vectors means:
- re-running `index` (e.g. after an Atlas reset) does not recompute them;
- several candidate models can be compared offline without re-embedding;
- the cache file can be shipped as a snapshot so others can skip the step entirely.

The key includes a fingerprint of every embedded text plus the exact model revision, so
any change to the source, the chunker or the model invalidates the cache automatically.
"""

import hashlib
import zipfile
from pathlib import Path

import numpy as np

from code_assistant.embeddings import EmbeddingModelSpec


def fingerprint(texts: list[str], spec: EmbeddingModelSpec) -> str:
    digest = hashlib.sha256(f"{spec.name}@{spec.revision}".encode())
    for text in texts:
        digest.update(hashlib.sha256(text.encode()).digest())
    return digest.hexdigest()[:16]


def cache_path(cache_dir: Path, spec: EmbeddingModelSpec, key: str) -> Path:
    slug = spec.name.replace("/", "__")
    return cache_dir / "embeddings" / f"{slug}-{key}.npz"


def load(path: Path, ids: list[str]) -> np.ndarray | None:
    """Return cached vectors, or None if absent or not matching the expected chunk ids."""
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            if data["ids"].tolist() != ids:
                return None
            return data["vectors"]
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):  # corrupt or partial file
        return None


def save(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")  # atomic: never leave a half-written cache
    with partial.open("wb") as f:
        np.savez_compressed(f, ids=np.array(ids), vectors=vectors.astype(np.float32))
    partial.replace(path)
