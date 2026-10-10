"""Local, open-source embedding models via sentence-transformers.

Each supported model is pinned to an exact Hugging Face revision so that re-indexing
produces the same vectors. Retrieval is asymmetric: queries get an instruction prefix
(as each model's authors recommend), documents are embedded as-is.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str
    revision: str
    query_prompt: str
    max_seq_length: int
    trust_remote_code: bool = False


EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    spec.name: spec
    for spec in [
        # General-purpose, strong on code retrieval, 32k context, Apache-2.0.
        # The task instruction replaces the default "web search" one, as recommended by Qwen.
        EmbeddingModelSpec(
            name="Qwen/Qwen3-Embedding-0.6B",
            revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            query_prompt=(
                "Instruct: Given a question about the Ultralytics YOLO codebase, "
                "retrieve the source code that answers it\nQuery:"
            ),
            max_seq_length=2048,
        ),
        # ModernBERT-based (149M), 8k context, Apache-2.0, trained with code retrieval data.
        EmbeddingModelSpec(
            name="ibm-granite/granite-embedding-english-r2",
            revision="47ea694b257b703fee9253d75c2b1f2985180498",
            query_prompt="",
            max_seq_length=2048,
        ),
        # ModernBERT-based (149M), 8k context, Apache-2.0, strong on code retrieval (CoIR).
        EmbeddingModelSpec(
            name="Alibaba-NLP/gte-modernbert-base",
            revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
            query_prompt="",
            max_seq_length=2048,
        ),
        # Small general-purpose baseline (33M), 512-token context, MIT.
        EmbeddingModelSpec(
            name="BAAI/bge-small-en-v1.5",
            revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            query_prompt="Represent this sentence for searching relevant passages: ",
            max_seq_length=512,
        ),
    ]
}


def resolve_device(requested: str | None = None) -> str:
    """Use the requested device, else the fastest available: NVIDIA GPU, Apple GPU, CPU."""
    if requested:
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Thin wrapper that applies the model-specific query prompt and L2-normalizes vectors."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        if model_name not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown embedding model {model_name!r}. Known: {list(EMBEDDING_MODELS)}"
            )
        self.spec = EMBEDDING_MODELS[model_name]
        self.device = resolve_device(device)
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> "SentenceTransformer":
        # Lazy: loading the model takes seconds, and many commands never need it.
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Half precision on NVIDIA halves memory (fits 4 GB cards) with negligible
            # effect on retrieval; CPU and Apple GPU stay in float32 for numerical safety.
            dtype = "float16" if self.device == "cuda" else "float32"
            self._model = SentenceTransformer(
                self.spec.name,
                revision=self.spec.revision,
                trust_remote_code=self.spec.trust_remote_code,
                device=self.device,
                model_kwargs={"dtype": dtype},
            )
            self._model.max_seq_length = self.spec.max_seq_length
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(
        self, texts: list[str], batch_size: int = 8, show_progress: bool = False
    ) -> "np.ndarray":
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype("float32")

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(text, prompt=self.spec.query_prompt, normalize_embeddings=True)
        return vector.tolist()
