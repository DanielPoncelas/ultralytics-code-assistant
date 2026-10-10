"""Open-source embedding models, run locally (sentence-transformers) or served by OpenRouter.

Each local model is pinned to an exact Hugging Face revision so that re-indexing produces the
same vectors. Retrieval is asymmetric: queries get an instruction prefix (as each model's
authors recommend), documents are embedded as-is.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx
    import numpy as np
    from sentence_transformers import SentenceTransformer

API_BATCH_SIZE = 16  # chunks per /embeddings request
API_RETRIES = 5  # free models allow ~20 requests per minute: back off on 429
API_BACKOFF_S = 10.0


class EmbeddingError(RuntimeError):
    """An OpenRouter-served embedding model failed (e.g. no credits, provider down)."""


@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str
    revision: str
    params_m: int  # parameters in millions (from the Hugging Face safetensors metadata)
    query_prompt: str
    max_seq_length: int
    trust_remote_code: bool = False
    # Served by OpenRouter's /embeddings API through this provider only, so documents and
    # queries always get vectors from the same deployment. The weights are open, but the
    # hosted revision can't be pinned: `revision` then only labels the cache file.
    openrouter_provider: str | None = None


EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    spec.name: spec
    for spec in [
        # General-purpose, strong on code retrieval, 32k context, Apache-2.0.
        # The task instruction replaces the default "web search" one, as recommended by Qwen.
        EmbeddingModelSpec(
            name="Qwen/Qwen3-Embedding-0.6B",
            revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            params_m=596,
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
            params_m=149,
            query_prompt="",
            max_seq_length=2048,
        ),
        # ModernBERT-based (149M), 8k context, Apache-2.0, strong on code retrieval (CoIR).
        EmbeddingModelSpec(
            name="Alibaba-NLP/gte-modernbert-base",
            revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
            params_m=149,
            query_prompt="",
            max_seq_length=2048,
        ),
        # Small general-purpose baseline (33M), 512-token context, MIT.
        EmbeddingModelSpec(
            name="BAAI/bge-small-en-v1.5",
            revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            params_m=33,
            query_prompt="Represent this sentence for searching relevant passages: ",
            max_seq_length=512,
        ),
        # API-served (D28): the 0.6B model above at 13x the size, with no truncation (32k
        # context) and no local download. Apache-2.0 weights.
        EmbeddingModelSpec(
            name="qwen/qwen3-embedding-8b",
            revision="openrouter",
            params_m=8000,
            query_prompt=(
                "Instruct: Given a question about the Ultralytics YOLO codebase, "
                "retrieve the source code that answers it\nQuery:"
            ),
            max_seq_length=32768,
            openrouter_provider="nebius",
        ),
    ]
}

# The original selection (D18) compared local models only; API models are opt-in (D28).
LOCAL_MODELS = [name for name, spec in EMBEDDING_MODELS.items() if not spec.openrouter_provider]


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

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        http_client: "httpx.Client | None" = None,
    ) -> None:
        if model_name not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown embedding model {model_name!r}. Known: {list(EMBEDDING_MODELS)}"
            )
        self.spec = EMBEDDING_MODELS[model_name]
        self.api = self.spec.openrouter_provider is not None
        self.device = "openrouter" if self.api else resolve_device(device)
        self._model: SentenceTransformer | None = None
        self._http = http_client

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
        if self.api:
            return len(self.embed_query("dimension probe"))
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(
        self, texts: list[str], batch_size: int = 8, show_progress: bool = False
    ) -> "np.ndarray":
        if self.api:
            return self._embed_api(texts, show_progress)
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype("float32")

    def embed_query(self, text: str) -> list[float]:
        if self.api:
            return self._embed_api([self.spec.query_prompt + text])[0].tolist()
        vector = self.model.encode(text, prompt=self.spec.query_prompt, normalize_embeddings=True)
        return vector.tolist()

    def _embed_api(self, texts: list[str], show_progress: bool = False) -> "np.ndarray":
        """Embed through OpenRouter in batches; L2-normalized like the local models."""
        import httpx
        import numpy as np

        if self._http is None:
            from code_assistant.config import get_settings

            settings = get_settings()
            self._http = httpx.Client(base_url=settings.openrouter_base_url, timeout=120)
            key = settings.openrouter_api_key.get_secret_value()
            self._http.headers["Authorization"] = f"Bearer {key}"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), API_BATCH_SIZE):
            payload = {
                "model": self.spec.name,
                "input": texts[start : start + API_BATCH_SIZE],
                "provider": {"order": [self.spec.openrouter_provider], "allow_fallbacks": False},
            }
            try:
                for attempt in range(API_RETRIES):
                    response = self._http.post("/embeddings", json=payload)
                    if response.status_code not in (429, 500, 502, 503):
                        break
                    time.sleep(API_BACKOFF_S * (attempt + 1))
                response.raise_for_status()
            except httpx.HTTPError as exc:
                status = (
                    exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                )
                reason = "needs OpenRouter credits" if status == 402 else "is unavailable"
                detail = f"HTTP {status}" if status else "network error"
                raise EmbeddingError(f"{self.spec.name} {reason} ({detail}).") from exc
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors += [item["embedding"] for item in data]
            if show_progress:
                print(f"  {len(vectors)}/{len(texts)} chunks embedded", flush=True)
        array = np.asarray(vectors, dtype=np.float32)
        return array / np.linalg.norm(array, axis=1, keepdims=True)
