"""Application settings loaded from environment variables and the `.env` file.

Settings are validated once at startup so a misconfiguration fails fast with a
clear message instead of surfacing later as an obscure network error.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Credentials. `SecretStr` keeps them out of reprs, logs and tracebacks.
    mongodb_uri: SecretStr
    openrouter_api_key: SecretStr

    # MongoDB layout.
    mongodb_database: str = "code_assistant"
    mongodb_collection: str = "chunks"
    vector_index_name: str = "chunks_vector_index"

    # Source code to index, pinned to an exact release for reproducibility.
    ultralytics_repo_url: str = "https://github.com/ultralytics/ultralytics"
    ultralytics_tag: str = "v8.4.175"
    ultralytics_commit: str = "69c95d1add13a232382e53b3aa5094ec0c7738db"
    source_dirs: tuple[str, ...] = ("ultralytics/models", "ultralytics/engine", "ultralytics/data")
    cache_dir: Path = Path(".cache")

    # Embedding model (must be a key of `embeddings.EMBEDDING_MODELS`).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # "cuda", "mps" or "cpu". Empty = auto-detect the fastest available device.
    embedding_device: str | None = None

    # OpenRouter endpoint.
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Answer generation. Models are tried in order; the next one is used if a model is
    # rate-limited, unavailable or returns an empty answer. Comma-separated in the env.
    llm_models: Annotated[tuple[str, ...], NoDecode] = (
        "poolside/laguna-s-2.1:free",
        "cohere/north-mini-code:free",
        "google/gemma-4-31b-it:free",
    )
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000
    llm_timeout_s: float = 90.0
    # Chunks given to the LLM: Hit@k plateaus at 6 (80%) in the retrieval evaluation.
    retrieval_k: int = 6

    @field_validator("llm_models", mode="before")
    @classmethod
    def _split_models(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(model.strip() for model in value.split(",") if model.strip())
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first use."""
    return Settings()  # type: ignore[call-arg]  # values come from env / .env
