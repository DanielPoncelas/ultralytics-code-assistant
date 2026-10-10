"""Connectivity checks for the external services the assistant depends on."""

from dataclasses import dataclass

import httpx
from pymongo import MongoClient

TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class OpenRouterKeyInfo:
    is_free_tier: bool
    free_requests_remaining: int | None
    free_requests_limit: int | None


def check_mongodb(uri: str) -> str:
    """Ping the cluster and return its server version. Raises on failure."""
    with MongoClient(uri, serverSelectionTimeoutMS=TIMEOUT_SECONDS * 1000) as client:
        client.admin.command("ping")
        return client.server_info()["version"]


def check_openrouter(base_url: str, api_key: str) -> OpenRouterKeyInfo:
    """Validate the API key and return its free-model quota. Raises on failure."""
    response = httpx.get(
        f"{base_url}/key",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()["data"]
    quota = data.get("free_model_daily_requests") or {}
    return OpenRouterKeyInfo(
        is_free_tier=data.get("is_free_tier", False),
        free_requests_remaining=quota.get("remaining"),
        free_requests_limit=quota.get("limit"),
    )
