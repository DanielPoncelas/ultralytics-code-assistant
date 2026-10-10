"""Fetch the Ultralytics source at a pinned release and verify its commit."""

import subprocess
from pathlib import Path


class SourceVerificationError(RuntimeError):
    """The checked-out commit does not match the pinned one."""


def fetch_source(repo_url: str, tag: str, expected_commit: str, cache_dir: Path) -> Path:
    """Shallow-clone `repo_url` at `tag` into the cache (once) and return the checkout path.

    The commit is verified on every call: a moved tag or a corrupted cache fails loudly
    instead of silently indexing different code.
    """
    checkout = cache_dir / f"ultralytics-{tag}"
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _git(
            "-c", "advice.detachedHead=false",
            "clone", "--quiet", "--depth", "1", "--branch", tag, repo_url, str(checkout),
        )  # fmt: skip

    actual_commit = _git("-C", str(checkout), "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise SourceVerificationError(
            f"{checkout} is at commit {actual_commit}, expected {expected_commit} for {tag}. "
            f"Delete the folder to re-clone, or update the pinned commit."
        )
    return checkout


def iter_python_files(checkout: Path, source_dirs: tuple[str, ...]) -> list[Path]:
    """All `.py` files under the configured directories, in a deterministic order."""
    missing = [d for d in source_dirs if not (checkout / d).is_dir()]
    if missing:
        raise FileNotFoundError(f"Source directories not found in {checkout}: {missing}")
    return sorted({path for d in source_dirs for path in (checkout / d).rglob("*.py")})


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()
