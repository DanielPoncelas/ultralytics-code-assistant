"""Data model for code chunks."""

import hashlib
from dataclasses import dataclass
from typing import Literal

ChunkKind = Literal["module", "class", "function", "method"]


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of source code plus the metadata extracted from its AST.

    `code` is what the LLM and the UI show. `embedding_text()` adds a contextual
    header (file, symbol, signature) so that an isolated method body still
    carries where it lives in the codebase.
    """

    kind: ChunkKind
    path: str  # repo-relative, e.g. "ultralytics/engine/trainer.py"
    qualified_name: str  # e.g. "BaseTrainer._do_train"; the module name for module chunks
    code: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None
    decorators: tuple[str, ...] = ()
    parent_class: str | None = None
    bases: tuple[str, ...] = ()
    part: int = 1
    total_parts: int = 1
    # Disambiguates symbols that share a qualified name in one file (e.g. property getter/setter).
    occurrence: int = 1

    @property
    def module(self) -> str:
        """Dotted module path: 'ultralytics/engine/trainer.py' -> 'ultralytics.engine.trainer'."""
        dotted = self.path.removesuffix(".py").replace("/", ".")
        return dotted.removesuffix(".__init__")

    @property
    def id(self) -> str:
        """Stable identifier: same symbol and part -> same id across re-indexing runs."""
        key = f"{self.path}::{self.qualified_name}::{self.occurrence}::{self.part}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def embedding_text(self) -> str:
        header = [f"# File: {self.path}", f"# Module: {self.module}"]
        if self.kind == "method":
            header.append(f"# Symbol: {self.qualified_name} (method of class {self.parent_class})")
        elif self.kind != "module":
            header.append(f"# Symbol: {self.qualified_name} ({self.kind})")
        if self.kind == "class" and self.bases:
            header.append(f"# Bases: {', '.join(self.bases)}")
        if self.signature:
            header.append(f"# Signature: {self.signature}")
        if self.total_parts > 1:
            header.append(f"# Part: {self.part}/{self.total_parts}")
        return "\n".join(header) + "\n\n" + self.code
