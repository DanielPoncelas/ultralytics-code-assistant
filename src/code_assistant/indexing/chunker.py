"""Split Python source into semantically meaningful chunks using the stdlib `ast`.

Chunk types:
- function / method: the full definition (decorators, signature, docstring, body).
- class summary: header, docstring, class-level statements and the signature plus first
  docstring line of each method. Method bodies live in their own chunks.
- module summary: module docstring, imports, constants and other top-level statements,
  plus the signature of each top-level function and class.

Nothing is lost: every line of a file ends up in a function/method chunk or in a summary.
Comments between statements (which `ast` does not see) are attached to the statement that
follows them, as in "# NOTE: kept for backward compatibility" above a class.
Units longer than `max_chars` are split at statement boundaries, never mid-statement.
"""

import ast
import textwrap
from collections import Counter
from collections.abc import Sequence

from code_assistant.indexing.models import Chunk, ChunkKind

# ~1,000-1,300 tokens of Python. Keeps chunks focused for retrieval while ~95% of
# functions and methods in the indexed code fit in a single chunk.
MAX_CHUNK_CHARS = 4000
# Lines repeated at the start of a continuation part, so a cut does not strip context.
OVERLAP_LINES = 3

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def chunk_source(
    source: str,
    path: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_lines: int = OVERLAP_LINES,
) -> list[Chunk]:
    """Chunk the contents of one Python file. `path` is the repo-relative file path."""
    builder = _ChunkBuilder(source, path, max_chars, overlap_lines)
    return builder.build()


class _ChunkBuilder:
    def __init__(self, source: str, path: str, max_chars: int, overlap_lines: int) -> None:
        source = source.removeprefix("\ufeff")  # a UTF-8 BOM would make ast.parse fail
        self.tree = ast.parse(source)
        # Split on "\n" only: str.splitlines() also breaks on form feeds and other separators
        # that Python's tokenizer doesn't count, which would shift every line after them.
        self.lines = source.split("\n")
        if self.lines and self.lines[-1] == "":
            self.lines.pop()
        self.path = path
        self.max_chars = max_chars
        self.overlap_lines = overlap_lines
        self.seen: Counter[str] = Counter()

    def build(self) -> list[Chunk]:
        chunks = self._module_summary()
        for node in self.tree.body:
            if isinstance(node, FunctionNode):
                chunks += self._function_chunks(node, kind="function", parent=None)
            elif isinstance(node, ast.ClassDef):
                chunks += self._class_chunks(node)
        return chunks

    # -- unit builders ---------------------------------------------------------------

    def _function_chunks(
        self, node: FunctionNode, kind: ChunkKind, parent: ast.ClassDef | None
    ) -> list[Chunk]:
        qualified_name = f"{parent.name}.{node.name}" if parent else node.name
        start, end = self._start_line(node), node.end_lineno or node.lineno
        boundaries = {
            stmt.lineno
            for stmt in ast.walk(node)
            if isinstance(stmt, ast.stmt) and stmt is not node
        }
        parts = self._split_source_span(start, end, boundaries)
        return self._make_chunks(
            parts,
            kind=kind,
            qualified_name=qualified_name,
            signature=_function_signature(node),
            docstring=ast.get_docstring(node),
            decorators=_decorators(node),
            parent_class=parent.name if parent else None,
        )

    def _class_chunks(self, node: ast.ClassDef) -> list[Chunk]:
        start, end = self._start_line(node), node.end_lineno or node.lineno
        body_indent = " " * node.body[0].col_offset

        # Header: comments, decorators and `class X(...):` line(s), verbatim.
        pieces: list[list[str]] = [self.lines[start - 1 : self._start_line(node.body[0]) - 1]]
        methods: list[FunctionNode] = []
        for stmt in node.body:
            if isinstance(stmt, FunctionNode):
                methods.append(stmt)
                pieces.append(_signature_stub(stmt, body_indent))
            else:  # docstring, attributes, nested classes...: kept verbatim
                pieces.append(self.lines[self._start_line(stmt) - 1 : stmt.end_lineno])

        summary = self._make_chunks(
            self._split_pieces(pieces, start, end),
            kind="class",
            qualified_name=node.name,
            signature=_class_signature(node),
            docstring=ast.get_docstring(node),
            decorators=_decorators(node),
            bases=tuple(ast.unparse(base) for base in node.bases),
        )
        method_chunks = [
            chunk
            for method in methods
            for chunk in self._function_chunks(method, kind="method", parent=node)
        ]
        return summary + method_chunks

    def _module_summary(self) -> list[Chunk]:
        pieces: list[list[str]] = []
        for stmt in self.tree.body:
            if isinstance(stmt, FunctionNode | ast.ClassDef):
                pieces.append(self._leading_comments(stmt))
            if isinstance(stmt, FunctionNode):
                pieces.append(_signature_stub(stmt, indent=""))
            elif isinstance(stmt, ast.ClassDef):
                header = f"class {stmt.name}({', '.join(ast.unparse(b) for b in stmt.bases)}):"
                pieces.append([header, *_docstring_stub(stmt, "    ")])
            else:  # docstring, imports, constants, __all__, if/try blocks...: kept verbatim
                pieces.append(self.lines[self._start_line(stmt) - 1 : stmt.end_lineno])
        pieces = [piece for piece in pieces if piece]
        if not pieces:
            return []
        return self._make_chunks(
            self._split_pieces(pieces, 1, len(self.lines)),
            kind="module",
            qualified_name=self.path.removesuffix(".py")
            .replace("/", ".")
            .removesuffix(".__init__"),
            docstring=ast.get_docstring(self.tree),
        )

    def _start_line(self, node: ast.stmt) -> int:
        """First line of a statement, including its decorators and the comments right above it."""
        first = _decorated_start(node)
        line = first - 1  # 1-based line just above
        while line >= 1 and (
            self.lines[line - 1].strip().startswith("#") or not self.lines[line - 1].strip()
        ):
            line -= 1
        # `line` is now the previous code line (or 0); skip blank lines down to the first comment.
        start = line + 1
        while start < first and not self.lines[start - 1].strip():
            start += 1
        return start

    def _leading_comments(self, node: ast.stmt) -> list[str]:
        """The comment lines attached above a statement (used in summaries)."""
        return self.lines[self._start_line(node) - 1 : _decorated_start(node) - 1]

    # -- splitting -------------------------------------------------------------------

    def _split_source_span(
        self, start: int, end: int, boundaries: set[int]
    ) -> list[tuple[str, int, int]]:
        """Split source lines [start, end] (1-based) into (code, start, end) parts."""
        lines = self.lines[start - 1 : end]
        cut_points = {line - start for line in boundaries if start < line <= end}
        return [
            (textwrap.dedent("\n".join(lines[i:j])), start + i, start + j - 1)
            for i, j in _split_ranges(lines, cut_points, self.max_chars, self.overlap_lines)
        ]

    def _split_pieces(
        self, pieces: Sequence[list[str]], start: int, end: int
    ) -> list[tuple[str, int, int]]:
        """Split a synthesized summary. Every part spans the full source range it summarizes."""
        lines = [line for piece in pieces for line in piece]
        cut_points, offset = set(), 0
        for piece in pieces:
            cut_points.add(offset)
            offset += len(piece)
        return [
            ("\n".join(lines[i:j]), start, end)
            for i, j in _split_ranges(lines, cut_points, self.max_chars, overlap_lines=0)
        ]

    def _make_chunks(
        self, parts: list[tuple[str, int, int]], kind: ChunkKind, qualified_name: str, **metadata
    ) -> list[Chunk]:
        self.seen[qualified_name] += 1
        return [
            Chunk(
                kind=kind,
                path=self.path,
                qualified_name=qualified_name,
                code=code,
                start_line=start,
                end_line=end,
                part=index,
                total_parts=len(parts),
                occurrence=self.seen[qualified_name],
                **metadata,
            )
            for index, (code, start, end) in enumerate(parts, start=1)
        ]


def _split_ranges(
    lines: Sequence[str], cut_points: set[int], max_chars: int, overlap_lines: int
) -> list[tuple[int, int]]:
    """Greedily group lines into [i, j) ranges of at most `max_chars` characters.

    Each range ends right before a cut point (a statement start) when possible. A single
    statement longer than `max_chars` is the only case where a range ends mid-statement.
    """
    ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        size, j = 0, i
        while j < len(lines) and (j == i or size + len(lines[j]) + 1 <= max_chars):
            size += len(lines[j]) + 1
            j += 1
        if j < len(lines):
            # Prefer the latest statement boundary inside the window (but keep progress).
            boundary = max((c for c in cut_points if i < c <= j), default=None)
            if boundary is not None and boundary > i + overlap_lines:
                j = boundary
        ranges.append((i, j))
        if j >= len(lines):
            break
        i = max(j - overlap_lines, i + 1)
    return ranges


# -- AST helpers ---------------------------------------------------------------------


def _decorated_start(node: ast.stmt) -> int:
    """First line of a statement, including its decorators."""
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])


def _decorators(node: FunctionNode | ast.ClassDef) -> tuple[str, ...]:
    return tuple(f"@{ast.unparse(d)}" for d in node.decorator_list)


def _function_signature(node: FunctionNode) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"


def _class_signature(node: ast.ClassDef) -> str:
    args = [ast.unparse(b) for b in node.bases] + [ast.unparse(k) for k in node.keywords]
    return f"class {node.name}({', '.join(args)})" if args else f"class {node.name}"


def _docstring_stub(node: FunctionNode | ast.ClassDef, indent: str) -> list[str]:
    docstring = ast.get_docstring(node)
    if not docstring:
        return []
    first_line = docstring.strip().splitlines()[0]
    return [f'{indent}"""{first_line}"""']


def _signature_stub(node: FunctionNode, indent: str) -> list[str]:
    """`def f(...):` followed by the first docstring line and `...`, as in a type stub."""
    lines = [f"{indent}{d}" for d in _decorators(node)]
    lines.append(f"{indent}{_function_signature(node)}:")
    lines += _docstring_stub(node, indent + "    ")
    lines.append(f"{indent}    ...")
    return lines
