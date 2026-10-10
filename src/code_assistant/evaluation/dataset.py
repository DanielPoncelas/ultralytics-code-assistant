"""Hand-labelled evaluation questions (see eval/questions.toml)."""

import tomllib
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_QUESTIONS = Path("eval/questions.toml")


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    category: str
    source: str
    relevant: tuple[str, ...] = ()  # "<path>::<qualified name>" of chunks that answer it
    in_scope: bool = True
    mentions_symbol: bool = False
    example: bool = False
    notes: str = ""
    # What a correct answer must state, written from the source code (for answer evaluation),
    # and where each fact can be checked ("<path>:<lines>").
    key_facts: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


class DatasetError(ValueError):
    """The question file is inconsistent."""


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[EvalQuestion]:
    with path.open("rb") as f:
        raw = tomllib.load(f)["question"]
    allowed = {f.name for f in fields(EvalQuestion)}
    questions = []
    for entry in raw:
        unknown = set(entry) - allowed
        if unknown:
            raise DatasetError(f"{entry.get('id')}: unknown field(s) {sorted(unknown)}")
        tuples = {
            name: tuple(entry.get(name, [])) for name in ("relevant", "key_facts", "evidence")
        }
        questions.append(EvalQuestion(**{**entry, **tuples}))
    validate(questions)
    return questions


def validate(questions: list[EvalQuestion]) -> None:
    duplicates = [qid for qid, n in Counter(q.id for q in questions).items() if n > 1]
    if duplicates:
        raise DatasetError(f"Duplicate question ids: {duplicates}")
    for q in questions:
        if q.in_scope and not q.relevant:
            raise DatasetError(f"{q.id}: in-scope questions need at least one relevant symbol")
        if not q.in_scope and (q.relevant or q.key_facts):
            raise DatasetError(f"{q.id}: out-of-scope questions must not list relevant code/facts")


def unknown_labels(questions: list[EvalQuestion], known_symbols: set[str]) -> dict[str, list[str]]:
    """Relevant symbols that don't exist in the index (typos, or code moved in a new release)."""
    missing = {q.id: [s for s in q.relevant if s not in known_symbols] for q in questions}
    return {qid: symbols for qid, symbols in missing.items() if symbols}
