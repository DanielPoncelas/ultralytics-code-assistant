"""Answer-quality evaluation with an LLM judge, and LLM selection.

For each question and candidate LLM, the assistant answers from the same retrieved sources.
A judge model from a different family then grades the answer:

- key-fact coverage: which of the hand-written key facts (from the source code) the
  answer states correctly;
- groundedness: claims that are not supported by the provided sources;
- refusal: whether the answer says it couldn't find the information.

Deterministic checks (citations present and valid) need no judge.
"""

import json
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from code_assistant.llm import Message
from code_assistant.prompt import format_user_message
from code_assistant.retrieval import SearchResult

JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
# Pre-registered LLM selection rule (see choose_llm).
COVERAGE_TIE_MARGIN = 0.05
MIN_OOS_REFUSAL = 0.5
# Added after the first run, when one free candidate was rate-limited upstream for most of it:
# a model that couldn't answer most questions is reported as unavailable, not ranked.
MAX_ERROR_RATE = 0.5

JUDGE_SYSTEM_PROMPT = """\
You are a strict, fair grader of answers produced by a code assistant for the Ultralytics \
YOLO codebase. You receive the question, the source code given to the assistant, a list of \
KEY FACTS that a correct answer must contain (empty for questions the sources cannot \
answer), and the assistant's ANSWER.

Grade as follows:
- facts_covered: for each key fact, in order, true if the answer states it (possibly in \
other words), false if it is missing or stated incorrectly.
- unsupported_claims: specific technical claims in the answer (function names, arguments, \
default values, behaviours) that are NOT supported by the provided sources. Ignore \
generic advice and statements that the information is missing. Quote each briefly.
- refused: true if the answer says, in substance, that it couldn't find the answer in the \
indexed code (a partial answer that only flags a missing detail is NOT a refusal).

Reply with a single JSON object and nothing else:
{"facts_covered": [true, false, ...], "unsupported_claims": ["..."], "refused": false, \
"comment": "one short sentence"}
"""


def build_judge_messages(
    question: str, key_facts: Sequence[str], sources: Sequence[SearchResult], answer: str
) -> list[Message]:
    facts = json.dumps(list(key_facts), ensure_ascii=False)
    user = (
        f"{format_user_message(question, sources)}\n\n"
        f"KEY FACTS: {facts}\n\n"
        f"ANSWER:\n<answer>\n{answer}\n</answer>"
    )
    return [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class Verdict:
    facts_covered: tuple[bool, ...]
    unsupported_claims: tuple[str, ...]
    refused: bool
    comment: str = ""

    @property
    def coverage(self) -> float:
        return statistics.fmean(self.facts_covered) if self.facts_covered else 0.0


class JudgeParseError(ValueError):
    """The judge did not return the expected JSON."""


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(text: str, n_facts: int) -> Verdict:
    """Extract the judge's JSON (tolerating code fences or surrounding prose)."""
    match = _JSON_OBJECT.search(text)
    if not match:
        raise JudgeParseError(f"No JSON object in judge output: {text[:200]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"Invalid JSON from judge: {text[:200]!r}") from exc
    covered = tuple(bool(x) for x in data.get("facts_covered", []))
    if len(covered) != n_facts:
        raise JudgeParseError(f"Expected {n_facts} fact verdicts, got {len(covered)}")
    return Verdict(
        facts_covered=covered,
        unsupported_claims=tuple(str(c) for c in data.get("unsupported_claims", [])),
        refused=bool(data.get("refused", False)),
        comment=str(data.get("comment", "")),
    )


@dataclass(frozen=True)
class AnswerRecord:
    """One graded answer."""

    model: str
    question_id: str
    category: str
    in_scope: bool
    answer: str
    answered_by: str  # model reported by OpenRouter
    latency_s: float
    sources: tuple[str, ...]  # retrieved symbols, in prompt order
    retrieval_hit: bool  # a relevant symbol was among the sources
    citations: tuple[int, ...]
    invalid_citations: tuple[int, ...]
    verdict: Verdict | None
    error: str = ""  # generation or judging failure
    retried: bool = False  # re-run after a transient failure in the first pass

    @property
    def grounded(self) -> bool:
        return self.verdict is not None and not self.verdict.unsupported_claims

    @property
    def citations_ok(self) -> bool:
        return bool(self.citations) and not self.invalid_citations


@dataclass
class LLMResult:
    model: str
    records: list[AnswerRecord] = field(default_factory=list)

    def _scored(self, in_scope: bool) -> list[AnswerRecord]:
        return [r for r in self.records if r.in_scope == in_scope and r.verdict is not None]

    @property
    def errors(self) -> int:
        return sum(1 for r in self.records if r.verdict is None)

    @property
    def coverage(self) -> float:
        """Mean share of key facts covered, in-scope questions (failed answers count as 0)."""
        in_scope = [r for r in self.records if r.in_scope]
        return statistics.fmean(r.verdict.coverage if r.verdict else 0.0 for r in in_scope)

    @property
    def fully_correct(self) -> float:
        in_scope = [r for r in self.records if r.in_scope]
        return statistics.fmean(
            float(r.verdict is not None and all(r.verdict.facts_covered)) for r in in_scope
        )

    @property
    def grounded_rate(self) -> float:
        scored = self._scored(True) + self._scored(False)
        return statistics.fmean(float(r.grounded) for r in scored) if scored else 0.0

    @property
    def citation_rate(self) -> float:
        """In-scope answers that cite at least one source and no non-existent one."""
        in_scope = [r for r in self.records if r.in_scope]
        return statistics.fmean(float(r.citations_ok) for r in in_scope)

    @property
    def oos_refusal_rate(self) -> float | None:
        """Out-of-scope questions correctly answered with 'couldn't find it' (None if none ran)."""
        oos = self._scored(False)
        return statistics.fmean(float(r.verdict.refused) for r in oos) if oos else None

    @property
    def false_refusal_rate(self) -> float:
        """In-scope questions where the right code WAS retrieved, but the answer refused."""
        hits = [r for r in self._scored(True) if r.retrieval_hit]
        return statistics.fmean(float(r.verdict.refused) for r in hits) if hits else 0.0

    @property
    def median_latency_s(self) -> float:
        latencies = [r.latency_s for r in self.records if not r.error]
        return statistics.median(latencies) if latencies else float("nan")


def choose_llm(results: Sequence[LLMResult]) -> tuple[LLMResult, str]:
    """Pre-registered rule.

    1. Ineligible: models that refuse fewer than half of the out-of-scope questions
       (i.e. that tend to invent answers when the code isn't there).
    2. Among the rest, the best key-fact coverage wins; within 5 points of the best, prefer
       the more grounded model (fewer unsupported claims), then the faster one.
    """
    available = [r for r in results if r.errors <= MAX_ERROR_RATE * len(r.records)]
    unavailable = sorted(r.model for r in results if r not in available)
    eligible = [
        r for r in available if r.oos_refusal_rate is None or r.oos_refusal_rate >= MIN_OOS_REFUSAL
    ]
    if not eligible:
        raise ValueError("No model meets the out-of-scope refusal requirement")
    best = max(eligible, key=lambda r: r.coverage)
    close = [r for r in eligible if best.coverage - r.coverage <= COVERAGE_TIE_MARGIN + 1e-9]
    chosen = max(close, key=lambda r: (r.grounded_rate, -r.median_latency_s))
    excluded = sorted(r.model for r in available if r not in eligible)
    reason = (
        f"{chosen.model}: key-fact coverage {chosen.coverage:.1%}, "
        f"grounded {chosen.grounded_rate:.1%}."
    )
    if chosen is not best:
        reason += (
            f" {best.model} covers {best.coverage:.1%}, within {COVERAGE_TIE_MARGIN:.0%}, "
            "so groundedness decided."
        )
    if excluded:
        reason += (
            f" Ineligible (refused < {MIN_OOS_REFUSAL:.0%} of out-of-scope questions): {excluded}."
        )
    if unavailable:
        reason += f" Unavailable (failed > {MAX_ERROR_RATE:.0%} of requests): {unavailable}."
    return chosen, reason
