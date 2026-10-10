"""Fact recall: are the key facts in the sources the LLM receives?

Hit@k and evidence coverage are computed from labels. Hit@k needs only one relevant code unit,
and the labelled evidence lines can miss a fact that a docstring also states. Here a judge
reads the sources exactly as the LLM gets them and says, for each key fact, whether they show
it. A "yes" counts only if the judge quotes a line that really is in the sources, so an
invented justification can't inflate the score. A blind human review of a sample
(`render_fact_review_sheet`, then `eval agreement`) checks the judge itself.
"""

import json
import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from code_assistant.evaluation.answer_eval import JudgeParseError
from code_assistant.evaluation.dataset import EvalQuestion
from code_assistant.llm import LLMError, Message, OpenRouterClient
from code_assistant.prompt import format_user_message
from code_assistant.retrieval import SearchResult

MIN_QUOTE_CHARS = 10  # a quote like "conf" would match almost any source

FACT_JUDGE_PROMPT = """\
You check whether source code retrieved by a code assistant contains the information needed \
to answer a question about the Ultralytics YOLO codebase. You receive the SOURCES, the \
question, and a list of KEY FACTS that a correct answer contains.

For each key fact, in order, decide whether a reader who sees ONLY these sources could state \
it: the sources say it (in code, comments or docstrings) or show it directly (e.g. a default \
value assigned in the code, or one function calling another). General knowledge about \
Ultralytics does not count. A related function that does not show the fact does not count.

- available: true or false.
- quote: if available, copy the single most relevant line from the sources exactly as \
written; otherwise "".

Reply with a single JSON object and nothing else:
{"facts": [{"available": true, "quote": "..."}, ...]}
"""


@dataclass(frozen=True)
class FactVerdict:
    available: tuple[bool, ...]
    rejected_quotes: int = 0  # judged available, but the quote isn't in the sources

    @property
    def share(self) -> float:
        return statistics.fmean(self.available) if self.available else 0.0


def build_fact_messages(
    question: str, key_facts: Sequence[str], sources: Sequence[SearchResult]
) -> list[Message]:
    facts = json.dumps(list(key_facts), ensure_ascii=False)
    user = f"{format_user_message(question, sources)}\n\nKEY FACTS: {facts}"
    return [{"role": "system", "content": FACT_JUDGE_PROMPT}, {"role": "user", "content": user}]


def _squash(text: str) -> str:
    return " ".join(text.split())


def parse_fact_verdict(text: str, n_facts: int, sources: Sequence[SearchResult]) -> FactVerdict:
    """A fact counts as available only if its quote is in the sources (ignoring whitespace)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        facts = json.loads(match.group(0))["facts"] if match else None
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise JudgeParseError(f"Invalid JSON from judge: {text[:200]!r}") from exc
    if not isinstance(facts, list) or len(facts) != n_facts:
        raise JudgeParseError(f"Expected {n_facts} fact verdicts: {text[:200]!r}")
    code = _squash("\n".join(s.code for s in sources))
    claimed = [isinstance(f, dict) and bool(f.get("available")) for f in facts]
    quotes = [_squash(str(f.get("quote", ""))) if isinstance(f, dict) else "" for f in facts]
    verified = [
        yes and len(quote) >= MIN_QUOTE_CHARS and quote in code
        for yes, quote in zip(claimed, quotes, strict=True)
    ]
    return FactVerdict(tuple(verified), rejected_quotes=sum(claimed) - sum(verified))


def judge_facts(
    judge: OpenRouterClient,
    question: EvalQuestion,
    sources: Sequence[SearchResult],
    wait: Callable[[], None],
) -> tuple[FactVerdict | None, str]:
    """(verdict, error). One retry if the judge returns malformed JSON."""
    messages = build_fact_messages(question.question, question.key_facts, sources)
    error = ""
    for _ in range(2):
        wait()
        try:
            text = judge.complete(messages).text
            return parse_fact_verdict(text, len(question.key_facts), sources), ""
        except (LLMError, JudgeParseError) as exc:
            error = f"judge: {exc}"
    return None, error


def render_fact_review_sheet(
    sample: Sequence[tuple[EvalQuestion, Sequence[SearchResult]]], strategy: str
) -> str:
    """Blind sheet (no judge verdicts), in the format `judge_review.parse_review_sheet` reads."""
    lines = [
        "# Fact recall review sheet",
        "",
        f"Blind review of the sources retrieved by `{strategy}` for {len(sample)} questions, "
        "sampled at random (seed 0).",
        "",
        "For each question, tick `[x]` every key fact that a reader of these sources alone "
        "would know: the sources say it or show it directly (e.g. a default value assigned in "
        "the code). Leave `[ ]` otherwise. The judge's verdicts are deliberately not shown. "
        "When done, run `uv run code-assistant eval agreement --sheet "
        "eval/results/fact_review.md --results eval/results/facts.jsonl`.",
        "",
    ]
    for i, (q, sources) in enumerate(sample, start=1):
        lines += [f"## {i}. `{q.id}`", "", f"**Question:** {q.question}", "", "**Key facts:**", ""]
        lines += [f"- [ ] {fact}" for fact in q.key_facts]
        lines += ["", "<details><summary>Sources</summary>", ""]
        for n, s in enumerate(sources, start=1):
            lines += [
                f"**[{n}] `{s.qualified_name}`** ({s.path}, lines {s.start_line}–{s.end_line})",
                "",
                "```python",
                s.code,
                "```",
                "",
            ]
        lines += ["</details>", ""]
    return "\n".join(lines)
