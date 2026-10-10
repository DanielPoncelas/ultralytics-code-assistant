"""Markdown report for the answer-quality evaluation."""

from collections.abc import Sequence
from datetime import date

from code_assistant.config import Settings
from code_assistant.evaluation.answer_eval import (
    COVERAGE_TIE_MARGIN,
    MIN_OOS_REFUSAL,
    AnswerRecord,
    LLMResult,
)
from code_assistant.evaluation.dataset import EvalQuestion


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render_answers_markdown(
    results: Sequence[LLMResult],
    chosen: LLMResult,
    reason: str,
    questions: Sequence[EvalQuestion],
    judge_model: str,
    settings: Settings,
) -> str:
    by_id = {q.id: q for q in questions}
    n_in = sum(q.in_scope for q in questions)
    lines = [
        "# Answer evaluation",
        "",
        f"_Generated on {date.today().isoformat()} by `uv run code-assistant eval answers`._",
        "",
        f"- **Questions:** {n_in} in scope with hand-written key facts (each verifiable in the "
        f"source, with file:line evidence in `eval/questions.toml`), plus "
        f"{len(questions) - n_in} out-of-scope questions that should be declined.",
        f"- **Same context for every model:** top {settings.retrieval_k} chunks retrieved once per "
        f"question with `{settings.embedding_model}`.",
        f"- **Judge:** `{judge_model}` (different model family from every candidate), temperature "
        "0, given the question, the sources, the key facts and the answer.",
        "- **Metrics:**",
        "  - *Coverage*: mean share of key facts stated correctly.",
        "  - *Fully correct*: all key facts stated.",
        "  - *Grounded*: no claim unsupported by the sources, as judged.",
        "  - *Citations OK*: cites at least one source and no non-existent one, checked "
        "deterministically.",
        '  - *OOS refusal*: out-of-scope questions answered with "couldn\'t find it".',
        "  - *False refusal*: refused although the right code was retrieved.",
        "",
        "## Results",
        "",
        "| Model | Coverage | Fully correct | Grounded | Citations OK | OOS refusal | "
        "False refusal | Errors | Median latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: -r.coverage):
        marker = " ✅" if r is chosen else ""
        n_oos = sum(1 for x in r.records if not x.in_scope)
        oos = (
            f"{round(r.oos_refusal_rate * n_oos)}/{n_oos}"
            if r.oos_refusal_rate is not None
            else "–"
        )
        lines.append(
            f"| `{r.model}`{marker} | **{_pct(r.coverage)}** | {_pct(r.fully_correct)} | "
            f"{_pct(r.grounded_rate)} | {_pct(r.citation_rate)} | "
            f"{oos} | {_pct(r.false_refusal_rate)} | "
            f"{r.errors} | {r.median_latency_s:.1f} s |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"Rule (fixed before running): models that decline fewer than {MIN_OOS_REFUSAL:.0%} of "
        "out-of-scope questions are ineligible. Among the rest, the best coverage wins; within "
        f"{COVERAGE_TIE_MARGIN:.0%} of it, the more grounded model wins, then the faster one.",
        "",
        f"**Chosen: `{chosen.model}`.** {reason}",
        "",
    ]

    in_scope = [r for r in chosen.records if r.in_scope]
    retrieval_misses = [r for r in in_scope if not r.retrieval_hit]
    lines += [
        f"## Where answers lose points (`{chosen.model}`)",
        "",
        "| | Questions | Mean coverage |",
        "|---|---|---|",
        _row("Right code retrieved", [r for r in in_scope if r.retrieval_hit]),
        _row("Right code not retrieved", retrieval_misses),
        "",
        "With the right code in context, a remaining gap is a generation problem; without it, "
        "the answer can only be partial and should say what is missing.",
        "",
        "### Incomplete answers",
        "",
    ]
    for r in sorted(in_scope, key=lambda r: r.verdict.coverage if r.verdict else -1):
        if r.verdict and r.verdict.coverage == 1:
            continue
        q = by_id[r.question_id]
        cause = "retrieval miss" if not r.retrieval_hit else "generation"
        lines.append(f"- **`{q.id}`** ({cause}): {_facts_line(q, r)}")
    lines += ["", "### Unsupported claims flagged by the judge", ""]
    flagged = [r for r in chosen.records if r.verdict and r.verdict.unsupported_claims]
    lines += [
        f"- **`{r.question_id}`**: " + "; ".join(f"“{c}”" for c in r.verdict.unsupported_claims)
        for r in flagged
    ] or ["None."]
    lines += ["", "### Out-of-scope questions", ""]
    for r in (r for r in chosen.records if not r.in_scope):
        outcome = "declined ✅" if r.verdict and r.verdict.refused else "answered anyway ❌"
        lines.append(f"- **`{r.question_id}`**: {outcome}")
    errors = [r for res in results for r in res.records if r.error]
    if errors:
        lines += ["", "## Errors", ""]
        lines += [f"- `{r.model}` / `{r.question_id}`: {r.error[:200]}" for r in errors]
    return "\n".join(lines) + "\n"


def _row(label: str, records: list[AnswerRecord]) -> str:
    scored = [r.verdict.coverage for r in records if r.verdict]
    mean = _pct(sum(scored) / len(scored)) if scored else "–"
    return f"| {label} | {len(records)} | {mean} |"


def _facts_line(question: EvalQuestion, record: AnswerRecord) -> str:
    if not record.verdict:
        return f"no verdict ({record.error[:80]})"
    missing = [
        f for f, ok in zip(question.key_facts, record.verdict.facts_covered, strict=True) if not ok
    ]
    return f"{record.verdict.coverage:.0%} covered; missing: " + " / ".join(
        f"“{m}”" for m in missing
    )
