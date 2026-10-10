"""Render retrieval evaluation results as Markdown (for humans) and JSON (for tooling)."""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date

from code_assistant.evaluation.dataset import EvalQuestion
from code_assistant.evaluation.retrieval_eval import (
    MRR_DEPTH,
    TIE_MARGIN,
    AtlasAgreement,
    ModelResult,
    bootstrap_ci,
)


@dataclass(frozen=True)
class CorpusInfo:
    tag: str
    commit: str
    chunks: int
    files: int


def _pct(value: float) -> str:
    return "–" if math.isnan(value) else f"{value:.1%}"


def _short(symbol: str) -> str:
    """Drop the common 'ultralytics/' prefix to keep tables readable."""
    return symbol.removeprefix("ultralytics/")


def render_markdown(
    results: Sequence[ModelResult],
    chosen: ModelResult,
    reason: str,
    questions: Sequence[EvalQuestion],
    corpus: CorpusInfo,
    agreement: AtlasAgreement | None = None,
) -> str:
    in_scope = [q for q in questions if q.in_scope]
    by_id = {q.id: q for q in questions}
    lines = [
        "# Retrieval evaluation",
        "",
        f"_Generated on {date.today().isoformat()} by `uv run code-assistant eval retrieval`._",
        "",
        f"- **Corpus:** Ultralytics `{corpus.tag}` (`{corpus.commit[:7]}`), {corpus.files} files, "
        f"{corpus.chunks} chunks.",
        f"- **Questions:** {len(in_scope)} in scope (from real GitHub issues, hand-labelled before "
        f"running any model; {sum(q.mentions_symbol for q in in_scope)} name a symbol or "
        f"argument), plus {len(questions) - len(in_scope)} out-of-scope questions not scored here.",
        "- **Search:** exact cosine over cached vectors; parts of the same symbol are collapsed, "
        "so rankings contain distinct code units.",
        "- **Metrics:** Hit@k = share of questions with at least one relevant symbol in the top k. "
        f"MRR@{MRR_DEPTH} = mean of 1/rank of the first relevant symbol (0 if not in the top "
        f"{MRR_DEPTH}). 95% CI from a 10,000-sample bootstrap over questions.",
        "- **Query latency:** median time to embed one question on CPU.",
        "",
        "## Results",
        "",
        "| Model | Params | Dim | Hit@1 | Hit@5 (95% CI) | MRR@10 | Query latency (CPU) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (-r.hit_at(5), -r.mrr)):
        low, high = bootstrap_ci(r.hits(5))
        marker = " ✅" if r is chosen else ""
        lines.append(
            f"| `{r.model}`{marker} | {r.params_m}M | {r.dimensions} | {_pct(r.hit_at(1))} | "
            f"**{_pct(r.hit_at(5))}** ({_pct(low)}–{_pct(high)}) | {r.mrr:.3f} | "
            f"{r.median_latency_ms:.0f} ms |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"Rule (fixed before running the evaluation): pick the best Hit@5; if a smaller model is "
        f"within {TIE_MARGIN:.0%} of it, pick the smaller model.",
        "",
        f"**Chosen: `{chosen.model}`.** {reason}",
        "",
        f"With {len(in_scope)} questions, one question is worth {1 / len(in_scope):.1%}, and the "
        "confidence intervals above overlap. Small differences between models are not "
        "statistically meaningful, which is why the rule prefers the cheaper model on near-ties.",
        "",
    ]

    categories = sorted({o.category for o in chosen.outcomes})
    lines += [
        "## Hit@5 by category",
        "",
        "| Model | " + " | ".join(f"{c} (n={_count(chosen, c)})" for c in categories) + " |",
        "|---|" + "---|" * len(categories),
    ]
    for r in results:
        cells = [_pct(r.hit_at(5, [o for o in r.outcomes if o.category == c])) for c in categories]
        lines.append(f"| `{r.model}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Hit@5 by question type",
        "",
        "| Model | Names a symbol | Plain language |",
        "|---|---|---|",
    ]
    for r in results:
        named = [o for o in r.outcomes if o.mentions_symbol]
        plain = [o for o in r.outcomes if not o.mentions_symbol]
        lines.append(
            f"| `{r.model}` | {_pct(r.hit_at(5, named))} (n={len(named)}) | "
            f"{_pct(r.hit_at(5, plain))} (n={len(plain)}) |"
        )

    if agreement:
        lines += [
            "",
            "## Atlas approximate search vs exact search",
            "",
            f"For the deployed model (`{agreement.model}`), Atlas `$vectorSearch` "
            f"(numCandidates = 20 × limit) was run on the same questions.",
            "",
            "- Mean overlap of the top-5 with the exact search: "
            f"**{_pct(agreement.overlap_at_5)}**",
            f"- Hit@5 exact: {_pct(agreement.hit_at_5_exact)} · Hit@5 Atlas: "
            f"{_pct(agreement.hit_at_5_atlas)}",
        ]

    misses = [o for o in chosen.outcomes if o.rank is None or o.rank > 5]
    lines += [
        "",
        f"## Failure analysis (`{chosen.model}`)",
        "",
        f"{len(misses)} of {len(chosen.outcomes)} questions have no relevant symbol in the top 5.",
        "",
    ]
    for o in misses:
        q = by_id[o.question_id]
        rank = f"rank {o.rank}" if o.rank else f"not in top {MRR_DEPTH}"
        lines += [
            f"### `{q.id}` ({rank})",
            "",
            f"> {q.question}",
            "",
            f"- Expected: {', '.join(f'`{_short(s)}`' for s in q.relevant)}",
            f"- Retrieved: {', '.join(f'`{_short(s)}`' for s in o.top[:3])}",
            "",
        ]

    all_miss = [
        qid
        for qid in (o.question_id for o in chosen.outcomes)
        if all(
            (out := next(x for x in r.outcomes if x.question_id == qid)).rank is None
            or out.rank > 5
            for r in results
        )
    ]
    lines += [
        "## Questions every model misses",
        "",
        ", ".join(f"`{qid}`" for qid in all_miss) if all_miss else "None.",
        "",
    ]
    return "\n".join(lines)


def _count(result: ModelResult, category: str) -> int:
    return sum(o.category == category for o in result.outcomes)


def render_json(
    results: Sequence[ModelResult],
    chosen: ModelResult,
    reason: str,
    corpus: CorpusInfo,
    agreement: AtlasAgreement | None = None,
) -> str:
    payload = {
        "corpus": asdict(corpus),
        "chosen": chosen.model,
        "reason": reason,
        "models": [
            {
                "model": r.model,
                "params_m": r.params_m,
                "dimensions": r.dimensions,
                "hit_at_1": r.hit_at(1),
                "hit_at_5": r.hit_at(5),
                "hit_at_5_ci95": bootstrap_ci(r.hits(5)),
                "mrr_at_10": r.mrr,
                "median_query_latency_ms": r.median_latency_ms,
                "questions": [asdict(o) for o in r.outcomes],
            }
            for r in results
        ],
        "atlas_agreement": asdict(agreement) if agreement else None,
    }
    return json.dumps(payload, indent=2)
