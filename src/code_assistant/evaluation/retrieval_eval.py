"""Offline retrieval evaluation and embedding model selection.

Every candidate model is evaluated with an *exact* (brute-force cosine) search over its
cached chunk vectors. This isolates the quality of the model from the approximate search
of the vector database and needs no re-upload per model. A separate check
(`compare_with_atlas`) verifies that Atlas' approximate search agrees with the exact one
for the deployed model.
"""

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from code_assistant.evaluation.dataset import EvalQuestion

MRR_DEPTH = 10
# Pre-registered decision rule: best Hit@5 wins, unless a smaller model is within this margin.
TIE_MARGIN = 0.03


@dataclass(frozen=True)
class QuestionOutcome:
    question_id: str
    category: str
    mentions_symbol: bool
    rank: int | None  # 1-based rank of the first relevant symbol, None if not in the top 10
    top: tuple[str, ...]  # retrieved symbols, best first


@dataclass
class ModelResult:
    model: str
    params_m: int
    dimensions: int
    outcomes: list[QuestionOutcome]
    query_latency_ms: list[float] = field(default_factory=list)

    def hits(self, k: int, outcomes: Sequence[QuestionOutcome] | None = None) -> list[float]:
        return [float(o.rank is not None and o.rank <= k) for o in (outcomes or self.outcomes)]

    def hit_at(self, k: int, outcomes: Sequence[QuestionOutcome] | None = None) -> float:
        return statistics.fmean(self.hits(k, outcomes))

    @property
    def mrr(self) -> float:
        return statistics.fmean(1 / o.rank if o.rank else 0.0 for o in self.outcomes)

    @property
    def median_latency_ms(self) -> float:
        return statistics.median(self.query_latency_ms) if self.query_latency_ms else float("nan")


def rank_symbols(
    query_vector: np.ndarray, matrix: np.ndarray, symbols: Sequence[str], depth: int = MRR_DEPTH
) -> list[str]:
    """Exact search: rank chunks by cosine similarity, collapse parts, return distinct symbols."""
    scores = matrix @ query_vector  # vectors are L2-normalized, so dot product == cosine
    ranked: list[str] = []
    for index in np.argsort(-scores):
        symbol = symbols[index]
        if symbol not in ranked:
            ranked.append(symbol)
            if len(ranked) == depth:
                break
    return ranked


def first_relevant_rank(ranked: Sequence[str], relevant: Sequence[str]) -> int | None:
    for position, symbol in enumerate(ranked, start=1):
        if symbol in relevant:
            return position
    return None


def evaluate_model(
    model_name: str,
    params_m: int,
    questions: Sequence[EvalQuestion],
    matrix: np.ndarray,
    symbols: Sequence[str],
    embed_query: Callable[[str], list[float]],
) -> ModelResult:
    """Score one model on the in-scope questions. `embed_query` is timed per question."""
    embed_query("warm-up")  # exclude one-off initialisation from latency
    result = ModelResult(
        model=model_name, params_m=params_m, dimensions=matrix.shape[1], outcomes=[]
    )
    for q in questions:
        if not q.in_scope:
            continue
        started = time.perf_counter()
        vector = np.asarray(embed_query(q.question), dtype=np.float32)
        result.query_latency_ms.append((time.perf_counter() - started) * 1000)
        ranked = rank_symbols(vector, matrix, symbols)
        result.outcomes.append(
            QuestionOutcome(
                question_id=q.id,
                category=q.category,
                mentions_symbol=q.mentions_symbol,
                rank=first_relevant_rank(ranked, q.relevant),
                top=tuple(ranked),
            )
        )
    return result


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of `values`."""
    rng = np.random.default_rng(seed)
    data = np.asarray(values)
    means = rng.choice(data, size=(n_resamples, len(data)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def choose_model(
    results: Sequence[ModelResult], margin: float = TIE_MARGIN
) -> tuple[ModelResult, str]:
    """Apply the pre-registered rule and explain the outcome."""
    best = max(results, key=lambda r: (r.hit_at(5), r.mrr))
    eligible = [r for r in results if best.hit_at(5) - r.hit_at(5) <= margin + 1e-9]
    chosen = min(eligible, key=lambda r: (r.params_m, -r.hit_at(5), -r.mrr))
    if chosen is best:
        reason = (
            f"{best.model} has the best Hit@5 ({best.hit_at(5):.1%}) and no smaller model is "
            f"within {margin:.0%}."
        )
    else:
        reason = (
            f"{best.model} has the best Hit@5 ({best.hit_at(5):.1%}), but {chosen.model} is within "
            f"{margin:.0%} ({chosen.hit_at(5):.1%}) with {chosen.params_m}M instead of "
            f"{best.params_m}M parameters, so the smaller model wins."
        )
    return chosen, reason


@dataclass(frozen=True)
class AtlasAgreement:
    model: str
    overlap_at_5: float  # mean |exact top-5 ∩ Atlas top-5| / 5
    hit_at_5_exact: float
    hit_at_5_atlas: float


def compare_with_atlas(
    exact: ModelResult,
    questions: Sequence[EvalQuestion],
    atlas_search: Callable[[str], list[str]],
) -> AtlasAgreement:
    """Check that Atlas' approximate search returns what the exact search returns."""
    by_id = {q.id: q for q in questions}
    overlaps, atlas_hits = [], []
    for outcome in exact.outcomes:
        atlas_top = atlas_search(by_id[outcome.question_id].question)
        overlaps.append(len(set(outcome.top[:5]) & set(atlas_top[:5])) / 5)
        rank = first_relevant_rank(atlas_top, by_id[outcome.question_id].relevant)
        atlas_hits.append(float(rank is not None and rank <= 5))
    return AtlasAgreement(
        model=exact.model,
        overlap_at_5=statistics.fmean(overlaps),
        hit_at_5_exact=exact.hit_at(5),
        hit_at_5_atlas=statistics.fmean(atlas_hits),
    )
