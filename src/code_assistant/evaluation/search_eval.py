"""Compare search strategies on Atlas: vector only, hybrid (RRF) and reranked.

Per strategy, on the in-scope questions:
- Hit@k and MRR: rank of the first relevant code unit (`relevant` labels);
- Evidence@k: share of the labelled evidence lines in the top k (`evidence` labels);
- Facts@k (optional, LLM judge): share of the key facts the top k shows (`key_facts`);
- latency: median search time, including the query embedding and the reranker call.

By default it queries the deployed Atlas index. With `embedding_model`, the vector side is an
exact search over that model's saved chunk vectors instead (`LocalVectorStore`), so embedding
models can be compared without re-uploading each one to Atlas.
"""

import json
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from code_assistant.config import Settings
from code_assistant.embeddings import Embedder
from code_assistant.evaluation.answer_eval import JUDGE_MODEL
from code_assistant.evaluation.answer_runner import (
    JUDGE_MAX_TOKENS,
    MIN_SECONDS_BETWEEN_REQUESTS,
    Throttle,
)
from code_assistant.evaluation.dataset import EvalQuestion, load_questions
from code_assistant.evaluation.fact_recall import (
    FactVerdict,
    judge_facts,
    render_fact_review_sheet,
)
from code_assistant.evaluation.retrieval_eval import (
    evidence_coverage,
    evidence_lines,
    first_relevant_rank,
)
from code_assistant.indexing.pipeline import build_document, embed_chunks, load_chunks
from code_assistant.indexing.source import fetch_source
from code_assistant.llm import OpenRouterClient
from code_assistant.rerank import OpenRouterReranker
from code_assistant.retrieval import Retriever
from code_assistant.store import EMBEDDING_FIELD, ChunkStore

RERANK_INTERVAL_S = 3.5  # stay under the ~20 requests/minute of free OpenRouter models
REVIEW_SAMPLE = 10


class LocalVectorStore:
    """Exact vector search over saved chunk vectors; keyword search still runs on Atlas."""

    def __init__(self, documents: list[dict[str, Any]], vectors: np.ndarray, atlas: ChunkStore):
        self.documents, self.vectors, self.atlas = documents, vectors, atlas

    @classmethod
    def load(cls, settings: Settings, model: str, atlas: ChunkStore) -> "LocalVectorStore":
        source, _, chunks = load_chunks(settings, log=lambda _: None)
        vectors = embed_chunks(chunks, model, settings)  # from embeddings/ when saved
        documents = [build_document(c, [], source, model) for c in chunks]
        for document in documents:
            del document[EMBEDDING_FIELD]
        return cls(documents, vectors, atlas)

    def vector_search(
        self,
        query_vector: list[float],
        limit: int,
        num_candidates: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if filters:
            raise NotImplementedError("The evaluation doesn't use filters")
        scores = self.vectors @ np.asarray(query_vector, dtype=np.float32)  # cosine
        return [
            {**self.documents[i], "score": float(scores[i])} for i in np.argsort(-scores)[:limit]
        ]

    def text_search(
        self, query: str, limit: int, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self.atlas.text_search(query, limit, filters)


@dataclass
class StrategyResult:
    name: str
    ranks: list[int | None] = field(default_factory=list)
    evidence: list[float] = field(default_factory=list)
    facts: list[FactVerdict | None] = field(default_factory=list)  # None: not judged
    latencies: list[float] = field(default_factory=list)
    rerank_failures: int = 0

    def row(self, k: int) -> str:
        n = len(self.ranks)
        hits = [sum(1 for r in self.ranks if r and r <= depth) / n for depth in (1, 5, k)]
        mrr = sum(1 / r for r in self.ranks if r) / n
        full = sum(1 for e in self.evidence if e == 1) / n
        judged = [v for v in self.facts if v]
        facts = (
            f"{statistics.fmean(v.share for v in judged):.1%} | "
            f"{sum(1 for v in judged if all(v.available)) / len(judged):.1%}"
            if judged
            else "– | –"
        )
        return (
            f"| {self.name} | "
            + " | ".join(f"{h:.1%}" for h in hits)
            + f" | {mrr:.3f} | {statistics.fmean(self.evidence):.1%} | {full:.1%} | {facts}"
            + f" | {statistics.median(self.latencies):.2f} s |"
        )

    def cell(self, i: int) -> str:
        verdict = self.facts[i]
        facts = f", {sum(verdict.available)}/{len(verdict.available)} facts" if verdict else ""
        return f"{self.ranks[i] or '–'} ({self.evidence[i]:.0%}{facts})"


def run_search_evaluation(
    settings: Settings,
    questions_path: Path,
    output: Path,
    with_facts: bool = False,
    embedding_model: str | None = None,
    log: Callable[[str], None] = print,
) -> list[StrategyResult]:
    questions = [q for q in load_questions(questions_path) if q.in_scope]
    checkout = fetch_source(
        settings.ultralytics_repo_url,
        settings.ultralytics_tag,
        settings.ultralytics_commit,
        settings.cache_dir,
    )
    evidence = {q.id: evidence_lines(q.evidence, checkout) for q in questions}
    k = settings.retrieval_k
    model = embedding_model or settings.embedding_model
    embedder = Embedder(model, device="cpu")
    embedder.embed_query("warm-up")  # exclude model loading from the latency
    reranker = OpenRouterReranker.from_settings(settings)
    judge = None
    if with_facts:
        judge = OpenRouterClient.from_settings(settings, models=[JUDGE_MODEL], max_retries=4)
        judge.max_tokens = JUDGE_MAX_TOKENS
    judge_throttle = Throttle(MIN_SECONDS_BETWEEN_REQUESTS)
    sample = random.Random(0).sample(questions, min(REVIEW_SAMPLE, len(questions)))
    results: list[StrategyResult] = []
    records: list[dict] = []
    with ChunkStore.from_settings(settings) as atlas:
        store = LocalVectorStore.load(settings, model, atlas) if embedding_model else atlas
        strategies = [
            ("Vector only", Retriever(store, embedder)),
            ("Hybrid (vector + keyword, RRF)", Retriever(store, embedder, hybrid=True)),
        ]
        if reranker:
            name = f"Reranked ({reranker.model})"
            strategies.append((name, Retriever(store, embedder, reranker=reranker)))
        for name, retriever in strategies:
            result = StrategyResult(name)
            throttle = Throttle(RERANK_INTERVAL_S if retriever.reranker else 0)
            review = []  # sources of the sampled questions; the last (default) strategy's are kept
            for q in questions:
                throttle.wait()
                started = time.perf_counter()
                found = retriever.search(q.question, k=10)
                result.latencies.append(time.perf_counter() - started)
                sources = found[:k]
                result.ranks.append(first_relevant_rank([r.symbol for r in found], q.relevant))
                result.evidence.append(evidence_coverage(sources, evidence[q.id]))
                verdict = None
                if judge:
                    verdict, error = judge_facts(judge, q, sources, judge_throttle.wait)
                    records.append(_record(name, q, verdict, error))
                result.facts.append(verdict)
                if q in sample:
                    review.append((q, sources))
            result.rerank_failures = retriever.rerank_failures
            results.append(result)
            log(result.row(k))
    if judge:
        judge.close()
        if not embedding_model:  # the judge is reviewed on the deployed setup only
            _write_judgements(records, review, results[-1].name, output.parent, log)
    report = render_search_report(results, questions, settings, embedding_model)
    output.write_text(report, encoding="utf-8")
    return results


def _record(name: str, q: EvalQuestion, verdict: FactVerdict | None, error: str) -> dict:
    """Same shape as the answer records, so `eval agreement` can compare a review with it."""
    return {
        "model": name,
        "question_id": q.id,
        "verdict": {"facts_covered": list(verdict.available)} if verdict else None,
        "rejected_quotes": verdict.rejected_quotes if verdict else 0,
        "error": error,
    }


def _write_judgements(
    records: list[dict],
    review: list[tuple[EvalQuestion, list]],
    strategy: str,
    directory: Path,
    log: Callable[[str], None],
) -> None:
    (directory / "facts.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    sheet = directory / "fact_review.md"
    if sheet.exists():
        log(f"Kept {sheet} (it may contain your ticks); delete it to sample a new one.")
    else:
        sheet.write_text(render_fact_review_sheet(review, strategy), encoding="utf-8")
        log(f"Wrote the blind review sheet {sheet}.")


def render_search_report(
    results: Sequence[StrategyResult],
    questions: Sequence[EvalQuestion],
    settings: Settings,
    embedding_model: str | None = None,
) -> str:
    k = settings.retrieval_k
    if embedding_model:
        setup = "exact search over its saved vectors, keyword search on Atlas"
        command = f"eval search --facts --embedding-model {embedding_model}"
    else:
        setup, command = "live Atlas queries", "eval search --facts"
    notes = []
    for result in results:
        if result.rerank_failures:
            notes.append(
                f"⚠ {result.name}: the reranker failed on {result.rerank_failures} questions, "
                "which fell back to vector order. Re-run for clean numbers."
            )
        judged = [v for v in result.facts if v]
        if judged and len(judged) < len(result.facts):
            notes.append(
                f"⚠ {result.name}: the judge failed on {len(result.facts) - len(judged)} "
                "questions, which are left out of Facts."
            )
        if rejected := sum(v.rejected_quotes for v in judged):
            notes.append(
                f"{result.name}: {rejected} facts the judge called available were rejected "
                "because its quote isn't in the sources."
            )
    per_question = "\n".join(
        f"| `{q.id}` | " + " | ".join(r.cell(i) for r in results) + " |"
        for i, q in enumerate(questions)
    )
    return (
        "# Search strategy evaluation\n\n"
        f"{len(questions)} in-scope questions, `{embedding_model or settings.embedding_model}`, "
        f"{setup} (generated by `uv run code-assistant {command}`). The LLM sees the top "
        f"{k}. From most lenient to strictest:\n\n"
        f"- **Hit@{k}:** at least one relevant code unit is in the top {k}.\n"
        f"- **Evidence@{k}:** share of the labelled evidence lines (`evidence` in "
        f"`eval/questions.toml`) in the top {k}; Full evidence counts the questions where "
        "all of it is.\n"
        f"- **Facts@{k}:** share of the key facts that the top {k} shows, judged by "
        f"`{JUDGE_MODEL}` from the sources alone and backed by a quote that must be in them; "
        "All facts counts the questions where every key fact is (the answer is fully there).\n"
        "- **Latency:** median search time per question, including the query embedding on CPU "
        "and, for the reranker, the OpenRouter call.\n\n"
        + "".join(f"{note}\n\n" for note in notes)
        + f"| Search | Hit@1 | Hit@5 | Hit@{k} | MRR@10 | Evidence@{k} | Full evidence@{k} "
        f"| Facts@{k} | All facts@{k} | Latency |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
        + "\n".join(r.row(k) for r in results)
        + "\n\n## Per question\n\n"
        "Rank of the first relevant code unit (– = not in the top 10), then in brackets the "
        f"evidence in the top {k} and the key facts it shows.\n\n"
        f"| Question | {' | '.join(r.name for r in results)} |\n"
        f"|---|{'---|' * len(results)}\n" + per_question + "\n"
    )
