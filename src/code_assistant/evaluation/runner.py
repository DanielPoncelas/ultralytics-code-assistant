"""Orchestrates the retrieval evaluation: load data, score models, check Atlas, write reports."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from code_assistant.config import Settings
from code_assistant.embeddings import Embedder
from code_assistant.evaluation.dataset import DatasetError, load_questions, unknown_labels
from code_assistant.evaluation.report import CorpusInfo, render_json, render_markdown
from code_assistant.evaluation.retrieval_eval import (
    AtlasAgreement,
    ModelResult,
    choose_model,
    compare_with_atlas,
    evaluate_model,
)
from code_assistant.indexing.pipeline import embed_chunks, load_chunks
from code_assistant.retrieval import Retriever
from code_assistant.store import ChunkStore


@dataclass(frozen=True)
class RetrievalEvaluation:
    results: list[ModelResult]
    chosen: ModelResult
    reason: str
    agreement: AtlasAgreement | None
    markdown_path: Path
    json_path: Path


def run_retrieval_evaluation(
    settings: Settings,
    model_names: Sequence[str],
    questions_path: Path,
    output_dir: Path,
    check_atlas: bool = True,
    log: Callable[[str], None] = print,
) -> RetrievalEvaluation:
    questions = load_questions(questions_path)
    source, files, chunks = load_chunks(settings, log)
    symbols = [f"{c.path}::{c.qualified_name}" for c in chunks]
    missing = unknown_labels(questions, set(symbols))
    if missing:
        raise DatasetError(f"Relevant symbols not found in the index: {missing}")

    results = []
    for name in model_names:
        vectors = embed_chunks(chunks, name, settings, log)  # cached after the first `index`
        embedder = Embedder(name, device="cpu")  # CPU: the latency a typical user will see
        log(f"Evaluating {name} ...")
        result = evaluate_model(
            name, embedder.spec.params_m, questions, vectors, symbols, embedder.embed_query
        )
        log(f"  Hit@1 {result.hit_at(1):.1%} · Hit@5 {result.hit_at(5):.1%} · MRR {result.mrr:.3f}")
        results.append(result)
        del embedder  # free the model before loading the next one

    chosen, reason = choose_model(results)
    agreement = _atlas_agreement(settings, results, questions, log) if check_atlas else None

    corpus = CorpusInfo(source.tag, source.commit, len(chunks), len(files))
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path, json_path = output_dir / "retrieval.md", output_dir / "retrieval.json"
    markdown_path.write_text(render_markdown(results, chosen, reason, questions, corpus, agreement))
    json_path.write_text(render_json(results, chosen, reason, corpus, agreement))
    return RetrievalEvaluation(results, chosen, reason, agreement, markdown_path, json_path)


def _atlas_agreement(settings, results, questions, log) -> AtlasAgreement | None:
    with ChunkStore.from_settings(settings) as store:
        indexed_model = store.indexed_model()
        exact = next((r for r in results if r.model == indexed_model), None)
        if exact is None:
            log(f"Skipping Atlas check: deployed model {indexed_model!r} was not evaluated.")
            return None
        log(f"Comparing Atlas $vectorSearch with exact search for {indexed_model} ...")
        retriever = Retriever(store, Embedder(indexed_model, device="cpu"))
        return compare_with_atlas(
            exact, questions, lambda q: [r.symbol for r in retriever.search(q, k=10)]
        )
