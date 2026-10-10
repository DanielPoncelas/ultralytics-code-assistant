"""Orchestrates the answer evaluation: retrieve once, generate per LLM, judge, report."""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from code_assistant.config import Settings
from code_assistant.embeddings import Embedder
from code_assistant.evaluation.answer_eval import (
    JUDGE_MODEL,
    AnswerRecord,
    JudgeParseError,
    LLMResult,
    build_judge_messages,
    choose_llm,
    parse_verdict,
)
from code_assistant.evaluation.answer_report import render_answers_markdown
from code_assistant.evaluation.dataset import DatasetError, EvalQuestion, load_questions
from code_assistant.llm import LLMError, OpenRouterClient
from code_assistant.prompt import build_messages, extract_citations
from code_assistant.retrieval import Retriever, SearchResult, check_index_compatible
from code_assistant.store import ChunkStore

# OpenRouter allows 20 free-model requests per minute; stay safely below it.
MIN_SECONDS_BETWEEN_REQUESTS = 3.5
JUDGE_MAX_TOKENS = 4000  # the judge is a reasoning model: leave room to think and answer


class Throttle:
    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def wait(self) -> None:
        delay = self._last + self.min_interval_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


@dataclass(frozen=True)
class AnswerEvaluation:
    results: list[LLMResult]
    chosen: LLMResult
    reason: str
    markdown_path: Path
    jsonl_path: Path


def run_answer_evaluation(
    settings: Settings,
    models: Sequence[str],
    questions_path: Path,
    output_dir: Path,
    judge_model: str = JUDGE_MODEL,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> AnswerEvaluation:
    questions = load_questions(questions_path)[:limit]
    missing = [q.id for q in questions if q.in_scope and not q.key_facts]
    if missing:
        raise DatasetError(f"In-scope questions without key_facts: {missing}")

    sources = _retrieve_all(settings, questions, log)
    throttle = Throttle(MIN_SECONDS_BETWEEN_REQUESTS)
    judge = OpenRouterClient.from_settings(settings, models=[judge_model], max_retries=4)
    judge.max_tokens = JUDGE_MAX_TOKENS

    results = []
    for model in models:
        log(f"Generating and judging answers with {model} ...")
        llm = OpenRouterClient.from_settings(settings, models=[model], max_retries=4)
        result = LLMResult(model)
        for i, q in enumerate(questions, start=1):
            record = _answer_and_grade(llm, judge, throttle, settings, q, sources[q.id])
            result.records.append(record)
            status = (
                f"ERROR ({record.error[:120]})"
                if record.error
                else f"coverage {record.verdict.coverage:.0%}"
            )
            log(f"  [{i}/{len(questions)}] {q.id}: {status}")
        llm.close()
        log(
            f"  coverage {result.coverage:.1%} · grounded {result.grounded_rate:.1%} · "
            f"errors {result.errors}"
        )
        results.append(result)
    judge.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path, jsonl_path = output_dir / "answers.md", output_dir / "answers.jsonl"
    with jsonl_path.open("w") as f:  # raw data first, so nothing is lost if a later step fails
        for result in results:
            for record in result.records:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    chosen, reason = choose_llm(results)
    markdown_path.write_text(
        render_answers_markdown(results, chosen, reason, questions, judge_model, settings)
    )
    return AnswerEvaluation(results, chosen, reason, markdown_path, jsonl_path)


def _retrieve_all(
    settings: Settings, questions: Sequence[EvalQuestion], log: Callable[[str], None]
) -> dict[str, list[SearchResult]]:
    """Retrieve once per question so every LLM answers from identical sources."""
    log(f"Retrieving sources for {len(questions)} questions ...")
    with ChunkStore.from_settings(settings) as store:
        check_index_compatible(store.indexed_model(), settings.embedding_model)
        embedder = Embedder(settings.embedding_model, device=settings.embedding_device)
        retriever = Retriever(store, embedder)
        return {q.id: retriever.search(q.question, k=settings.retrieval_k) for q in questions}


def _answer_and_grade(
    llm: OpenRouterClient,
    judge: OpenRouterClient,
    throttle: Throttle,
    settings: Settings,
    question: EvalQuestion,
    sources: list[SearchResult],
) -> AnswerRecord:
    base = {
        "model": llm.models[0],
        "question_id": question.id,
        "category": question.category,
        "in_scope": question.in_scope,
        "sources": tuple(s.symbol for s in sources),
        "retrieval_hit": any(s.symbol in question.relevant for s in sources),
    }
    throttle.wait()
    try:
        completion = llm.complete(
            build_messages(question.question, sources, settings.ultralytics_tag)
        )
    except LLMError as exc:
        return AnswerRecord(
            **base,
            answer="",
            answered_by="",
            latency_s=0.0,
            citations=(),
            invalid_citations=(),
            verdict=None,
            error=f"generation: {exc}",
        )

    citations = tuple(extract_citations(completion.text))
    invalid = tuple(n for n in citations if not 1 <= n <= len(sources))
    verdict, error = None, ""
    messages = build_judge_messages(question.question, question.key_facts, sources, completion.text)
    for _ in range(2):  # one retry if the judge returns malformed JSON
        throttle.wait()
        try:
            verdict = parse_verdict(judge.complete(messages).text, len(question.key_facts))
            error = ""
            break
        except (LLMError, JudgeParseError) as exc:
            error = f"judge: {exc}"
    return AnswerRecord(
        **base,
        answer=completion.text,
        answered_by=completion.model,
        latency_s=completion.latency_s,
        citations=citations,
        invalid_citations=invalid,
        verdict=verdict,
        error=error,
    )
