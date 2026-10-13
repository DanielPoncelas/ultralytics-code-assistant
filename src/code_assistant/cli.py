"""Command-line entry point: `uv run code-assistant <command>`."""

import re
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from code_assistant.config import get_settings
from code_assistant.health import check_mongodb, check_openrouter

app = typer.Typer(no_args_is_help=True, add_completion=False)
eval_app = typer.Typer(no_args_is_help=True, help="Evaluate retrieval and answer quality.")
app.add_typer(eval_app, name="eval")

OK = typer.style("✔", fg=typer.colors.GREEN)
FAIL = typer.style("✘", fg=typer.colors.RED)


@app.callback()
def main() -> None:
    """RAG code assistant for the Ultralytics YOLO codebase."""


@app.command()
def check() -> None:
    """Verify configuration and connectivity to MongoDB Atlas and OpenRouter."""
    try:
        settings = get_settings()
    except ValidationError as exc:
        missing = ", ".join(str(err["loc"][0]).upper() for err in exc.errors())
        typer.echo(f"{FAIL} Invalid or missing settings: {missing}")
        typer.echo("  Copy .env.example to .env and fill in the values (see README → Quick start).")
        raise typer.Exit(code=1) from exc
    typer.echo(f"{OK} Settings loaded")

    failed = False

    try:
        version = check_mongodb(settings.mongodb_uri.get_secret_value())
        typer.echo(f"{OK} MongoDB Atlas reachable (server {version})")
    except Exception as exc:  # report any driver error, don't crash
        failed = True
        typer.echo(f"{FAIL} MongoDB Atlas unreachable: {exc}")
        typer.echo("  Check MONGODB_URI and that your IP is in Atlas → Network Access.")

    try:
        info = check_openrouter(
            settings.openrouter_base_url, settings.openrouter_api_key.get_secret_value()
        )
        quota = (
            f"free-model requests today: {info.free_requests_remaining}/{info.free_requests_limit}"
            if info.free_requests_limit is not None
            else "quota unknown"
        )
        typer.echo(f"{OK} OpenRouter key valid ({quota})")
    except Exception as exc:
        failed = True
        typer.echo(f"{FAIL} OpenRouter check failed: {exc}")
        typer.echo("  Check OPENROUTER_API_KEY (openrouter.ai → Keys).")

    if failed:
        raise typer.Exit(code=1)


@app.command()
def index(
    model: Annotated[
        str | None,
        typer.Option(
            help="Embedding model (default: EMBEDDING_MODEL, then ALT_EMBEDDING_MODEL if set)."
        ),
    ] = None,
    upload: Annotated[
        bool, typer.Option(help="Upload to Atlas. --no-upload only computes and caches vectors.")
    ] = True,
) -> None:
    """Fetch the pinned Ultralytics source, chunk it, embed it and store it in Atlas."""
    from code_assistant.indexing.pipeline import run_indexing  # heavy imports (torch)

    settings = get_settings()
    for name in [model] if model else settings.indexed_models:
        report = run_indexing(settings, model_name=name, upload=upload, log=typer.echo)
        summary = (
            f"{report.chunks} chunks from {report.files} files "
            f"({report.dimensions}-d vectors) in {report.seconds:.0f}s"
        )
        if upload:
            summary += (
                f": {report.upserted} new, {report.modified} updated, {report.deleted} removed"
            )
        typer.echo(f"{OK} Indexed {name}: {summary}.")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Natural-language question about the codebase.")],
    k: Annotated[int, typer.Option("-k", "--top-k", help="Number of distinct code units.")] = 5,
    folder: Annotated[str | None, typer.Option(help="Only search models, engine or data.")] = None,
    task: Annotated[str | None, typer.Option(help="Only search a YOLO task, e.g. segment.")] = None,
    kind: Annotated[
        str | None, typer.Option(help="Only return function, method, class or module chunks.")
    ] = None,
    show_code: Annotated[bool, typer.Option(help="Print the code of each result.")] = False,
    rerank: Annotated[bool, typer.Option(help="Reorder the results with RERANK_MODEL.")] = True,
) -> None:
    """Search the indexed code, as the chat does (no LLM involved)."""
    from code_assistant.embeddings import Embedder
    from code_assistant.rerank import OpenRouterReranker
    from code_assistant.retrieval import IndexMismatchError, Retriever, check_index_compatible
    from code_assistant.store import ChunkStore

    settings = get_settings()
    filters = {
        name: value
        for name, value in {"folder": folder, "task": task, "kind": kind}.items()
        if value is not None
    }
    with ChunkStore.from_settings(settings) as store:
        try:
            check_index_compatible(store.indexed_model(), settings.embedding_model)
        except IndexMismatchError as exc:
            typer.echo(f"{FAIL} {exc}")
            raise typer.Exit(code=1) from exc
        embedder = Embedder(settings.embedding_model, device=settings.embedding_device)
        retriever = Retriever(
            store,
            embedder,
            hybrid=settings.hybrid_search,
            reranker=OpenRouterReranker.from_settings(settings),
        )
        results = retriever.search(query, k=k, filters=filters, rerank=rerank)

    if not results:
        typer.echo("No results.")
    for rank, result in enumerate(results, start=1):
        part = f" (part {result.part}/{result.total_parts})" if result.total_parts > 1 else ""
        typer.echo(
            f"{rank}. [{result.score:.3f}] {result.kind:<8} {result.qualified_name}{part}\n"
            f"   {result.path}:{result.start_line}-{result.end_line}\n"
            f"   {result.github_url}"
        )
        if show_code:
            typer.echo(typer.style(result.code, dim=True) + "\n")


def connect_assistant(settings, models: list[str] | None = None):
    """`build_assistant`, turning setup failures into a clear message instead of a traceback."""
    from pymongo.errors import PyMongoError

    from code_assistant.retrieval import IndexMismatchError

    try:
        return build_assistant(settings, models)
    except IndexMismatchError as exc:
        typer.echo(f"{FAIL} {exc}")
        raise typer.Exit(code=1) from exc
    except PyMongoError as exc:  # usually an IP missing from the Atlas access list
        typer.echo(f"{FAIL} MongoDB Atlas unreachable: {str(exc)[:200]}")
        typer.echo("  Check MONGODB_URI and that your IP is in Atlas → Security → Network Access.")
        raise typer.Exit(code=1) from exc


def build_assistant(settings, models: list[str] | None = None):
    """Wire retriever + LLM client from settings (fails fast on an incompatible index)."""
    from code_assistant.assistant import CodeAssistant
    from code_assistant.embeddings import Embedder
    from code_assistant.llm import OpenRouterClient
    from code_assistant.rerank import OpenRouterReranker
    from code_assistant.retrieval import Retriever, check_index_compatible
    from code_assistant.store import ChunkStore

    store = ChunkStore.from_settings(settings)
    try:
        check_index_compatible(store.indexed_model(), settings.embedding_model)
        embedder = Embedder(settings.embedding_model, device=settings.embedding_device)
        llm = OpenRouterClient.from_settings(settings, models=models, interactive=True)
        paid_llm = (
            OpenRouterClient.from_settings(
                settings, models=[settings.paid_llm_model, *llm.models], interactive=True
            )
            if settings.paid_llm_model
            else None
        )
    except Exception:
        store.close()  # don't leak the Mongo client when setup fails
        raise
    reranker = OpenRouterReranker.from_settings(settings)
    retriever = Retriever(store, embedder, hybrid=settings.hybrid_search, reranker=reranker)
    alt_retriever = None
    alt = settings.alt_embedding_model
    if alt and alt != settings.embedding_model:
        alt_store = ChunkStore.from_settings(settings, alt, client=store.client)
        if alt_store.indexed_model() == alt:
            alt_embedder = Embedder(alt, device=settings.embedding_device)
            alt_retriever = Retriever(
                alt_store, alt_embedder, hybrid=settings.hybrid_search, reranker=reranker
            )
        else:
            typer.echo(f"{FAIL} {alt} is not indexed (run `code-assistant index`); not offered.")
    return CodeAssistant(
        retriever,
        llm,
        tag=settings.ultralytics_tag,
        k=settings.retrieval_k,
        paid_llm=paid_llm,
        alt_retriever=alt_retriever,
    ), store


@app.command(name="app")
def run_app(
    host: Annotated[
        str, typer.Option(help="Interface to bind (0.0.0.0 for the network).")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to serve on.")] = 7860,
    share: Annotated[bool, typer.Option(help="Create a temporary public Gradio link.")] = False,
) -> None:
    """Start the chat interface in the browser."""
    import logging
    import os

    # Gradio sends usage analytics by default; a local tool doesn't need to.
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    from code_assistant.app import CSS, FORCE_LIGHT, THEME, build_app

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "sentence_transformers"):  # one log line per HTTP request otherwise
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = get_settings()
    typer.echo("Loading the embedding model and connecting to Atlas ...")
    assistant, store = connect_assistant(settings)
    try:
        assistant.retriever.embedder.embed_query(
            "warm-up"
        )  # load the model before the first question
        tasks = sorted(t for t in store.collection.distinct("task") if t)
        demo = build_app(assistant, settings.ultralytics_tag, tasks)
        typer.echo(f"{OK} Ready. Models (fallback order): {', '.join(settings.llm_models)}")
        demo.launch(
            server_name=host,
            server_port=port,
            share=share,
            theme=THEME,
            css=CSS,
            head=FORCE_LIGHT,
            footer_links=[],
        )
    finally:
        store.close()


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question about the Ultralytics codebase.")],
    model: Annotated[
        list[str] | None,
        typer.Option("--model", help="LLM(s) to use, in fallback order (default: LLM_MODELS)."),
    ] = None,
) -> None:
    """Answer a question with retrieved code and an LLM, citing the sources."""
    from code_assistant.assistant import Answer
    from code_assistant.llm import LLMError

    settings = get_settings()
    assistant, store = connect_assistant(settings, model)
    try:
        answer = None
        for event in assistant.ask_stream(question):
            if isinstance(event, Answer):
                answer = event
            elif isinstance(event, str):
                typer.echo(event, nl=False)
    except LLMError as exc:
        typer.echo(f"\n{FAIL} {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        store.close()

    assert answer is not None
    typer.echo("\n\n" + typer.style("Sources", bold=True))
    for number, source in enumerate(answer.sources, start=1):
        cited = "*" if number in answer.citations else " "
        typer.echo(f" {cited}[{number}] {source.qualified_name}  {source.github_url}")
    fallback = (
        f" after {', '.join(a.model for a in answer.failed_attempts)} failed"
        if answer.failed_attempts
        else ""
    )
    typer.echo(
        typer.style(
            f"\nAnswered by {answer.model}{fallback} in {answer.latency_s:.1f}s. "
            "* = cited in the answer.",
            dim=True,
        )
    )
    if answer.truncated:
        typer.echo(f"{FAIL} The answer was cut off: the model ran out of tokens.")
    if answer.invalid_citations:
        typer.echo(f"{FAIL} Answer cites non-existent sources: {answer.invalid_citations}")


def _demote_headings(markdown: str) -> str:
    """Nest the answer's own headings under the example's '##' heading (code blocks untouched)."""
    lines, in_code = [], False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        elif not in_code:
            line = re.sub(r"^(#{1,4}) ", r"\1## ", line)
        lines.append(line)
    return "\n".join(lines)


def _source_line(number: int, source, cited: bool) -> str:
    bold = "**" if cited else ""
    link = f"{bold}[`{source.qualified_name}`]({source.github_url}){bold}"
    return f"{number}. {link} — `{source.path}` L{source.start_line}-{source.end_line}"


@app.command()
def examples(
    output: Annotated[Path, typer.Option(help="Markdown file to write.")] = Path(
        "docs/examples.md"
    ),
    questions: Annotated[Path, typer.Option(help="Question file.")] = Path("eval/questions.toml"),
) -> None:
    """Answer the example questions (flagged in eval/questions.toml) and save them unedited."""
    from datetime import date

    from code_assistant.evaluation.dataset import load_questions
    from code_assistant.llm import LLMError

    settings = get_settings()
    selected = [q for q in load_questions(questions) if q.example]
    assistant, store = connect_assistant(settings)
    sections = [
        "# Example answers",
        "",
        f"Generated on {date.today().isoformat()} with `uv run code-assistant examples` "
        f"(index: Ultralytics `{settings.ultralytics_tag}`, embeddings: "
        f"`{settings.embedding_model}`, top {settings.retrieval_k} sources). Answers are "
        "reproduced exactly as generated; LLM output varies between runs.",
        "",
    ]
    try:
        for i, q in enumerate(selected, start=1):
            typer.echo(f"[{i}/{len(selected)}] {q.id} ...")
            try:
                answer = assistant.ask(q.question)
            except LLMError as exc:  # keep the other answers
                sections += [f"## {i}. {q.question}", "", f"*No answer: {exc}*", ""]
                continue
            sources = "\n".join(
                _source_line(n, src, cited=n in answer.citations)
                for n, src in enumerate(answer.sources, start=1)
            )
            fallback = "".join(
                f" after `{a.model}` failed ({a.error})" for a in answer.failed_attempts
            )
            truncated = " · ⚠ truncated (token budget ran out)" if answer.truncated else ""
            sections += [
                f"## {i}. {q.question}",
                "",
                f"*Source: [issue #{q.source.rsplit('/', 1)[-1]}]({q.source}) · answered by "
                f"`{answer.model}`{fallback} in {answer.latency_s:.1f} s{truncated}*",
                "",
                _demote_headings(answer.text),
                "",
                "**Retrieved sources** (bold = cited in the answer):",
                "",
                sources,
                "",
            ]
    finally:
        store.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sections), encoding="utf-8")
    typer.echo(f"{OK} Wrote {output}")


@eval_app.command("retrieval")
def eval_retrieval(
    models: Annotated[
        list[str] | None,
        typer.Option(
            "--model", help="Model(s) to evaluate (repeatable). Default: the four local candidates."
        ),
    ] = None,
    questions: Annotated[Path, typer.Option(help="Question file.")] = Path("eval/questions.toml"),
    output: Annotated[Path, typer.Option(help="Report directory.")] = Path("eval/results"),
    atlas: Annotated[
        bool, typer.Option(help="Also compare Atlas $vectorSearch with the exact search.")
    ] = True,
) -> None:
    """Score embedding models on the labelled questions and apply the selection rule."""
    from code_assistant.embeddings import LOCAL_MODELS
    from code_assistant.evaluation.runner import run_retrieval_evaluation

    evaluation = run_retrieval_evaluation(
        get_settings(),
        models or LOCAL_MODELS,
        questions,
        output,
        check_atlas=atlas,
        log=typer.echo,
    )
    typer.echo(f"{OK} Chosen model: {evaluation.chosen.model}. {evaluation.reason}")
    if evaluation.agreement:
        typer.echo(
            f"{OK} Atlas vs exact search: top-5 overlap {evaluation.agreement.overlap_at_5:.1%}"
        )
    typer.echo(f"{OK} Report: {evaluation.markdown_path} (raw data: {evaluation.json_path})")


@eval_app.command("answers")
def eval_answers(
    models: Annotated[
        list[str] | None,
        typer.Option("--model", help="LLM(s) to compare (repeatable). Default: LLM_MODELS."),
    ] = None,
    judge: Annotated[
        str | None, typer.Option(help="Judge model (default: Nemotron-3-Super).")
    ] = None,
    questions: Annotated[Path, typer.Option(help="Question file.")] = Path("eval/questions.toml"),
    output: Annotated[Path, typer.Option(help="Report directory.")] = Path("eval/results"),
    limit: Annotated[int | None, typer.Option(help="Only the first N questions.")] = None,
    retry_failed: Annotated[
        bool, typer.Option(help="Only re-run answers that failed in the existing results.")
    ] = False,
    append: Annotated[
        bool, typer.Option(help="Keep existing results of other models and add these.")
    ] = False,
) -> None:
    """Generate answers with each LLM, grade them with an LLM judge, and choose an LLM.

    Needs ~2 OpenRouter requests per question and model (answer + judge): with 46 questions,
    ~92 per model. Free accounts get 50 free-model requests/day; accounts that purchased at
    least 10 credits get 1,000/day.
    """
    from code_assistant.evaluation.answer_eval import JUDGE_MODEL
    from code_assistant.evaluation.answer_runner import retry_failed_answers, run_answer_evaluation

    settings = get_settings()
    if retry_failed:
        evaluation = retry_failed_answers(
            settings,
            questions,
            output,
            judge_model=judge or JUDGE_MODEL,
            models=models,
            log=typer.echo,
        )
        typer.echo(f"{OK} Chosen LLM: {evaluation.chosen.model}. {evaluation.reason}")
        typer.echo(f"{OK} Report: {evaluation.markdown_path}")
        return
    evaluation = run_answer_evaluation(
        settings,
        models or list(settings.llm_models),
        questions,
        output,
        judge_model=judge or JUDGE_MODEL,
        limit=limit,
        append=append,
        log=typer.echo,
    )
    typer.echo(f"{OK} Chosen LLM: {evaluation.chosen.model}. {evaluation.reason}")
    typer.echo(f"{OK} Report: {evaluation.markdown_path} (raw data: {evaluation.jsonl_path})")


@eval_app.command("review")
def eval_review(
    model: Annotated[
        str | None, typer.Option(help="Model to review (default: first in results).")
    ] = None,
    n: Annotated[int, typer.Option(help="Number of answers to sample.")] = 10,
    results: Annotated[Path, typer.Option(help="Answer results.")] = Path(
        "eval/results/answers.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Sheet to write.")] = Path(
        "eval/results/judge_review.md"
    ),
    force: Annotated[
        bool, typer.Option(help="Overwrite an existing (possibly filled) sheet.")
    ] = False,
) -> None:
    """Write a blind review sheet: a human ticks the key facts each sampled answer states."""
    import json

    from code_assistant.evaluation.dataset import load_questions
    from code_assistant.evaluation.judge_review import render_review_sheet, sample_records

    if output.exists() and not force:
        typer.echo(f"{FAIL} {output} exists (it may contain your ticks). Use --force to overwrite.")
        raise typer.Exit(code=1)
    records = [json.loads(line) for line in results.read_text().splitlines()]
    model = model or records[0]["model"]
    sample = sample_records([r for r in records if r["model"] == model], n=n)
    questions = {q.id: q for q in load_questions()}
    output.write_text(render_review_sheet(sample, questions, model))
    typer.echo(
        f"{OK} Wrote {output} with {len(sample)} answers from {model}. Tick the facts, then "
        "run `uv run code-assistant eval agreement`."
    )


@eval_app.command("agreement")
def eval_agreement(
    sheet: Annotated[Path, typer.Option(help="Filled review sheet.")] = Path(
        "eval/results/judge_review.md"
    ),
    results: Annotated[Path, typer.Option(help="Answer results.")] = Path(
        "eval/results/answers.jsonl"
    ),
) -> None:
    """Compare the human ticks in a review sheet (answers or fact recall) with the judge."""
    import json
    import re

    from code_assistant.evaluation.judge_review import agreement, parse_review_sheet

    text = sheet.read_text()
    model = re.search(r"by `([^`]+)`", text).group(1)  # answers or sources by `<model>`
    human = parse_review_sheet(text)
    judge = {
        r["question_id"]: r["verdict"]["facts_covered"]
        for r in map(json.loads, results.read_text().splitlines())
        if r["model"] == model and r["verdict"] and r["question_id"] in human
    }
    result = agreement(human, judge)
    typer.echo(
        f"{OK} Judge vs human on {result.facts} facts ({len(human)} questions, {model}): "
        f"{result.rate:.1%} agreement, Cohen's kappa {result.kappa:.2f}. "
        f"Judge stricter than human on {result.human_yes_judge_no}, more lenient on "
        f"{result.human_no_judge_yes}."
    )


@eval_app.command("search")
def eval_search(
    questions: Annotated[Path, typer.Option(help="Question file.")] = Path("eval/questions.toml"),
    output: Annotated[Path, typer.Option(help="Report file.")] = Path("eval/results/search.md"),
    facts: Annotated[
        bool,
        typer.Option(
            help="Also ask the LLM judge which key facts the sources show "
            "(~40 free requests per strategy; writes facts.jsonl and a blind review sheet)."
        ),
    ] = False,
    embedding_model: Annotated[
        str | None,
        typer.Option(
            help="Evaluate another embedding model from its saved vectors (exact search) "
            "instead of the deployed Atlas index."
        ),
    ] = None,
) -> None:
    """Compare vector, hybrid and reranked retrieval."""
    from code_assistant.evaluation.search_eval import run_search_evaluation

    run_search_evaluation(
        get_settings(),
        questions,
        output,
        with_facts=facts,
        embedding_model=embedding_model,
        log=typer.echo,
    )
    typer.echo(f"{OK} Report: {output}")
