"""Command-line entry point: `uv run code-assistant <command>`."""

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
        typer.echo("  Copy .env.example to .env and fill in the values (see README → Setup).")
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
        str | None, typer.Option(help="Embedding model (default: EMBEDDING_MODEL setting).")
    ] = None,
    upload: Annotated[
        bool, typer.Option(help="Upload to Atlas. --no-upload only computes and caches vectors.")
    ] = True,
) -> None:
    """Fetch the pinned Ultralytics source, chunk it, embed it and store it in Atlas."""
    from code_assistant.indexing.pipeline import run_indexing  # heavy imports (torch)

    report = run_indexing(get_settings(), model_name=model, upload=upload, log=typer.echo)
    summary = (
        f"{report.chunks} chunks from {report.files} files "
        f"({report.dimensions}-d vectors) in {report.seconds:.0f}s"
    )
    if upload:
        summary += f": {report.upserted} new, {report.modified} updated, {report.deleted} removed"
    typer.echo(f"{OK} Indexed {summary}.")


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
) -> None:
    """Semantic search over the indexed code (no LLM involved)."""
    from code_assistant.embeddings import Embedder
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
        results = Retriever(store, embedder).search(query, k=k, filters=filters)

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


def build_assistant(settings, models: list[str] | None = None):
    """Wire retriever + LLM client from settings (fails fast on an incompatible index)."""
    from code_assistant.assistant import CodeAssistant
    from code_assistant.embeddings import Embedder
    from code_assistant.llm import OpenRouterClient
    from code_assistant.retrieval import Retriever, check_index_compatible
    from code_assistant.store import ChunkStore

    store = ChunkStore.from_settings(settings)
    check_index_compatible(store.indexed_model(), settings.embedding_model)
    retriever = Retriever(
        store, Embedder(settings.embedding_model, device=settings.embedding_device)
    )
    llm = OpenRouterClient.from_settings(settings, models=models)
    return CodeAssistant(
        retriever, llm, tag=settings.ultralytics_tag, k=settings.retrieval_k
    ), store


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
    from code_assistant.retrieval import IndexMismatchError

    settings = get_settings()
    try:
        assistant, store = build_assistant(settings, model)
    except IndexMismatchError as exc:
        typer.echo(f"{FAIL} {exc}")
        raise typer.Exit(code=1) from exc
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
    if answer.invalid_citations:
        typer.echo(f"{FAIL} Answer cites non-existent sources: {answer.invalid_citations}")


@eval_app.command("retrieval")
def eval_retrieval(
    models: Annotated[
        list[str] | None,
        typer.Option("--model", help="Model(s) to evaluate (repeatable). Default: all candidates."),
    ] = None,
    questions: Annotated[Path, typer.Option(help="Question file.")] = Path("eval/questions.toml"),
    output: Annotated[Path, typer.Option(help="Report directory.")] = Path("eval/results"),
    atlas: Annotated[
        bool, typer.Option(help="Also compare Atlas $vectorSearch with the exact search.")
    ] = True,
) -> None:
    """Score embedding models on the labelled questions and apply the selection rule."""
    from code_assistant.embeddings import EMBEDDING_MODELS
    from code_assistant.evaluation.runner import run_retrieval_evaluation

    evaluation = run_retrieval_evaluation(
        get_settings(),
        models or list(EMBEDDING_MODELS),
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
