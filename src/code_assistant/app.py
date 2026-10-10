"""Gradio chat interface.

Design: the code used for an answer is shown the way YOLO draws a detection, as a box with a
coloured label tab (symbol name and relevance score, like class name and confidence),
using Ultralytics' own annotation palette. Each [n] citation in the answer is a link to the
exact lines on GitHub, coloured like its source box.

Security: the answer text is LLM output, so Gradio's HTML sanitisation stays on and
citations are plain Markdown links (coloured with CSS attribute selectors, not inline
HTML). Everything rendered in the sources panel is HTML-escaped.
"""

import html
import logging
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import gradio as gr
from pymongo.errors import PyMongoError

from code_assistant.assistant import Answer, CodeAssistant
from code_assistant.llm import LLMError
from code_assistant.prompt import map_prose, replace_citations
from code_assistant.retrieval import SearchResult

logger = logging.getLogger(__name__)

# Subset of ultralytics.utils.plotting.Colors (the palette YOLO draws detections with),
# skipping the near-white and the navy that is used for text.
# Only colours that reach WCAG AA (4.5:1) with white or navy label text.
PALETTE = ("042AFF", "0BDBEB", "00DFB7", "FF6FDD", "FC6D2F", "BD00FF")

EXAMPLES = {
    "Resume an interrupted training run": "My training run was interrupted at epoch 63. How do I "
    "resume it from the last checkpoint, and what state gets restored?",
    "Default augmentations": "Which data augmentations are applied by default when training a "
    "detection model?",
    "val() vs predict()": "Why do model.val() and model.predict() give different detections on "
    "the same images?",
    "ONNX export with NMS": "How can I export a model to ONNX with a dynamic batch size and NMS "
    "included in the graph?",
    "Crop detected objects": "How do I crop the detected objects out of an image and save them "
    "as separate files?",
}

ANY = "Any"
NO_ANSWER = "No model could answer right now. The free models may be busy: try again in a minute."
Chat = list[dict[str, Any]]


# -- pure helpers (unit-tested) ------------------------------------------------------------


def source_color(number: int) -> str:
    return f"#{PALETTE[(number - 1) % len(PALETTE)]}"


def _luminance(hex_color: str) -> float:
    channels = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def label_text_color(hex_color: str) -> str:
    """White or navy, whichever contrasts more with the swatch (WCAG)."""
    return max(("#FFFFFF", "#111F68"), key=lambda ink: contrast(hex_color, ink))


def citation_url(source: SearchResult, number: int) -> str:
    """GitHub permalink with a `?source=n` marker that the CSS uses to colour the citation.

    GitHub ignores unknown query parameters, so the link still opens the exact lines.
    """
    base, _, fragment = source.github_url.partition("#")
    return f"{base}?source={number}#{fragment}"


def linkify_citations(text: str, sources: Sequence[SearchResult]) -> str:
    """Turn [n] citations into links to source n (same parsing rules as the metrics).

    A run with a number that matches no source stays plain text, so a wrong citation is
    visible rather than silently linked somewhere.
    """

    def link(numbers: list[int], original: str) -> str:
        if not all(1 <= n <= len(sources) for n in numbers):
            return original
        return "".join(f"[{n}]({citation_url(sources[n - 1], n)})" for n in numbers)

    return replace_citations(text, link)


_IMAGE = re.compile(r"!\[")


def render_answer(text: str, sources: Sequence[SearchResult]) -> str:
    """Model output -> safe Markdown for the chat.

    Gradio's sanitiser removes scripts but keeps style attributes, forms and remote images,
    which model output could use to cover the page or to send data out. Outside code, `<`
    is escaped (no raw HTML) and images become plain links (nothing is fetched).
    """

    def neutralise(prose: str) -> str:
        return _IMAGE.sub("[", prose.replace("<", "&lt;"))

    return linkify_citations(map_prose(text, neutralise), sources)


def render_sources(
    sources: Sequence[SearchResult], cited: Sequence[int] = (), answered: bool = False
) -> str:
    """HTML for the sources panel. All values are escaped."""
    if not sources:
        return (
            '<div class="sources-empty">The code used for each answer appears here, '
            "with links to the exact lines on GitHub.</div>"
        )
    boxes = []
    for number, source in enumerate(sources, start=1):
        color = source_color(number)
        state = "cited" if number in cited else ("uncited" if answered else "retrieved")
        part = f" (part {source.part} of {source.total_parts})" if source.total_parts > 1 else ""
        boxes.append(
            f'<article class="detection {state}" style="--box: {color}; '
            f'--label-ink: {label_text_color(color)}">'
            f'<header class="detection-label"><span>[{number}] '
            f"{html.escape(source.qualified_name)}</span>"
            f'<span class="detection-score">{source.score:.3f}</span></header>'
            f'<p class="detection-meta"><span class="detection-path">'
            f"{html.escape(source.path)}, lines {source.start_line}–{source.end_line}{part}."
            f'</span> <a href="{html.escape(source.github_url)}" target="_blank" '
            'rel="noopener">View on GitHub</a></p>'
            f"<details><summary>Show code</summary><pre><code>{html.escape(source.code)}"
            "</code></pre></details></article>"
        )
    cited = [n for n in cited if 1 <= n <= len(sources)]
    if answered:
        note = (
            f"{len(cited)} of {len(sources)} retrieved code units are cited in the answer. "
            "Dashed boxes were retrieved but not used."
            if cited
            else "The answer doesn't cite any of the retrieved code."
        )
    else:
        note = "Retrieved code, most relevant first."
    return f'<p class="sources-note">{note}</p>' + "".join(boxes)


def render_status(answer: Answer | None = None, error: str | None = None, stage: str = "") -> str:
    if error:
        return f'<p class="status status-error">{html.escape(error)}</p>'
    if stage:
        return f'<p class="status status-busy">{html.escape(stage)}</p>'
    if answer is None:
        return ""
    lines = [f"Answered by {html.escape(answer.model)} in {answer.latency_s:.1f} s."]
    if answer.search_query and answer.search_query != answer.question:
        lines.append(f"Searched for: “{html.escape(answer.search_query)}”.")
    if answer.failed_attempts:
        failed = ", ".join(
            f"{html.escape(a.model)} ({html.escape(_short_error(a.error))})"
            for a in answer.failed_attempts
        )
        lines.append(f"Tried first: {failed}.")
    if answer.truncated:
        lines.append("The answer was cut off because the model ran out of tokens.")
    if answer.invalid_citations:
        lines.append(
            "The answer cites sources that don't exist: "
            + ", ".join(f"[{n}]" for n in answer.invalid_citations)
            + "."
        )
    return '<p class="status">' + " ".join(lines) + "</p>"


def _short_error(error: str) -> str:
    if "429" in error:
        return "rate-limited"
    if "empty completion" in error:
        return "empty answer"
    if "timeout" in error.lower() or "Timeout" in error:
        return "timed out"
    return error.split(":", 1)[0][:40]


def build_filters(folder: str | None, task: str | None) -> dict[str, str] | None:
    filters = {
        name: value
        for name, value in (("folder", folder), ("task", task))
        if value and value != ANY
    }
    return filters or None


# -- chat handler --------------------------------------------------------------------------


def respond(
    assistant: CodeAssistant,
    message: str,
    chat: Chat,
    history: list[dict[str, str]],
    folder: str | None,
    task: str | None,
    use_paid: bool = False,
    rerank: bool = True,
) -> Iterator[tuple[str, Chat, list[dict[str, str]], str, str]]:
    """Stream (textbox, chat, history, sources HTML, status HTML) updates for one question.

    `chat` is what the user sees (citations rendered as links); `history` keeps the raw
    question/answer text that is sent back to the LLM for follow-up questions.
    """
    question = (message or "").strip()
    if not question:
        yield "", chat, history, gr.skip(), gr.skip()
        return
    chat = [*chat, {"role": "user", "content": question}, {"role": "assistant", "content": ""}]
    sources: list[SearchResult] = []
    # Only this first update clears the textbox: later ones skip it, so a follow-up question
    # typed while the answer is still streaming is not erased.
    yield "", chat, history, render_sources([]), render_status(stage="Searching the code…")

    keep = gr.skip()
    partial = ""
    filters = build_filters(folder, task)
    try:
        for event in assistant.ask_stream(
            question, history=history, filters=filters, use_paid=use_paid, rerank=rerank
        ):
            if isinstance(event, Answer):
                chat[-1] = {"role": "assistant", "content": render_answer(event.text, sources)}
                history = [
                    *history,
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": event.text},
                ]
                cited = render_sources(sources, event.citations, answered=True)
                yield keep, chat, history, cited, render_status(event)
            elif isinstance(event, str):
                partial += event
                chat[-1] = {"role": "assistant", "content": render_answer(partial, sources)}
                yield keep, chat, history, keep, keep
            else:
                sources = event
                stage = "Writing the answer…" if sources else "No matching code found."
                yield keep, chat, history, render_sources(sources), render_status(stage=stage)
    except LLMError as exc:
        logger.warning("LLM failure: %s", exc)
        content = render_answer(partial, sources) if partial else NO_ANSWER
        chat[-1] = {"role": "assistant", "content": content}
        yield keep, chat, history, keep, render_status(error=f"No answer. {_summary(exc)}")
    except PyMongoError as exc:
        logger.warning("Search failure: %s", exc)
        chat[-1] = {"role": "assistant", "content": "The code search is unavailable right now."}
        error = "Couldn't reach MongoDB Atlas. Check your connection and MONGODB_URI."
        yield keep, chat, history, render_sources([]), render_status(error=error)
    except Exception as exc:  # never leave the UI hanging on an unexpected error
        logger.exception("Unexpected error while answering")
        content = render_answer(partial, sources) if partial else "Something went wrong."
        chat[-1] = {"role": "assistant", "content": content}
        error = f"Unexpected error ({type(exc).__name__}). See the server log for details."
        yield keep, chat, history, keep, render_status(error=error)


def _summary(exc: LLMError) -> str:
    text = str(exc)
    if "All models failed" in text:
        attempts = [line.strip(" -") for line in text.splitlines()[1:]]
        return (
            "Tried: "
            + "; ".join(f"{a.split(':', 1)[0]} ({_short_error(a)})" for a in attempts)
            + "."
        )
    return text[:200]


# -- layout --------------------------------------------------------------------------------


def build_app(assistant: CodeAssistant, tag: str, tasks: Sequence[str]) -> gr.Blocks:
    with gr.Blocks(title="Ultralytics code assistant") as demo:
        history = gr.State([])
        with gr.Column(elem_id="masthead"):
            gr.HTML(
                '<div class="brand"><span class="brand-mark" aria-hidden="true"></span>'
                "<h1>Ultralytics code assistant</h1></div>"
                "<p>Ask how YOLO works. Answers come from the source code and cite the lines "
                "they use.</p>"
                f'<p class="scope">Searches ultralytics/models, engine and data at release '
                f"{html.escape(tag)}.</p>"
            )
        with gr.Row(equal_height=False, elem_id="workspace"):
            with gr.Column(scale=7, min_width=360):
                chatbot = gr.Chatbot(
                    elem_id="answer-chat",
                    height=560,
                    show_label=False,
                    buttons=["copy"],
                    placeholder=(
                        "<div class='chat-empty'><p>Ask about training, prediction, datasets "
                        "or export.</p><p>Every answer links to the code it is based on.</p></div>"
                    ),
                )
                status_html = gr.HTML("", elem_id="status")
                textbox = gr.Textbox(
                    placeholder="How do I resume an interrupted training run?",
                    show_label=False,
                    submit_btn="Ask",
                    autofocus=True,
                    elem_id="question",
                )
                paid_model = assistant.paid_llm.models[0] if assistant.paid_llm else ""
                use_paid = gr.Checkbox(
                    False,
                    label=f"Use a paid model ({paid_model}); free models remain the fallback",
                    visible=bool(paid_model),
                    elem_id="use-paid",
                )
                rerank_model = getattr(assistant.retriever.reranker, "model", "")
                use_rerank = gr.Checkbox(
                    bool(rerank_model),
                    label=f"Rerank the sources with {rerank_model}",
                    visible=bool(rerank_model),
                    elem_id="use-rerank",
                )
                with gr.Row(elem_id="examples"):
                    example_buttons = [
                        (gr.Button(label, size="sm", variant="secondary"), question)
                        for label, question in EXAMPLES.items()
                    ]
                with (
                    gr.Accordion("Narrow the search", open=False, elem_id="filters"),
                    gr.Row(),
                ):
                    folder = gr.Dropdown(
                        [ANY, "models", "engine", "data"], value=ANY, label="Folder"
                    )
                    task = gr.Dropdown([ANY, *tasks], value=ANY, label="YOLO task")
                clear = gr.Button(
                    "New conversation", size="sm", variant="secondary", elem_id="new-conversation"
                )
            with gr.Column(scale=5, min_width=320, elem_id="sources-column"):
                gr.HTML("<h2>Code used for this answer</h2>")
                sources_html = gr.HTML(render_sources([]), elem_id="sources")

        outputs = [textbox, chatbot, history, sources_html, status_html]

        def handle(message, chat, hist, folder_value, task_value, paid, rerank):
            yield from respond(
                assistant, message, chat, hist, folder_value, task_value, paid, rerank
            )

        def ask_example(question: str):
            def handler(chat, hist, folder_value, task_value, paid, rerank):
                yield from respond(
                    assistant, question, chat, hist, folder_value, task_value, paid, rerank
                )

            return handler

        # One answer at a time per conversation: lock the question box and the examples while
        # an answer streams (Gradio captures the chat when an event fires, so a second,
        # overlapping answer would overwrite the first), and unlock them afterwards.
        controls = [textbox, *(button for button, _ in example_buttons)]

        def lock():
            return [gr.update(interactive=False)] * len(controls)

        def unlock():
            return [gr.update(interactive=True)] * len(controls)

        answering = [textbox.submit(lock, None, controls, queue=False).then(
            handle, [textbox, chatbot, history, folder, task, use_paid, use_rerank], outputs
        )]  # fmt: skip
        for button, question in example_buttons:
            answering.append(
                button.click(lock, None, controls, queue=False).then(
                    ask_example(question),
                    [chatbot, history, folder, task, use_paid, use_rerank],
                    outputs,
                )
            )
        for event in answering:
            event.then(unlock, None, controls, queue=False)

        def reset():
            unlocked = [gr.update(interactive=True)] * (len(controls) - 1)
            return gr.update(value="", interactive=True), [], [], render_sources([]), "", *unlocked

        reset_outputs = [*outputs, *controls[1:]]
        # Both reset the hidden history and sources too, and stop an answer that is still
        # streaming (otherwise its next update would bring the old conversation back).
        clear.click(reset, None, reset_outputs, cancels=answering)
        chatbot.clear(reset, None, reset_outputs, cancels=answering)
    return demo


THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#EEF1FF",
        c100="#DFE4FF",
        c200="#BFC9FF",
        c300="#8E9FFF",
        c400="#5B73FF",
        c500="#2F4BFF",
        c600="#042AFF",
        c700="#0322D6",
        c800="#021BAA",
        c900="#111F68",
        c950="#0A1340",
    ),  # fmt: skip
    neutral_hue=gr.themes.colors.zinc,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#F5F6F8",
    body_text_color="#1A1D2B",
    body_text_color_subdued="#6B7084",
    background_fill_secondary="#F5F6F8",
    color_accent_soft="#EEF1FF",
    block_background_fill="#FFFFFF",
    block_border_color="#E8E9EF",
    block_border_width="1px",
    block_radius="16px",
    block_shadow="0 1px 2px rgba(16,24,40,.04), 0 12px 32px -18px rgba(16,24,40,.12)",
    border_color_primary="#E8E9EF",
    input_background_fill="#FFFFFF",
    input_border_color="#E8E9EF",
    input_radius="999px",
    button_primary_background_fill="#042AFF",
    button_primary_background_fill_hover="#2A47FF",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F1F2F6",
    button_secondary_border_color="#E8E9EF",
    button_secondary_text_color="#1A1D2B",
    button_large_radius="999px",
    button_small_radius="999px",
)

# Light mode only: Gradio follows the OS theme, so pin ?__theme=light.
FORCE_LIGHT = """<script>
  const url = new URL(window.location);
  if (url.searchParams.get("__theme") !== "light") {
    url.searchParams.set("__theme", "light");
    window.location.replace(url);
  }
</script>"""

CSS = (Path(__file__).parent / "app.css").read_text(encoding="utf-8")
