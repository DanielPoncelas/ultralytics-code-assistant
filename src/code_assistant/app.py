"""Gradio chat interface.

Design: the code used for an answer is shown the way YOLO draws a detection, as a box with a
coloured label tab (symbol name and similarity score, like class name and confidence),
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
from code_assistant.retrieval import SearchResult

logger = logging.getLogger(__name__)

# Subset of ultralytics.utils.plotting.Colors (the palette YOLO draws detections with),
# skipping the near-white and the navy that is used for text.
PALETTE = ("042AFF", "0BDBEB", "00DFB7", "FF6FDD", "FF444F", "BD00FF")

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


def label_text_color(hex_color: str) -> str:
    """Readable text colour on a palette swatch (WCAG relative luminance)."""
    channels = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111F68" if luminance > 0.35 else "#FFFFFF"


def citation_url(source: SearchResult, number: int) -> str:
    """GitHub permalink with a `?source=n` marker that the CSS uses to colour the citation.

    GitHub ignores unknown query parameters, so the link still opens the exact lines.
    """
    base, _, fragment = source.github_url.partition("#")
    return f"{base}?source={number}#{fragment}"


_CITATION = re.compile(r"(?<![\w\])])[\[【](\d+(?:\s*,\s*\d+)*)[\]】](?!\()")
_FENCE = re.compile(r"(```.*?(?:```|\Z))", re.DOTALL)
_INLINE_CODE = re.compile(r"(`[^`\n]*`)")


def linkify_citations(text: str, sources: Sequence[SearchResult]) -> str:
    """Turn [n] citations into links to source n, leaving code untouched.

    Numbers that don't match a source stay as plain text, so a wrong citation is visible
    rather than silently linked somewhere.
    """

    def replace(match: re.Match[str]) -> str:
        numbers = [int(n) for n in match.group(1).split(",")]
        if not all(1 <= n <= len(sources) for n in numbers):
            return match.group(0)
        return "".join(f"[{n}]({citation_url(sources[n - 1], n)})" for n in numbers)

    parts = []
    for block in _FENCE.split(text):
        if block.startswith("```"):
            parts.append(block)
            continue
        for span in _INLINE_CODE.split(block):
            parts.append(span if span.startswith("`") else _CITATION.sub(replace, span))
    return "".join(parts)


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
            f'<span class="detection-score">{source.score:.2f}</span></header>'
            f'<p class="detection-meta"><span class="detection-path">'
            f"{html.escape(source.path)}, lines {source.start_line}–{source.end_line}{part}."
            f'</span> <a href="{html.escape(source.github_url)}" target="_blank" '
            'rel="noopener">View on GitHub</a></p>'
            f"<details><summary>Show code</summary><pre><code>{html.escape(source.code)}"
            "</code></pre></details></article>"
        )
    if answered:
        note = (
            f"{len(cited)} of {len(sources)} retrieved code units are cited in the answer. "
            "Dashed boxes were retrieved but not used."
            if cited
            else "The answer doesn't cite any of the retrieved code."
        )
    else:
        note = "Retrieved code, ranked by similarity to your question."
    return f'<p class="sources-note">{note}</p>' + "".join(boxes)


def render_status(answer: Answer | None = None, error: str | None = None, stage: str = "") -> str:
    if error:
        return f'<p class="status status-error">{html.escape(error)}</p>'
    if stage:
        return f'<p class="status status-busy">{html.escape(stage)}</p>'
    if answer is None:
        return ""
    lines = [f"Answered by {html.escape(answer.model)} in {answer.latency_s:.1f} s."]
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
        for event in assistant.ask_stream(question, history=history, filters=filters):
            if isinstance(event, Answer):
                chat[-1] = {"role": "assistant", "content": linkify_citations(event.text, sources)}
                history = [
                    *history,
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": event.text},
                ]
                cited = render_sources(sources, event.citations, answered=True)
                yield keep, chat, history, cited, render_status(event)
            elif isinstance(event, str):
                partial += event
                chat[-1] = {"role": "assistant", "content": linkify_citations(partial, sources)}
                yield keep, chat, history, keep, keep
            else:
                sources = event
                stage = "Writing the answer…" if sources else "No matching code found."
                yield keep, chat, history, render_sources(sources), render_status(stage=stage)
    except LLMError as exc:
        logger.warning("LLM failure: %s", exc)
        chat[-1] = {"role": "assistant", "content": partial or NO_ANSWER}
        yield keep, chat, history, keep, render_status(error=f"No answer. {_summary(exc)}")
    except PyMongoError as exc:
        logger.warning("Search failure: %s", exc)
        chat[-1] = {"role": "assistant", "content": "The code search is unavailable right now."}
        error = "Couldn't reach MongoDB Atlas. Check your connection and MONGODB_URI."
        yield keep, chat, history, render_sources([]), render_status(error=error)
    except Exception as exc:  # never leave the UI hanging on an unexpected error
        logger.exception("Unexpected error while answering")
        chat[-1] = {"role": "assistant", "content": partial or "Something went wrong."}
        yield keep, chat, history, keep, render_status(error=f"Unexpected error: {exc!r}"[:300])


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
                "<h1>Ultralytics code assistant</h1>"
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

        def handle(message, chat, hist, folder_value, task_value):
            yield from respond(assistant, message, chat, hist, folder_value, task_value)

        def ask_example(question: str):
            def handler(chat, hist, folder_value, task_value):
                yield from respond(assistant, question, chat, hist, folder_value, task_value)

            return handler

        textbox.submit(handle, [textbox, chatbot, history, folder, task], outputs)
        for button, question in example_buttons:
            button.click(ask_example(question), [chatbot, history, folder, task], outputs)

        def reset():
            return "", [], [], render_sources([]), ""

        clear.click(reset, None, outputs)
        # The chat's own bin icon only empties the chat: reset history and sources with it,
        # so the next question doesn't carry a conversation the user can no longer see.
        chatbot.clear(reset, None, outputs)
    return demo


THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#EEF2FF",
        c100="#DCE3FF",
        c200="#B6C4FF",
        c300="#8199FF",
        c400="#4C6BFF",
        c500="#042AFF",
        c600="#0322D6",
        c700="#021BAA",
        c800="#02147E",
        c900="#111F68",
        c950="#0A1340",
    ),  # fmt: skip
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Schibsted Grotesk"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="#F4F6FB",
    body_background_fill_dark="#0E1430",
    body_text_color="#111F68",
    body_text_color_dark="#E7EBFA",
    body_text_color_subdued="#5B6488",
    body_text_color_subdued_dark="#9AA3C7",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#151C3D",
    block_border_color="#D9DEEA",
    block_border_color_dark="#2A3462",
    border_color_primary="#D9DEEA",
    border_color_primary_dark="#2A3462",
    button_primary_background_fill="#042AFF",
    button_primary_background_fill_hover="#0322D6",
    button_primary_text_color="#FFFFFF",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#151C3D",
    shadow_drop="none",
    block_shadow="none",
)

CSS = (Path(__file__).parent / "app.css").read_text(encoding="utf-8")
