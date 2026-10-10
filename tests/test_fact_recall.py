import json

import pytest

from code_assistant.evaluation.answer_eval import JudgeParseError
from code_assistant.evaluation.dataset import EvalQuestion
from code_assistant.evaluation.fact_recall import (
    build_fact_messages,
    parse_fact_verdict,
    render_fact_review_sheet,
)
from code_assistant.evaluation.judge_review import parse_review_sheet
from code_assistant.retrieval import SearchResult

SOURCE = SearchResult(
    id="a", kind="method", path="ultralytics/engine/model.py", qualified_name="Model.predict",
    code='def predict(self):\n    custom = {"conf": 0.25,\n              "rect": True}',
    start_line=1, end_line=3, github_url="https://x", score=0.5,
)  # fmt: skip


def judge_output(*facts) -> str:
    return "Sure:\n" + json.dumps({"facts": [{"available": a, "quote": q} for a, q in facts]})


def test_available_facts_need_a_quote_that_is_really_in_the_sources():
    verdict = parse_fact_verdict(
        judge_output(
            (True, 'custom = {"conf": 0.25, "rect": True}'),  # spans two lines: whitespace ignored
            (True, "self.args.conf = 0.001"),  # invented: not in the sources
            (True, "conf"),  # too short to prove anything
            (False, ""),
        ),
        n_facts=4,
        sources=[SOURCE],
    )

    assert verdict.available == (True, False, False, False)
    assert verdict.rejected_quotes == 2
    assert verdict.share == 0.25


def test_malformed_or_incomplete_judge_output_raises():
    with pytest.raises(JudgeParseError):
        parse_fact_verdict("no json here", n_facts=1, sources=[SOURCE])
    with pytest.raises(JudgeParseError):
        parse_fact_verdict(judge_output((True, "x" * 20)), n_facts=2, sources=[SOURCE])


def test_judge_sees_the_sources_and_the_key_facts():
    messages = build_fact_messages("Why?", ["predict uses conf=0.25"], [SOURCE])

    assert '<source id="1" symbol="Model.predict"' in messages[1]["content"]
    assert 'KEY FACTS: ["predict uses conf=0.25"]' in messages[1]["content"]


def test_review_sheet_is_blind_and_parses_back():
    question = EvalQuestion(
        id="val-vs-predict", question="Why?", category="validation", source="https://x",
        in_scope=True, key_facts=("fact one", "fact two"),
    )  # fmt: skip

    sheet = render_fact_review_sheet([(question, [SOURCE])], "Reranked (r/model)")

    assert "retrieved by `Reranked (r/model)`" in sheet and "custom = {" in sheet
    assert parse_review_sheet(sheet.replace("- [ ] fact one", "- [x] fact one")) == {
        "val-vs-predict": [True, False]
    }
