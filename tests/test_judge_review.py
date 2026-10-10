import pytest

from code_assistant.evaluation.dataset import EvalQuestion
from code_assistant.evaluation.judge_review import (
    agreement,
    parse_review_sheet,
    render_review_sheet,
    sample_records,
)

QUESTIONS = {
    "q1": EvalQuestion(
        "q1", "How?", "training", "s", ("a::A",), key_facts=("fact one", "fact two")
    ),
    "q2": EvalQuestion("q2", "Why?", "data", "s", ("b::B",), key_facts=("fact three",)),
}


def rec(qid, covered, in_scope=True):
    verdict = {"facts_covered": covered} if covered is not None else None
    return {
        "question_id": qid,
        "in_scope": in_scope,
        "answer": "line 1\nline 2",
        "verdict": verdict,
    }


def test_sample_is_reproducible_and_only_in_scope_judged_answers():
    records = [rec("q1", [True]), rec("q2", None), rec("o", [], in_scope=False)] + [
        rec("q1", [False]) for _ in range(20)
    ]

    first = sample_records(records, n=5, seed=0)

    assert first == sample_records(records, n=5, seed=0)
    assert all(r["in_scope"] and r["verdict"] for r in first)


def test_sheet_is_blind_and_round_trips_through_the_parser():
    sheet = render_review_sheet([rec("q1", [True, False]), rec("q2", [True])], QUESTIONS, "m")

    assert "True" not in sheet and "judge:" not in sheet.lower()  # no verdicts leaked
    assert "> line 1\n> line 2" in sheet
    ticked = sheet.replace("- [ ] fact one", "- [x] fact one")

    assert parse_review_sheet(ticked) == {"q1": [True, False], "q2": [False]}


def test_agreement_rate_disagreement_direction_and_kappa():
    human = {"q1": [True, True, False, False]}
    judge = {"q1": [True, False, False, True]}

    result = agreement(human, judge)

    assert (result.facts, result.agreed) == (4, 2)
    assert result.rate == 0.5
    assert result.human_yes_judge_no == 1
    assert result.human_no_judge_yes == 1
    assert result.kappa == pytest.approx(0.0)  # no better than chance


def test_perfect_agreement_has_kappa_one():
    assert agreement({"q": [True, False]}, {"q": [True, False]}).kappa == pytest.approx(1.0)
