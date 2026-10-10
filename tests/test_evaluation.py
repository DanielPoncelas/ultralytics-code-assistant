from pathlib import Path

import numpy as np
import pytest

from code_assistant.evaluation.dataset import (
    DatasetError,
    EvalQuestion,
    load_questions,
    unknown_labels,
    validate,
)
from code_assistant.evaluation.report import CorpusInfo, render_markdown
from code_assistant.evaluation.retrieval_eval import (
    ModelResult,
    QuestionOutcome,
    bootstrap_ci,
    choose_model,
    compare_with_atlas,
    evaluate_model,
    first_relevant_rank,
    rank_symbols,
)

QUESTIONS_FILE = Path(__file__).parent.parent / "eval" / "questions.toml"


def question(qid, relevant=("a.py::A",), in_scope=True, text=None):
    return EvalQuestion(
        id=qid,
        question=text or qid,
        category="training",
        source="https://github.com/x",
        relevant=tuple(relevant),
        in_scope=in_scope,
    )


def outcome(rank, qid="q"):
    return QuestionOutcome(qid, "training", False, rank, ())


def result(model, params_m, ranks):
    return ModelResult(model, params_m, 8, [outcome(r, f"q{i}") for i, r in enumerate(ranks)])


# -- dataset ---------------------------------------------------------------------------


def test_question_file_is_consistent():
    questions = load_questions(QUESTIONS_FILE)

    assert len(questions) == 46
    assert sum(q.in_scope for q in questions) == 40
    assert len([q for q in questions if q.example]) == 5
    assert all(
        q.source.startswith("https://github.com/ultralytics/ultralytics/issues/") for q in questions
    )


def test_validate_rejects_inconsistent_questions():
    with pytest.raises(DatasetError, match="Duplicate"):
        validate([question("a"), question("a")])
    with pytest.raises(DatasetError, match="need at least one"):
        validate([question("a", relevant=())])
    with pytest.raises(DatasetError, match="must not list"):
        validate([question("a", in_scope=False)])


def test_unknown_labels_reports_missing_symbols():
    questions = [question("a", relevant=("x.py::X", "y.py::Y"))]

    assert unknown_labels(questions, {"x.py::X"}) == {"a": ["y.py::Y"]}
    assert unknown_labels(questions, {"x.py::X", "y.py::Y"}) == {}


# -- ranking and metrics ---------------------------------------------------------------


def test_rank_symbols_orders_by_cosine_and_collapses_parts():
    matrix = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    symbols = ["a.py::A", "a.py::A", "b.py::B"]  # first two rows are parts of the same symbol

    assert rank_symbols(np.array([1.0, 0.0]), matrix, symbols) == ["a.py::A", "b.py::B"]
    assert rank_symbols(np.array([0.0, 1.0]), matrix, symbols) == ["b.py::B", "a.py::A"]


def test_first_relevant_rank():
    assert first_relevant_rank(["x", "y", "z"], ["z", "y"]) == 2
    assert first_relevant_rank(["x"], ["z"]) is None


def test_hit_at_k_and_mrr():
    r = result("m", 100, [1, 3, None, 6])

    assert r.hit_at(1) == 0.25
    assert r.hit_at(5) == 0.5
    assert r.mrr == pytest.approx((1 + 1 / 3 + 0 + 1 / 6) / 4)


def test_evaluate_model_scores_only_in_scope_questions_and_times_queries():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    symbols = ["a.py::A", "b.py::B"]
    questions = [
        question("hit", relevant=("a.py::A",), text="about A"),
        question("second", relevant=("b.py::B",), text="about A again"),
        question("oos", relevant=(), in_scope=False),
    ]
    vectors = {"about A": [1.0, 0.0], "about A again": [1.0, 0.0], "warm-up": [0.0, 0.0]}

    r = evaluate_model("m", 10, questions, matrix, symbols, lambda text: vectors[text])

    assert [(o.question_id, o.rank) for o in r.outcomes] == [("hit", 1), ("second", 2)]
    assert len(r.query_latency_ms) == 2  # warm-up and out-of-scope questions are not timed


def test_bootstrap_ci_is_degenerate_for_constant_values_and_contains_the_mean():
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)
    low, high = bootstrap_ci([1.0, 0.0] * 20)
    assert low < 0.5 < high


# -- decision rule ---------------------------------------------------------------------


def test_choose_model_picks_best_when_no_smaller_model_is_close():
    big = result("big", 600, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # Hit@5 100%
    small = result("small", 150, [1, 1, 1, 1, 1, 1, 1, 1, None, None])  # 80%

    chosen, reason = choose_model([big, small])

    assert chosen is big
    assert "no smaller model" in reason


def test_choose_model_prefers_smaller_model_within_margin():
    # With 40 questions one miss is 2.5 points: within the 3-point margin. Two misses are not.
    big = result("big", 600, [1] * 40)
    one_miss = result("one-miss", 150, [1] * 39 + [None])
    two_misses = result("two-misses", 33, [1] * 38 + [None, None])

    chosen, reason = choose_model([big, one_miss, two_misses])

    assert chosen is one_miss
    assert "smaller model wins" in reason


def test_choose_model_breaks_size_ties_by_quality():
    a = result("a", 149, [1, 1, 1, 1, None])
    b = result("b", 149, [1, 1, 1, 1, 1])

    chosen, _ = choose_model([a, b])

    assert chosen is b


def test_compare_with_atlas_measures_overlap_and_hits():
    questions = [question("q0", relevant=("a",)), question("q1", relevant=("z",))]
    exact = ModelResult(
        "m",
        10,
        8,
        [
            QuestionOutcome("q0", "training", False, 1, ("a", "b", "c", "d", "e")),
            QuestionOutcome("q1", "training", False, None, ("a", "b", "c", "d", "e")),
        ],
    )
    atlas = {"q0": ["a", "b", "c", "d", "x"], "q1": ["z", "b", "c", "d", "e"]}

    agreement = compare_with_atlas(exact, questions, lambda text: atlas[text])

    assert agreement.overlap_at_5 == pytest.approx((4 / 5 + 4 / 5) / 2)
    assert agreement.hit_at_5_exact == 0.5
    assert agreement.hit_at_5_atlas == 1.0


def test_render_markdown_marks_chosen_model_and_lists_failures():
    questions = [question("q0"), question("q1")]
    good = ModelResult(
        "good",
        100,
        8,
        [
            QuestionOutcome("q0", "training", False, 1, ("a.py::A",)),
            QuestionOutcome("q1", "training", False, None, ("ultralytics/x.py::X",)),
        ],
        [10.0, 12.0],
    )

    report = render_markdown([good], good, "because", questions, CorpusInfo("v1", "abcdef1", 5, 2))

    assert "| `good` ✅ |" in report
    assert "### `q1` (not in top 10)" in report
    assert "`x.py::X`" in report  # 'ultralytics/' prefix trimmed
