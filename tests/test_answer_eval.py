import pytest

from code_assistant.evaluation.answer_eval import (
    AnswerRecord,
    JudgeParseError,
    LLMResult,
    Verdict,
    build_judge_messages,
    choose_llm,
    parse_verdict,
)


def record(qid="q", in_scope=True, covered=(True,), claims=(), refused=False, hit=True,
           citations=(1,), invalid=(), latency=1.0, verdict=True) -> AnswerRecord:  # fmt: skip
    return AnswerRecord(
        model="m", question_id=qid, category="training", in_scope=in_scope, answer="...",
        answered_by="m", latency_s=latency, sources=("a",), retrieval_hit=hit,
        citations=citations, invalid_citations=invalid,
        verdict=Verdict(tuple(covered), tuple(claims), refused) if verdict else None,
        error="" if verdict else "judge: failed",
    )  # fmt: skip


def test_parse_verdict_accepts_fenced_json_and_checks_fact_count():
    text = (
        '```json\n{"facts_covered": [true, false], "unsupported_claims": ["x"], '
        '"refused": false, "comment": "ok"}\n```'
    )

    verdict = parse_verdict(text, n_facts=2)

    assert verdict.facts_covered == (True, False)
    assert verdict.coverage == 0.5
    assert verdict.unsupported_claims == ("x",)
    with pytest.raises(JudgeParseError, match="Expected 3"):
        parse_verdict(text, n_facts=3)
    with pytest.raises(JudgeParseError, match="No JSON"):
        parse_verdict("I think it's fine", n_facts=0)


def test_judge_messages_contain_question_sources_facts_and_answer():
    messages = build_judge_messages("How?", ["fact A"], [], "My answer [1]")

    user = messages[1]["content"]
    assert "Question: How?" in user
    assert 'KEY FACTS: ["fact A"]' in user
    assert "<answer>\nMy answer [1]\n</answer>" in user


def test_metrics():
    result = LLMResult(
        "m",
        [
            record("a", covered=(True, True)),
            record("b", covered=(True, False), claims=("made up",)),
            record("c", covered=(False,), refused=True, hit=True, citations=()),
            record("d", verdict=False),  # judge failed: counts as 0 coverage
            record("o1", in_scope=False, covered=(), refused=True),
            record("o2", in_scope=False, covered=(), refused=False, claims=("invented",)),
        ],
    )

    assert result.coverage == pytest.approx((1 + 0.5 + 0 + 0) / 4)
    assert result.fully_correct == 0.25
    assert result.grounded_rate == pytest.approx(3 / 5)  # 5 judged answers, 2 with claims
    assert result.citation_rate == 0.75  # only c has no citations (d was answered with [1])
    assert result.oos_refusal_rate == 0.5
    assert result.false_refusal_rate == pytest.approx(1 / 3)
    assert result.errors == 1


def test_choose_llm_excludes_models_that_invent_out_of_scope_answers():
    honest = LLMResult(
        "honest", [record(covered=(True, False)), record(in_scope=False, covered=(), refused=True)]
    )
    inventive = LLMResult(
        "inventive",
        [record(covered=(True, True)), record(in_scope=False, covered=(), refused=False)],
    )

    chosen, reason = choose_llm([honest, inventive])

    assert chosen is honest
    assert "Ineligible" in reason and "inventive" in reason


def test_choose_llm_prefers_grounded_model_within_margin():
    best = LLMResult(
        "best",
        [
            record(covered=(True,) * 20, claims=("x",)),
            record(in_scope=False, covered=(), refused=True),
        ],
    )
    close = LLMResult(
        "close",
        [record(covered=(True,) * 19 + (False,)), record(in_scope=False, covered=(), refused=True)],
    )

    chosen, reason = choose_llm([best, close])

    assert chosen is close  # 95% vs 100% coverage, but no unsupported claims
    assert "groundedness decided" in reason


def test_choose_llm_without_out_of_scope_questions_does_not_exclude_models():
    only_in_scope = LLMResult("m", [record(covered=(True,))])

    assert only_in_scope.oos_refusal_rate is None
    chosen, _ = choose_llm([only_in_scope])
    assert chosen is only_in_scope


def test_answer_record_round_trips_through_json():
    import json
    from dataclasses import asdict

    from code_assistant.evaluation.answer_runner import record_from_dict

    original = record("q", covered=(True, False), claims=("x",))

    assert record_from_dict(json.loads(json.dumps(asdict(original)))) == original
    failed = record("q", verdict=False)
    assert record_from_dict(json.loads(json.dumps(asdict(failed)))) == failed
