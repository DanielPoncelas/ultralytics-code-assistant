"""Blind human review of the LLM judge.

`render_review_sheet` samples answers and writes a Markdown checklist with the key facts,
WITHOUT the judge's verdicts (so the reviewer is not anchored by them). After the reviewer
ticks the facts each answer states, `parse_review_sheet` reads the ticks back and
`agreement` compares them with the judge, fact by fact.
"""

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from code_assistant.evaluation.dataset import EvalQuestion

_SECTION = re.compile(r"^## \d+\. `([^`]+)`", re.MULTILINE)
_CHECKBOX = re.compile(r"^- \[([ xX])\] ", re.MULTILINE)


def sample_records(records: Sequence[dict], n: int, seed: int = 0) -> list[dict]:
    """Random, reproducible sample of in-scope answers that the judge graded."""
    eligible = [r for r in records if r["in_scope"] and r["verdict"]]
    return random.Random(seed).sample(eligible, min(n, len(eligible)))


def render_review_sheet(
    sample: Sequence[dict], questions: dict[str, EvalQuestion], model: str
) -> str:
    lines = [
        "# Judge review sheet",
        "",
        f"Blind review of {len(sample)} answers by `{model}`, sampled at random (seed 0).",
        "",
        "For each answer, tick `[x]` every key fact the answer states correctly (in any wording). "
        "Leave `[ ]` if it is missing or wrong. The judge's verdicts are deliberately not shown. "
        "When done, run `uv run code-assistant eval agreement`.",
        "",
    ]
    for i, record in enumerate(sample, start=1):
        q = questions[record["question_id"]]
        answer = "\n".join(f"> {line}" for line in record["answer"].splitlines())
        lines += [
            f"## {i}. `{q.id}`",
            "",
            f"**Question:** {q.question}",
            "",
            answer,
            "",
            "**Key facts:**",
            "",
        ]
        lines += [f"- [ ] {fact}" for fact in q.key_facts]
        lines.append("")
    return "\n".join(lines)


def parse_review_sheet(text: str) -> dict[str, list[bool]]:
    """{question_id: [ticked, ...]} in the order the facts appear."""
    reviews: dict[str, list[bool]] = {}
    sections = _SECTION.split(text)[1:]  # [id, body, id, body, ...]
    for qid, body in zip(sections[::2], sections[1::2], strict=True):
        reviews[qid] = [mark.lower() == "x" for mark in _CHECKBOX.findall(body)]
    return reviews


@dataclass(frozen=True)
class Agreement:
    facts: int
    agreed: int
    human_yes_judge_no: int  # judge stricter than the human
    human_no_judge_yes: int  # judge more lenient than the human
    kappa: float

    @property
    def rate(self) -> float:
        return self.agreed / self.facts if self.facts else 0.0


def agreement(human: dict[str, list[bool]], judge: dict[str, list[bool]]) -> Agreement:
    pairs = [
        (h, j)
        for qid, human_facts in human.items()
        for h, j in zip(human_facts, judge[qid], strict=True)
    ]
    n = len(pairs)
    agreed = sum(h == j for h, j in pairs)
    # Cohen's kappa: agreement corrected for the agreement expected by chance.
    p_observed = agreed / n if n else 0.0
    p_h, p_j = (sum(h for h, _ in pairs) / n, sum(j for _, j in pairs) / n) if n else (0.0, 0.0)
    p_expected = p_h * p_j + (1 - p_h) * (1 - p_j)
    kappa = (p_observed - p_expected) / (1 - p_expected) if p_expected < 1 else 1.0
    return Agreement(
        facts=n,
        agreed=agreed,
        human_yes_judge_no=sum(h and not j for h, j in pairs),
        human_no_judge_yes=sum(j and not h for h, j in pairs),
        kappa=kappa,
    )
