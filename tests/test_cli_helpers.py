from code_assistant.cli import _demote_headings


def test_demote_headings_nests_answer_headings_but_not_code_comments():
    answer = "## Basic\ntext\n```python\n# a comment\nx = 1\n```\n# Top"

    assert (
        _demote_headings(answer) == "#### Basic\ntext\n```python\n# a comment\nx = 1\n```\n### Top"
    )
