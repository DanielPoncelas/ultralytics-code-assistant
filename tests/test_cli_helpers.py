from code_assistant.cli import _demote_headings


def test_demote_headings_nests_answer_headings_but_not_code_comments():
    answer = "## Basic\ntext\n```python\n# a comment\nx = 1\n```\n# Top"

    assert (
        _demote_headings(answer) == "#### Basic\ntext\n```python\n# a comment\nx = 1\n```\n### Top"
    )


def test_unreachable_atlas_gives_a_clear_message_not_a_traceback(monkeypatch, capsys):
    import pytest
    import typer
    from pymongo.errors import ServerSelectionTimeoutError

    from code_assistant import cli

    def unreachable(settings, models=None):
        raise ServerSelectionTimeoutError("SSL handshake failed: TLSV1_ALERT_INTERNAL_ERROR")

    monkeypatch.setattr(cli, "build_assistant", unreachable)

    with pytest.raises(typer.Exit) as exc:
        cli.connect_assistant(settings=None)

    assert exc.value.exit_code == 1
    out = capsys.readouterr().out
    assert "MongoDB Atlas unreachable" in out and "Network Access" in out
