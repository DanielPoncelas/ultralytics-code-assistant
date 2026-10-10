import ast
import textwrap

from code_assistant.indexing.chunker import chunk_source

SAMPLE = textwrap.dedent(
    '''\
    """Sample module docstring."""

    import os

    DEFAULT_SIZE = 640


    def helper(x: int, y: int = 2) -> int:
        """Add two numbers."""

        def inner():
            return 1

        return x + y + inner()


    class Trainer(BaseTrainer):
        """Train a model.

        Longer description.
        """

        epochs = 100

        def __init__(self, cfg):
            """Initialize the trainer."""
            self.cfg = cfg

        @property
        def size(self):
            return self._size

        @size.setter
        def size(self, value):
            self._size = value

        async def fit(self) -> None:
            pass
    '''
)
PATH = "ultralytics/engine/sample.py"


def by_name(chunks, name):
    return [c for c in chunks if c.qualified_name == name]


def test_produces_one_chunk_per_unit():
    chunks = chunk_source(SAMPLE, PATH)

    kinds = sorted((c.kind, c.qualified_name) for c in chunks)
    assert kinds == [
        ("class", "Trainer"),
        ("function", "helper"),
        ("method", "Trainer.__init__"),
        ("method", "Trainer.fit"),
        ("method", "Trainer.size"),
        ("method", "Trainer.size"),
        ("module", "ultralytics.engine.sample"),
    ]


def test_function_chunk_keeps_full_source_and_nested_functions():
    (helper,) = by_name(chunk_source(SAMPLE, PATH), "helper")

    assert helper.code.startswith("def helper(x: int, y: int = 2) -> int:")
    assert "def inner():" in helper.code  # nested function stays inside its parent
    assert helper.signature == "def helper(x: int, y: int=2) -> int"  # ast.unparse canonical form
    assert helper.docstring == "Add two numbers."
    assert (helper.start_line, helper.end_line) == (8, 14)


def test_method_metadata_and_line_numbers_include_decorators():
    getter, setter = by_name(chunk_source(SAMPLE, PATH), "Trainer.size")

    assert getter.parent_class == "Trainer"
    assert getter.decorators == ("@property",)
    assert setter.decorators == ("@size.setter",)
    assert getter.code.startswith("@property")  # dedented and includes decorator
    assert SAMPLE.splitlines()[getter.start_line - 1].strip() == "@property"


def test_same_qualified_name_gets_distinct_ids():
    getter, setter = by_name(chunk_source(SAMPLE, PATH), "Trainer.size")

    assert (getter.occurrence, setter.occurrence) == (1, 2)
    assert getter.id != setter.id


def test_ids_are_stable_across_runs():
    first = [c.id for c in chunk_source(SAMPLE, PATH)]
    second = [c.id for c in chunk_source(SAMPLE, PATH)]

    assert first == second


def test_class_summary_has_header_attributes_and_method_stubs_but_no_bodies():
    (summary,) = by_name(chunk_source(SAMPLE, PATH), "Trainer")

    assert summary.bases == ("BaseTrainer",)
    assert summary.signature == "class Trainer(BaseTrainer)"
    assert "class Trainer(BaseTrainer):" in summary.code
    assert "epochs = 100" in summary.code
    assert "def __init__(self, cfg):" in summary.code
    assert '"""Initialize the trainer."""' in summary.code
    assert "async def fit(self) -> None:" in summary.code
    assert "self.cfg = cfg" not in summary.code  # bodies live in method chunks


def test_module_summary_has_imports_constants_and_symbol_stubs():
    (module,) = by_name(chunk_source(SAMPLE, PATH), "ultralytics.engine.sample")

    assert module.docstring == "Sample module docstring."
    assert "import os" in module.code
    assert "DEFAULT_SIZE = 640" in module.code
    assert "def helper(x: int, y: int=2) -> int:" in module.code
    assert "class Trainer(BaseTrainer):" in module.code
    assert "return x + y" not in module.code


def test_package_init_module_name_drops_init():
    chunks = chunk_source('"""Package."""\n', "ultralytics/data/__init__.py")

    assert chunks[0].qualified_name == "ultralytics.data"
    assert chunks[0].module == "ultralytics.data"


def test_empty_file_produces_no_chunks():
    assert chunk_source("", "ultralytics/data/__init__.py") == []


def test_embedding_text_has_contextual_header():
    (init,) = by_name(chunk_source(SAMPLE, PATH), "Trainer.__init__")
    text = init.embedding_text()

    assert text.startswith(
        "# File: ultralytics/engine/sample.py\n"
        "# Module: ultralytics.engine.sample\n"
        "# Symbol: Trainer.__init__ (method of class Trainer)\n"
        "# Signature: def __init__(self, cfg)\n"
    )
    assert text.endswith(init.code)


LONG_FUNCTION = "def long():\n" + "".join(
    f"    value_{i} = compute({i})  # padding to make the line longer\n" for i in range(200)
)


def test_long_function_is_split_at_statement_boundaries_with_overlap():
    chunks = chunk_source(LONG_FUNCTION, PATH, max_chars=2000, overlap_lines=3)
    parts = by_name(chunks, "long")

    assert len(parts) > 1
    assert [p.part for p in parts] == list(range(1, len(parts) + 1))
    assert all(p.total_parts == len(parts) for p in parts)
    assert all(len(p.code) <= 2000 for p in parts)
    for previous, current in zip(parts, parts[1:], strict=False):
        assert current.start_line == previous.end_line - 3 + 1  # 3 overlapping lines
    assert parts[0].start_line == 1
    assert parts[-1].end_line == len(LONG_FUNCTION.splitlines())


def test_split_never_cuts_inside_a_multiline_statement():
    body = "".join(
        f"    call_{i}(\n        argument_one={i},\n        argument_two={i},\n    )\n"
        for i in range(60)
    )
    source = "def long():\n" + body
    for part in by_name(chunk_source(source, PATH, max_chars=1000, overlap_lines=0), "long"):
        # every part must be parseable on its own -> no statement was cut in half
        ast.parse(textwrap.dedent(part.code).replace("def long():", "if True:"))


COMMENTED = textwrap.dedent(
    """\
    # License header

    import os

    # Cache version: bump to invalidate old caches
    CACHE_VERSION = "1.0"


    # NOTE: keep this class for backward compatibility
    @dataclass
    class Legacy:
        # Default size used when none is given
        size = 640

        # Kept for older callers
        def old(self):
            return 1
    """
)


def test_comments_between_statements_are_attached_to_the_next_statement():
    chunks = chunk_source(COMMENTED, PATH)
    (module,) = by_name(chunks, "ultralytics.engine.sample")
    (legacy,) = by_name(chunks, "Legacy")
    (old,) = by_name(chunks, "Legacy.old")

    assert "# Cache version: bump to invalidate old caches" in module.code
    assert "# NOTE: keep this class for backward compatibility" in module.code
    assert legacy.code.startswith("# NOTE: keep this class for backward compatibility\n@dataclass")
    assert "# Default size used when none is given" in legacy.code
    assert old.code.startswith("# Kept for older callers\ndef old(self):")
    assert old.start_line == 15  # line numbers include the attached comment


def test_every_non_blank_line_appears_in_some_chunk():
    for source in (SAMPLE, COMMENTED):
        covered = {line.strip() for c in chunk_source(source, PATH) for line in c.code.splitlines()}
        missing = [
            line for line in source.splitlines() if line.strip() and line.strip() not in covered
        ]
        assert missing == []
