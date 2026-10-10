from code_assistant.assistant import Answer, CodeAssistant
from code_assistant.llm import Attempt, Completion
from code_assistant.retrieval import SearchResult


def source(name: str) -> SearchResult:
    return SearchResult(
        id=name, kind="method", path="p.py", qualified_name=name, code="...",
        start_line=1, end_line=2, github_url="https://x", score=0.9,
    )  # fmt: skip


class FakeRetriever:
    def __init__(self, results):
        self.results, self.calls = results, []

    def search(self, query, k=5, filters=None):
        self.calls.append((query, k))
        return self.results[:k]


class FakeLLM:
    def __init__(self, text):
        self.text, self.messages = text, None

    def complete(self, messages):
        self.messages = messages
        return Completion(self.text, "x/model", 1.5, (Attempt("a/model", "HTTP 429"),))

    def stream(self, messages):
        self.messages = messages
        yield from ["Use ", "resume [1]"]
        yield Completion("Use resume [1]", "x/model", 1.0)


def test_ask_retrieves_k_sources_and_parses_citations():
    retriever = FakeRetriever([source("A"), source("B"), source("C")])
    llm = FakeLLM("Call resume [1] then check [2][7].")

    answer = CodeAssistant(retriever, llm, tag="v1", k=2).ask("how to resume?")

    assert retriever.calls == [("how to resume?", 2)]
    assert [s.qualified_name for s in answer.sources] == ["A", "B"]
    assert answer.citations == [1, 2, 7]
    assert answer.invalid_citations == [7]  # only 2 sources were provided
    assert [s.qualified_name for s in answer.cited_sources] == ["A", "B"]
    assert answer.model == "x/model"
    assert answer.failed_attempts[0].model == "a/model"
    assert "how to resume?" in llm.messages[-1]["content"]


def test_ask_stream_yields_sources_then_text_then_answer():
    assistant = CodeAssistant(FakeRetriever([source("A")]), FakeLLM(""), tag="v1")

    events = list(assistant.ask_stream("q"))

    assert [s.qualified_name for s in events[0]] == ["A"]
    assert events[1:3] == ["Use ", "resume [1]"]
    assert isinstance(events[-1], Answer)
    assert events[-1].citations == [1]
