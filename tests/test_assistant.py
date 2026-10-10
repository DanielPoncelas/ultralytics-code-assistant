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


class RewritingLLM(FakeLLM):
    """complete() answers the rewrite request; stream() answers the question."""

    def __init__(self, rewrite, error=None):
        super().__init__("")
        self.rewrite, self.error, self.rewrite_calls = rewrite, error, 0

    def complete(self, messages):
        self.rewrite_calls += 1
        if self.error:
            raise self.error
        return Completion(self.rewrite, "x/model", 0.5)


HISTORY = [
    {"role": "user", "content": "How does NMS work?"},
    {"role": "assistant", "content": "..."},
]


def test_follow_ups_are_rewritten_before_searching():
    retriever, llm = FakeRetriever([source("A")]), RewritingLLM('"How can NMS be disabled?"\nextra')

    *_, answer = CodeAssistant(retriever, llm, tag="v1").ask_stream("Can it be disabled?", HISTORY)

    assert retriever.calls[0][0] == "How can NMS be disabled?"  # cleaned: first line, no quotes
    assert answer.search_query == "How can NMS be disabled?"
    assert answer.question == "Can it be disabled?"


def test_first_question_is_not_rewritten():
    retriever, llm = FakeRetriever([source("A")]), RewritingLLM("unused")

    list(CodeAssistant(retriever, llm, tag="v1").ask_stream("How does NMS work?"))

    assert llm.rewrite_calls == 0 and retriever.calls[0][0] == "How does NMS work?"


def test_failed_rewrite_falls_back_to_the_original_question():
    from code_assistant.llm import LLMError

    retriever = FakeRetriever([source("A")])
    llm = RewritingLLM("", error=LLMError("all failed"))

    list(CodeAssistant(retriever, llm, tag="v1").ask_stream("Can it be disabled?", HISTORY))

    assert retriever.calls[0][0] == "Can it be disabled?"


def test_paid_option_uses_the_paid_client_for_rewrite_and_answer():
    free, paid = RewritingLLM("free query"), RewritingLLM("paid query")
    retriever = FakeRetriever([source("A")])
    assistant = CodeAssistant(retriever, free, tag="v1", paid_llm=paid)

    *_, answer = assistant.ask_stream("Can it be disabled?", HISTORY, use_paid=True)

    assert paid.rewrite_calls == 1 and free.rewrite_calls == 0
    assert retriever.calls[0][0] == "paid query"
    assert paid.messages is not None and free.messages is None
