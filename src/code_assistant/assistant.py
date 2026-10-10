"""Retrieval-augmented answering: retrieve code, build the prompt, generate a cited answer."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from code_assistant.llm import Attempt, Completion, LLMError, Message, OpenRouterClient
from code_assistant.prompt import build_messages, build_rewrite_messages, extract_citations
from code_assistant.retrieval import Retriever, SearchResult


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    sources: list[SearchResult]
    model: str
    latency_s: float
    citations: list[int] = field(default_factory=list)
    failed_attempts: tuple[Attempt, ...] = ()  # models tried before the one that answered
    truncated: bool = False  # the LLM ran out of tokens before finishing
    search_query: str = ""  # what was searched; differs from `question` for rewritten follow-ups

    @property
    def invalid_citations(self) -> list[int]:
        """Cited numbers that don't correspond to any provided source."""
        return [n for n in self.citations if not 1 <= n <= len(self.sources)]

    @property
    def cited_sources(self) -> list[SearchResult]:
        return [self.sources[n - 1] for n in self.citations if 1 <= n <= len(self.sources)]


class CodeAssistant:
    def __init__(self, retriever: Retriever, llm: OpenRouterClient, tag: str, k: int = 6) -> None:
        self.retriever = retriever
        self.llm = llm
        self.tag = tag
        self.k = k

    def retrieve(self, question: str, filters: dict[str, str] | None = None) -> list[SearchResult]:
        return self.retriever.search(question, k=self.k, filters=filters)

    def search_query(self, question: str, history: Sequence[Message]) -> str:
        """Rewrite a follow-up ("can it be disabled?") into a standalone query for retrieval.

        Retrieval only sees one string, so without this a follow-up loses its subject. Only
        runs when there is history; if the LLM call fails, the original question is used.
        """
        if not history:
            return question
        try:
            text = self.llm.complete(build_rewrite_messages(question, history)).text
        except LLMError:
            return question
        query = text.strip().splitlines()[0].strip().strip("\"'`") if text.strip() else ""
        return query[:300] or question

    def ask(
        self,
        question: str,
        history: Sequence[Message] = (),
        filters: dict[str, str] | None = None,
    ) -> Answer:
        query = self.search_query(question, history)
        sources = self.retrieve(query, filters)
        completion = self.llm.complete(build_messages(question, sources, self.tag, history))
        return self._answer(question, sources, completion, query)

    def ask_stream(
        self,
        question: str,
        history: Sequence[Message] = (),
        filters: dict[str, str] | None = None,
    ) -> Iterator[list[SearchResult] | str | Answer]:
        """Yield the retrieved sources first, then text deltas, then the final `Answer`."""
        query = self.search_query(question, history)
        sources = self.retrieve(query, filters)
        yield sources
        for event in self.llm.stream(build_messages(question, sources, self.tag, history)):
            if isinstance(event, Completion):
                yield self._answer(question, sources, event, query)
            else:
                yield event

    @staticmethod
    def _answer(
        question: str, sources: list[SearchResult], completion: Completion, query: str
    ) -> Answer:
        return Answer(
            question=question,
            text=completion.text,
            sources=sources,
            model=completion.model,
            latency_s=completion.latency_s,
            citations=extract_citations(completion.text),
            failed_attempts=completion.failed_attempts,
            truncated=completion.truncated,
            search_query=query,
        )
