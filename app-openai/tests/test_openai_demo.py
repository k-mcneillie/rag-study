"""The demo chunk stream decodes to the same events a live one would."""

from __future__ import annotations

from rag_chat_openai.client import _StreamDecoder
from rag_chat_openai.demo import demo_chunks
from rag_chat_openai.models import Event


def _decode(query: str) -> tuple[list[Event], _StreamDecoder]:
    """Run the canned payloads for a query through a fresh decoder.

    Args:
        query: The question to build the demo stream for.

    Returns:
        The decoded events and the decoder (for its trace).
    """
    decoder = _StreamDecoder()
    events: list[Event] = []
    for payload in demo_chunks(query):
        events.extend(decoder.decode(payload))
    return events, decoder


def test_demo_decodes_to_citations_then_reasoning_then_answer_then_done() -> None:
    """The exact event sequence: citations, 2 reasoning, 3 answer, done."""
    events, _ = _decode("what is X?")

    assert [event.kind for event in events] == [
        "citations",
        "reasoning",
        "reasoning",
        "answer",
        "answer",
        "answer",
        "done",
    ]
    assert len(events[0].citations) == 2
    assert events[0].citations[0].page_number == 3
    assert events[-1].text == "demo"


def test_demo_answer_echoes_the_query() -> None:
    """The question is quoted back in the answer fragments."""
    events, _ = _decode("how many?")

    answer = "".join(event.text for event in events if event.kind == "answer")
    assert "how many?" in answer


def test_demo_trace_keeps_every_raw_line_including_done() -> None:
    """The decoder's trace records each payload, ending with the sentinel."""
    _, decoder = _decode("q")

    assert decoder.trace[-1] == "[DONE]"
    assert len(decoder.trace) == len(demo_chunks("q"))
