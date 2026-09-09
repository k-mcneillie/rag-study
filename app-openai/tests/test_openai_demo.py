"""The demo chunk stream decodes to the same events a live one would."""

from __future__ import annotations

from rag_chat_openai.client import _StreamDecoder
from rag_chat_openai.demo import demo_chunks
from rag_chat_openai.models import CitationsEvent, DeltaEvent, DoneEvent, StreamEvent


def _decode(query: str) -> tuple[list[StreamEvent], _StreamDecoder]:
    """Run the canned payloads for a query through a fresh decoder.

    Args:
        query: The question to build the demo stream for.

    Returns:
        The decoded events and the decoder (for its trace).
    """
    decoder = _StreamDecoder()
    events: list[StreamEvent] = []
    for payload in demo_chunks(query):
        events.extend(decoder.decode(payload))
    return events, decoder


def test_demo_decodes_to_citations_then_reasoning_then_answer_then_done() -> None:
    """The exact event sequence: citations, 2 reasoning, 3 answer, done."""
    events, _ = _decode("what is X?")

    assert isinstance(events[0], CitationsEvent)
    assert len(events[0].items) == 2
    assert events[0].items[0].page_number == 3
    tail = [
        (type(event).__name__, getattr(event, "reasoning", None))
        for event in events[1:]
    ]
    assert tail == [
        ("DeltaEvent", True),
        ("DeltaEvent", True),
        ("DeltaEvent", False),
        ("DeltaEvent", False),
        ("DeltaEvent", False),
        ("DoneEvent", None),
    ]
    assert isinstance(events[-1], DoneEvent)


def test_demo_answer_echoes_the_query() -> None:
    """The question is quoted back in the answer fragments."""
    events, _ = _decode("how many?")

    answer = "".join(
        event.text
        for event in events
        if isinstance(event, DeltaEvent) and not event.reasoning
    )
    assert "how many?" in answer


def test_demo_trace_keeps_every_raw_line_including_done() -> None:
    """The decoder's trace records each payload, ending with the sentinel."""
    _, decoder = _decode("q")

    assert decoder.trace[-1] == "[DONE]"
    assert len(decoder.trace) == len(demo_chunks("q"))
