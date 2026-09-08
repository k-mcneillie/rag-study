"""The demo stream has the same shape as a live one."""

from __future__ import annotations

from rag_chat.demo import demo_events
from rag_chat.models import CitationsEvent, DeltaEvent, DoneEvent


async def test_demo_events_open_with_citations_and_close_with_done() -> None:
    """The order is citations, then deltas, then done."""
    events = [event async for event in demo_events("what is X?")]

    assert isinstance(events[0], CitationsEvent)
    assert len(events[0].items) == 2
    assert isinstance(events[-1], DoneEvent)
    assert all(isinstance(event, DeltaEvent) for event in events[1:-1])


async def test_demo_reasoning_precedes_the_answer_and_echoes_the_query() -> None:
    """Reasoning fragments come first, and the question is quoted back."""
    deltas = [
        event
        async for event in demo_events("how many?")
        if isinstance(event, DeltaEvent)
    ]

    reasoning = [d for d in deltas if d.reasoning]
    answer = [d for d in deltas if not d.reasoning]

    assert reasoning and answer
    assert deltas.index(reasoning[-1]) < deltas.index(answer[0])
    assert "how many?" in "".join(d.text for d in answer)
