"""Integration tests for the cross-encoder reranker.

These need the reranker weights on disk and are skipped when they are absent.
They check the property that justifies the stage existing: that a passage
which merely shares a subject with the query is pushed below one that answers
it, which vector similarity alone does not reliably do.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag.domain.models import Chunk, RankedChunk
from rag.retrieval.reranking.cross_encoder import CrossEncoderReranker

pytestmark = pytest.mark.integration

DEFAULT_MODEL_PATH = Path("models/rerankers/ms-marco-MiniLM-L-6-v2")

QUERY = "How does direct preference optimisation avoid training a reward model?"

#: A passage that answers the query, and one that merely shares its subject.
#: The second is the shape of result the initial ranker gets wrong: it is about
#: the right paper, and about the right topic, but answers nothing.
ANSWERING = (
    "Direct preference optimisation optimises the policy with a binary "
    "cross-entropy objective over preference pairs, which removes the need to "
    "fit an explicit reward model before reinforcement learning."
)
ON_TOPIC_BUT_USELESS = (
    "Author Contributions. EM wrote the first implementation of DPO and ran "
    "the first DPO experiments; CF, CM and SE supervised the research on "
    "direct preference optimisation and assisted in writing the paper."
)


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    """Load the cross-encoder, skipping if it is not provisioned.

    Returns:
        The reranker under test.
    """
    path = Path(os.environ.get("RAG_RERANKER_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.is_dir():
        pytest.skip(f"No cross-encoder at {path}")
    return CrossEncoderReranker(path)


def _ranked(identifier: str, text: str, score: float, rank: int) -> RankedChunk:
    """Build a ranked candidate.

    Args:
        identifier: The chunk's identifier.
        text: The chunk's text.
        score: The initial ranker's score.
        rank: The initial position.

    Returns:
        The ranked chunk.
    """
    return RankedChunk(
        chunk=Chunk(
            id=identifier, document_id="doc-1", page_number=rank + 1, text=text
        ),
        score=score,
        rank=rank,
    )


def test_an_answering_passage_is_promoted_over_a_merely_related_one(
    reranker: CrossEncoderReranker,
) -> None:
    """The reranker corrects an ordering the vector ranker got wrong.

    Args:
        reranker: The reranker under test.
    """
    # The initial ranker is deliberately given the wrong order: the useless
    # passage scores higher, as happened against the real corpus.
    candidates = [
        _ranked("useless", ON_TOPIC_BUT_USELESS, 0.65, 0),
        _ranked("answering", ANSWERING, 0.60, 1),
    ]

    reranked = reranker.rerank(QUERY, candidates, top_k=2)

    assert [chunk.chunk.id for chunk in reranked] == ["answering", "useless"]


def test_both_scores_are_kept_so_the_change_is_auditable(
    reranker: CrossEncoderReranker,
) -> None:
    """A reordering can be traced to the stage that caused it.

    Args:
        reranker: The reranker under test.
    """
    candidates = [_ranked("a", ANSWERING, 0.6, 0)]

    result = reranker.rerank(QUERY, candidates, top_k=1)[0]

    assert result.initial_score == 0.6
    assert result.rerank_score != 0.6


def test_results_are_ordered_and_ranked_consistently(
    reranker: CrossEncoderReranker,
) -> None:
    """Scores decrease down the list and ranks number from zero.

    Args:
        reranker: The reranker under test.
    """
    candidates = [
        _ranked("a", ANSWERING, 0.6, 0),
        _ranked("b", ON_TOPIC_BUT_USELESS, 0.6, 1),
        _ranked("c", "An unrelated passage about baking bread at home.", 0.6, 2),
    ]

    reranked = reranker.rerank(QUERY, candidates, top_k=3)

    assert [chunk.rank for chunk in reranked] == [0, 1, 2]
    scores = [chunk.rerank_score for chunk in reranked]
    assert scores == sorted(scores, reverse=True)


def test_reranking_narrows_to_top_k(reranker: CrossEncoderReranker) -> None:
    """Only the requested number of results is returned.

    Args:
        reranker: The reranker under test.
    """
    candidates = [_ranked(str(i), ANSWERING, 0.6, i) for i in range(5)]

    assert len(reranker.rerank(QUERY, candidates, top_k=2)) == 2


def test_an_empty_shortlist_is_handled(reranker: CrossEncoderReranker) -> None:
    """Nothing to rerank costs nothing and returns nothing.

    Args:
        reranker: The reranker under test.
    """
    assert reranker.rerank(QUERY, [], top_k=3) == []


def test_scoring_truncation_does_not_shorten_the_stored_passage(
    reranker: CrossEncoderReranker,
) -> None:
    """A long chunk is truncated for scoring only, never for the prompt.

    Args:
        reranker: The reranker under test.
    """
    long_text = ANSWERING + " " + ("Filler sentence. " * 400)
    candidates = [_ranked("long", long_text, 0.6, 0)]

    result = reranker.rerank(QUERY, candidates, top_k=1)[0]

    assert result.chunk.text == long_text
