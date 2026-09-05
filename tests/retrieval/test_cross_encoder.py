"""Unit tests for the cross-encoder reranker's argument validation.

Behaviour that needs the model lives in the integration suite; these checks
run without weights because they happen before anything is loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.retrieval.reranking.cross_encoder import CrossEncoderReranker


@pytest.mark.parametrize(
    ("batch_size", "max_text_chars"), [(0, 2000), (-1, 2000), (16, 0), (16, -1)]
)
def test_invalid_sizes_are_rejected_before_loading(
    tmp_path: Path, batch_size: int, max_text_chars: int
) -> None:
    """Nonsensical settings fail at construction, not on the first query.

    Args:
        tmp_path: Stands in for a model directory that is never opened.
        batch_size: The batch size under test.
        max_text_chars: The truncation limit under test.
    """
    with pytest.raises(ValueError, match="must be positive"):
        CrossEncoderReranker(
            tmp_path, batch_size=batch_size, max_text_chars=max_text_chars
        )


def test_a_missing_model_fails_loudly(tmp_path: Path) -> None:
    """A missing reranker is an error, never a silent download.

    Args:
        tmp_path: Temporary directory standing in for the model location.
    """
    from rag.model_assets import ModelUnavailableError

    with pytest.raises(ModelUnavailableError, match="never downloaded at runtime"):
        CrossEncoderReranker(tmp_path / "absent")
