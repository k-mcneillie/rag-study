"""Loading of local model assets, shared by both pipelines.

Ingestion embeds chunks and retrieval embeds queries. They are separate
concerns on separate schedules, but both load a model from disk, and the rules
for doing so safely are identical: the directory must exist, the library must
not reach the network, and code shipped alongside the weights must not run.

Those rules live here, in one place, rather than being written twice. A single
implementation is easier to audit than two that must be kept in agreement, and
model loading is exactly the kind of privileged operation where an
inconsistency between two copies would matter.

This module sits alongside :mod:`rag.domain` and :mod:`rag.storage` as shared
infrastructure. It is infrastructure for a model asset in the same way that
``storage`` is infrastructure for a database: neither pipeline owns it, and
neither pipeline reaches through it to the other.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when a local model cannot be loaded.

    Names the configured path so the problem can be fixed, without implying
    that downloading the model is an option.
    """


def load_sentence_transformer(model_path: Path) -> SentenceTransformer:
    """Load a sentence-transformers model from a local directory.

    Every route to the network is closed: the directory is checked first, the
    library is told to use local files only, and remote code execution is
    refused. A missing model fails immediately rather than silently reaching
    for the Hub and succeeding on a machine that happens to be online.

    Refusing remote code is a security boundary as much as an offline one.
    Loading a model can execute code shipped alongside its weights, so a model
    directory is trusted infrastructure configured by an operator, never a path
    chosen by a caller or derived from a document.

    Args:
        model_path: Directory holding the model's files.

    Returns:
        The loaded model.

    Raises:
        ModelUnavailableError: If the directory is missing or unloadable.
    """
    if not model_path.is_dir():
        raise ModelUnavailableError(
            f"No model at {model_path}. Model weights are local assets and are "
            f"never downloaded at runtime; place the model there and point the "
            f"configured model path at it."
        )

    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        logger.warning(
            "Failed to load model from %s: %s", model_path, type(exc).__name__
        )
        raise ModelUnavailableError(
            f"The model at {model_path} could not be loaded."
        ) from exc


def embedding_dimension(model: SentenceTransformer) -> int:
    """Read a model's output width across sentence-transformers versions.

    The accessor was renamed in recent releases; the old name still works but
    warns, so the new one is preferred where present.

    Args:
        model: The loaded model.

    Returns:
        The embedding dimensionality.
    """
    accessor = getattr(model, "get_embedding_dimension", None) or (
        model.get_sentence_embedding_dimension
    )
    return int(accessor() or 0)
