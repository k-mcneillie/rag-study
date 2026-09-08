"""``rag.assembly`` exposes the query-side composition helpers.

The functions themselves need a database and model weights to run and are
covered by the integration suite; this only pins that the module imports
cleanly and keeps its public surface, which the scripts and the API service
depend on by name.
"""

from __future__ import annotations

from rag import assembly


def test_assembly_exposes_the_query_side_builders() -> None:
    """The three builders the entry points import are present and callable."""
    assert callable(assembly.build_retrieval_orchestrator)
    assert callable(assembly.build_reranker)
    assert callable(assembly.build_chat_model)
