"""An HTTP API in front of the retrieval and generation pipelines.

This package is an entry point, not a layer. It lives outside ``src/rag`` for
the same reason ``scripts/`` and ``app/`` do: the package is a library that
assembles differently in different contexts, and the assembly belongs to the
context. Deleting this directory removes the API and nothing else.

It composes the query side through :mod:`rag.assembly` exactly as the scripts
do, and its own ingestion wiring in :mod:`service.ingest_wiring` (the package's
``assembly`` module deliberately never imports ``ingestion``). The prompt is
built and consumed entirely on this side of the boundary: no endpoint returns
an assembled :class:`~rag.domain.models.PromptContext` to a caller.
"""
