"""A standalone Chainlit chat interface for the rag-study API service.

The package talks to the API over HTTP and has no dependency on ``rag``:
the whole ``app/`` directory can be copied into another repository, pointed
at a running service with ``RAG_API_URL`` and ``RAG_API_KEY``, and run.
"""
