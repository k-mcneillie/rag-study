"""A simple, modular, offline-capable RAG package.

The package is built from two independent pipelines that share only data
contracts and storage:

* :mod:`rag.ingestion` turns source documents into embedded, persisted chunks.
* :mod:`rag.retrieval` turns a query into ranked context with provenance.

Neither pipeline imports the other. Both depend on :mod:`rag.domain` for the
data contracts they exchange and on :mod:`rag.storage` for persistence.
"""

__version__ = "0.1.0"
