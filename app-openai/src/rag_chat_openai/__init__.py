"""A standalone Chainlit chat interface for an OpenAI-language RAG service.

The package talks to the provider over HTTP in the OpenAI wire dialect
(``POST /v1/chat/completions`` streaming, ``Authorization: Bearer``) and has no
dependency on ``rag``: the whole ``app-openai/`` directory can be copied into
another repository, pointed at a provider with ``RAG_OPENAI_BASE_URL`` and
``RAG_OPENAI_API_KEY``, and run.
"""
