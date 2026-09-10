# rag-study chat app — OpenAI dialect

A standalone [Chainlit](https://chainlit.io) chat interface for a RAG service
that speaks the **OpenAI wire language** (`POST /v1/chat/completions` streaming,
`Authorization: Bearer`, `GET /v1/models`). It talks to the provider over HTTP
and has **no dependency on the `rag` package** — copy this directory into any
repository, point it at a provider, and it works.

It is the sibling of [`app/`](../app): the same Chainlit UI — streamed answers,
a collapsed reasoning step, numbered citation side-panels with page/section/score,
thumbs up/down rating buttons, offline demo mode — differing in the wire
language it speaks, and a little leaner inside. `app/` talks to this
repository's own `service/`; this one talks to a local OpenAI-compatible
provider that maintains its own index and does retrieval internally.

## Layout

A standard `src/` Python project:

```
app-openai/
├── pyproject.toml            # rag-chat-openai: its own dependencies and tooling
├── docs/design.md            # the wire dialect, the shared decoder, each file justified
├── src/rag_chat_openai/
│   ├── main.py               # Chainlit handlers: start, stream, cite, rate, upload
│   ├── client.py             # OpenAIRagClient + _StreamDecoder — the wire translation
│   ├── models.py             # Citation and the decoder's Event type (frozen dataclasses)
│   ├── demo.py               # canned OpenAI-shaped chunk stream, used with no provider
│   └── config.py             # AppConfig, read from the environment
└── tests/                    # pytest suite, travels with the app
```

See [docs/design.md](docs/design.md) for why this is an HTTP client rather than a
script that imports `rag`, why it is `async`, why the answer stream and the demo
share one decoder, and what the near-duplication of `app/` costs.

## Run it

```bash
pip install -e ".[dev]"     # installs the rag_chat_openai package and test tools
cp .env.example .env        # set RAG_OPENAI_BASE_URL and RAG_OPENAI_API_KEY
chainlit run src/rag_chat_openai/main.py     # http://localhost:8000
```

`pip install -e .` is required: Chainlit executes `main.py` directly, and its
`from rag_chat_openai...` imports resolve through the installed package.

With no provider reachable, or with `RAG_OPENAI_DEMO=1`, the app starts in
**demo mode**: a canned OpenAI-shaped chunk stream is decoded through the real
client, so the full interface — streaming, reasoning step, citation panels,
rating buttons, and the **"OpenAI wire trace"** step showing the raw
`chat.completion.chunk` lines — can be shown without a backend. Ratings are not
recorded in that mode.

On first run Chainlit writes a `.chainlit/` directory and a `chainlit.md` next to
wherever it is launched from; both are safe to commit or ignore.

## Configuration

| Variable                  | Default                   | Meaning                                                        |
|---------------------------|---------------------------|---------------------------------------------------------------|
| `RAG_OPENAI_BASE_URL`     | `https://api.openai.com`  | Provider root, with or without a trailing `/v1`.              |
| `RAG_OPENAI_API_KEY`      | *(empty)*                 | Bearer token. Empty for a provider that needs none.          |
| `RAG_OPENAI_MODEL`        | *(empty)*                 | Model id for the request; empty adopts the first `/v1/models` entry. |
| `RAG_OPENAI_FEEDBACK_URL` | *(empty)*                 | Endpoint the rating buttons POST to; empty stores nothing.   |
| `RAG_OPENAI_DEMO`         | *(off)*                   | Force demo mode (`1`/`true`/`yes`/`on`).                     |

## Checks

```bash
ruff check . && ruff format --check . && mypy src/ tests/ && pytest
```

The tooling config in `pyproject.toml` mirrors the parent rag-study repository:
Ruff (PEP 8, Google-style docstrings) and mypy on Python 3.12+.
