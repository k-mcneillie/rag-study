# rag-study chat app

A standalone [Chainlit](https://chainlit.io) chat interface for the rag-study
API service. It talks to the service over HTTP and has **no dependency on the
`rag` package** — copy this directory into any repository, point it at a
running service, and it works.

## Layout

A standard `src/` Python project:

```
app/
├── pyproject.toml          # rag-chat-app: its own dependencies and tooling
├── docs/design.md          # why an API + async, and each file justified
├── src/rag_chat/
│   ├── main.py             # Chainlit handlers: start, stream, cite, rate
│   ├── client.py           # RagApiClient — the four API calls over httpx
│   ├── models.py           # the wire shapes (plain dataclasses)
│   ├── demo.py             # canned answer stream, used with no service
│   └── config.py           # AppConfig, read from the environment
└── tests/                  # pytest suite, travels with the app
```

For why this is an HTTP client rather than a script that imports `rag`, why it
is `async`, and what each file earns over a naive single-file version, see
[docs/design.md](docs/design.md).

## Run it

```bash
pip install -e ".[dev]"     # installs the rag_chat package and test tools
cp .env.example .env        # set RAG_API_URL and RAG_API_KEY
chainlit run src/rag_chat/main.py     # http://localhost:8000
```

`pip install -e .` is required: Chainlit executes `main.py` directly, and its
`from rag_chat...` imports resolve through the installed package.

The API service is started separately — in the rag-study repository, `just
serve` (see `docs/api.md` there). With no service reachable, or with
`RAG_CHAT_DEMO=1`, the app starts in **demo mode**: a canned answer streams
through the full UI (streaming, reasoning step, citations panel, rating
buttons) so the interface can be shown without a backend. Ratings are not
recorded in that mode.

On first run Chainlit writes a `.chainlit/` directory and a `chainlit.md`
next to wherever it is launched from; both are safe to commit or ignore.

## Configuration

| Variable         | Default                 | Meaning                                       |
|------------------|-------------------------|----------------------------------------------|
| `RAG_API_URL`    | `http://localhost:8080` | Base URL of the API service.                  |
| `RAG_API_KEY`    | *(empty)*               | Shared secret for the service's `X-API-Key`.  |
| `RAG_CHAT_TOP_K` | `5`                     | Passages to request per question.             |
| `RAG_CHAT_DEMO`  | *(off)*                 | Force demo mode (`1`/`true`/`yes`/`on`).      |

## Checks

```bash
ruff check . && ruff format --check . && mypy src/ tests/ && pytest
```

The tooling config in `pyproject.toml` mirrors the parent rag-study
repository: Ruff (PEP 8, Google-style docstrings) and mypy on Python 3.12+.
