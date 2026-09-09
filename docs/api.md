# The HTTP API and the chat app

`service/` puts the query side of the package behind four HTTP endpoints.
`app/` is a Chainlit chat window that talks to those endpoints and nothing
else. Both are entry points, like `scripts/` — outside the package, deletable
without touching `src/rag`.

The point of the split is portability: `app/` is its own `src/` project with
its own `pyproject.toml` and no `rag` import, so it can be copied into another
repository and pointed at a deployed service. Its
[`docs/design.md`](../app/docs/design.md) justifies that choice — and `async`,
and each of its files — against a naive single-file version.

## Running it

Everything runs from the conda environment (`environment.yml`). Install the API
extra and the app once:

```bash
pip install -e ".[dev,api,ui]"    # the package + API deps
pip install -e app/               # the chat app (Chainlit loads it as a package)
```

Then:

```bash
just serve      # uvicorn service.app:create_app --factory  → http://localhost:8080
just ui         # chainlit run app/src/rag_chat/main.py      → http://localhost:8000
```

`just serve` needs the database reachable and the embedding model — and the
reranker, if one is configured — present. It builds the whole pipeline at
startup and exits non-zero if anything is missing; nothing is downloaded. The
answering model service (Ollama or an OpenAI-compatible server) must already be
running, exactly as for `scripts/answer.py`.

With no service reachable, or `RAG_CHAT_DEMO=1`, `just ui` starts in **demo
mode**: a canned answer streams through the full UI so the interface can be
shown with no backend. Ratings are not recorded in that mode.

## Authentication

One optional shared secret, `RAG_API_KEY`:

- **unset** — the service runs open; every route is reachable without a
  credential. Fine on a local machine, a footgun anywhere else. The startup log
  says so.
- **set** — every route except `GET /health` requires the value in an
  `X-API-Key` header, compared with `hmac.compare_digest`. A wrong or missing
  key gets `401`.

There is nothing finer: no per-user identity, no rotation, no rate limiting, no
TLS. Put a reverse proxy in front for the last two before exposing the service.
See [future-work.md](future-work.md) §3.

## Endpoints

### `GET /health`

Open. Reports what the service is configured to do.

```json
{"status": "ok", "model": "deepseek-r1:14b", "provider": "ollama",
 "reranker": "cross-encoder", "prompt": "retrieval_context/v1", "top_k": 5}
```

### `POST /answer` → `text/event-stream`

Body `{"query": str, "top_k": int | null}`. Retrieval runs, then generation, and
the answer streams back as server-sent events. Generation happens entirely
server-side: there is **no** endpoint that returns the assembled
`PromptContext`, because a client that could rebuild the prompt from parts
could move the boundary between instructions and document content (see
[architecture.md](architecture.md) §4).

| Event        | Data                                                        | When |
|--------------|-------------------------------------------------------------|------|
| `citations`  | `{"items": [{position, document_id, page_number, section, score}]}` | once, after retrieval, before the model is called — provenance only, never passage text |
| `delta`      | `{"reasoning": bool, "text": str}`                          | one per fragment |
| `done`       | `{"model_name": str, "prompt": {"name", "version"}}`        | once, after the last fragment |
| `no_context` | `{}`                                                        | instead of everything after `citations`, when retrieval matched nothing — the model was never called |
| `error`      | `{"message": str}`                                          | instead of `done`, when generation fails; the message is already scrubbed of prompt and answer text |

An empty query, or a `top_k` below 1, is `422` before the stream opens.

### `POST /feedback` → `204`

Body carries everything the feedback log records — `vote` (`up`/`down`),
`query`, `answer`, `model_name`, `prompt`, and `citations` (provenance only).
The service builds a `rag.feedback.FeedbackRecord` and appends one JSON line to
`RAG_FEEDBACK_PATH`. No passage text crosses the wire; the log is gitignored
and treated like the corpus.

Correlation is stateless — the app posts the fields back rather than the
service holding recent turns in memory. A `503` means the log is unwritable.

### `POST /documents`

`multipart/form-data` with one `file`. Must be a PDF within
`RAG_MAX_DOCUMENT_BYTES`. The service spools it to a temp file and runs the
ingestion pipeline inline, blocking until it is indexed.

| Status | Body |
|---|---|
| `201` | `{"status": "ingested", "document_id": str, "chunk_count": int}` |
| `200` | `{"status": "already_indexed", "document_id": null, "chunk_count": null}` |
| `415` | not a PDF |
| `413` | past the size limit |
| `422` | the document yielded no usable text |

The embedding model this needs is not loaded at startup — it is built on the
first upload and reused. A larger corpus would want an accepted +
background-job shape instead; see [future-work.md](future-work.md) §2.

## Configuration

The service reads every `RAG_*` variable the rest of the system does. Its own:

| Variable | Default | Meaning |
|---|---|---|
| `RAG_API_KEY` | *(unset — open)* | shared secret for `X-API-Key` |
| `RAG_API_HOST` | `127.0.0.1` | bind address (used by `just serve`) |
| `RAG_API_PORT` | `8080` | bind port |

The app (`app/.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `RAG_API_URL` | `http://localhost:8080` | which service to call |
| `RAG_API_KEY` | *(empty)* | must match the service's, when it has one |
| `RAG_CHAT_TOP_K` | `5` | passages to request per question |
| `RAG_CHAT_DEMO` | *(off)* | force demo mode |

## What the app does with a stream

Per session it calls `GET /health` for its banner, then for each question
consumes `POST /answer` and renders the reasoning in a collapsed step, the
answer as it streams, and each citation as a numbered side panel showing page,
section, and score — location, not text. `no_context` becomes a plain "nothing
matched" message; `error` becomes "**No answer.** …". The rating buttons `POST
/feedback`; in demo mode they show a toast and write nothing.

## A second client: the OpenAI dialect (`app-openai/`)

`app-openai/` is a sibling of `app/` — the same Chainlit UI — written against a
different wire contract. Instead of this repo's `service/` and its named SSE
events, it speaks the **OpenAI wire language** to a hosted RAG provider:

- `POST /v1/chat/completions` with `stream: true`, an SSE loop over
  `data: {chunk}` … `data: [DONE]`; `Authorization: Bearer`; `GET /v1/models`
  for the banner.
- The provider owns its index and does retrieval internally — the client sends
  only the question, and never receives an assembled prompt (the same trust
  boundary as `service/`).
- Citations ride as a `citations` vendor extension on the stream, carrying
  `{position, document_id, page_number, section, score}`; reasoning as the
  `reasoning_content` delta field or inline `<think>` tags.
- `POST /v1/files` for ingestion; an optional `RAG_OPENAI_FEEDBACK_URL` for the
  rating buttons (empty stores nothing).

Its `client.py` owns a `_StreamDecoder` that turns one `chat.completion.chunk`
into the same internal event union `app/` uses; the live stream and the offline
demo (`RAG_OPENAI_DEMO=1`) both run through it, and an "OpenAI wire trace" step
shows the raw chunks. Configuration is `RAG_OPENAI_BASE_URL` / `_API_KEY` /
`_MODEL` / `_VECTOR_STORE` / `_TOP_K` / `_FEEDBACK_URL` / `_TRACE` / `_DEMO`.
Run it with `just ui-openai`. Its design notes are in
[`app-openai/docs/design.md`](../app-openai/docs/design.md).

## Tests

`tests/service/` exercises every route against fake collaborators and a
`TestClient` — no database, no weights, no network. `app/tests/` covers the
client's SSE parsing, the demo stream, and that `main.py` loads under Chainlit's
module loader. `app-openai/tests/` does the same for the OpenAI-dialect decoder.
All run in `just check-all`.
