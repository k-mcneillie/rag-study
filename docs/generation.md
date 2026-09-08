# Generation and the chat interface

The package retrieves; it does not generate. This document covers the pieces on
the far side of that boundary: `rag.generation`, which answers from a
`PromptContext`; the HTTP API in [`service/`](../service) that exposes
retrieval-and-answer over the network; and the Chainlit application in
[`app/`](../app), a chat window that talks only to that API.

All of it is optional. Retrieval works, and is tested, with no model service
configured, the API not run, and the interface not installed. The wire
contract, the endpoints, and how to run the two is [api.md](api.md); this
document is about the generation contract itself.

## The boundary

```text
             ┌──────────────────── src/rag ────────────────────┐
  question ─>│ retrieval ──> PromptContext ──> generation      │─> AnswerDelta stream
             │  (unchanged)     (domain)       (sibling)       │
             └────────────────────────────────────────────────┬┘
                                                              │
    scripts/answer.py (terminal)                    service/app.py  ──> SSE stream
                                                    (FastAPI, async)      │
                                                          ▼               ▼
                                          rag.feedback              app/  (Chainlit)
                                          results/feedback.jsonl    an HTTP client,
                                                                    no rag import
```

Three properties hold, two of them enforced by `tests/test_architecture.py`:

- **`generation` imports `domain` and `config`, and nothing else.** It cannot
  see `retrieval`, `ingestion`, or `storage`. It receives a `PromptContext` and
  has no way to reconstruct one.
- **The rendered prompt is sent verbatim, and never leaves the service.**
  `TemplatePromptAugmenter` builds a prompt with a structural boundary between
  application instructions and untrusted document content. The API exposes an
  *answer* endpoint that runs generation server-side; it has no endpoint that
  returns a `PromptContext`, so no client can re-template that text or split it
  across message roles.
- **Every consumer lives outside the package.** `scripts/` and `service/`
  assemble their own pipeline through `rag.assembly`; `app/` assembles nothing
  and imports `rag` not at all. Deleting any of them removes that consumer and
  nothing else.

`rag.generation` is synchronous and the API is not. `asyncio.to_thread` bridges
the two, in `service/streaming.py`: each streamed fragment is pulled from the
synchronous generator in a worker thread. No signature inside `src/rag` is
coloured by it. `scripts/answer.py` consumes the same synchronous stream
directly.

## The generation contract

`rag/generation/interfaces.py`.

`BaseChatModel` exposes `model_name` and an abstract `stream(context)` that
yields `AnswerDelta` fragments, each marked as reasoning or answer. A concrete
`answer(context)` consumes that stream into an `Answer`. `GenerationError` is
raised when the service is unreachable, rejects the request, or streams
something unintelligible; its message names the service and the model, and
never quotes the prompt or the answer, because the prompt carries retrieved
document content.

## Choosing a model service

Set `RAG_LLM_PROVIDER`. It names a protocol, not a vendor.

| Provider | Speaks to | Notes |
|---|---|---|
| `ollama` | Ollama's native `/api/chat` | Default. Preferred for a reasoning model: reasoning returns in its own field rather than in `<think>` tags. Accepts `num_ctx`. |
| `vllm` | vLLM's OpenAI-compatible server | Point `RAG_LLM_BASE_URL` at the `/v1` root. Start vLLM with a reasoning parser to separate reasoning from the answer. |
| `openai` | Any OpenAI chat-completions API | LM Studio, llama.cpp's server, OpenAI, Together, Groq, OpenRouter. |

`vllm` and `openai` use the same client; they are kept as separate names so a
reader configuring vLLM finds it in the list.

The remaining variables are in `.env.example`: `RAG_LLM_BASE_URL`,
`RAG_LLM_MODEL`, `RAG_LLM_API_KEY`, `RAG_LLM_TIMEOUT_SECONDS`,
`RAG_LLM_TEMPERATURE`, `RAG_LLM_NUM_CTX`, `RAG_LLM_THINKING`. Defaults are in
[../README.md](../README.md#configuration).

`RAG_LLM_NUM_CTX` matters. A prompt carrying several retrieved passages is
easily long enough to be truncated by a default context window, and truncation
is silent — the passages simply never reach the model. It is sent only by the
Ollama client; the OpenAI protocol has no equivalent, and a service that sizes
its own context is trusted to do so.

## Provider selection

`rag/generation/providers.py` holds `CHAT_MODELS`, a dictionary mapping the
configured name to a client class, and `chat_model_for`, which looks up the
client and raises `GenerationError` naming the valid options if the name is
unknown. This is the one registry in the codebase. It exists because which
service answers is a deployment decision, and requiring a code edit to change
it would make the swappable interface swappable only in principle.

## Shared transport

`rag/generation/transport.py` holds what every client needs: build the
Authorization header (a bearer token when `RAG_LLM_API_KEY` is set, omitted
entirely when not), post the request, refuse to continue on an error status,
and yield the non-blank lines of the streamed body. `_check_status` distinguishes
three cases a reader can act on:

| Status | Message |
|---|---|
| 404 | `Model <model> is not available at <url>. Provision it there first; it is never downloaded automatically.` |
| 401 / 403 | `The model service at <url> rejected the credential (HTTP <code>). Check RAG_LLM_API_KEY.` |
| other error | `The model service at <url> rejected the request for model <model> (HTTP <code>).` |

The request body — which contains retrieved document content — and the
credential are never quoted in an exception.

## The two clients

### `OllamaChatModel`

`rag/generation/ollama.py`. Posts to `/api/chat` with `stream: true`, `think`
set from `RAG_LLM_THINKING`, and `options.num_ctx`. Reasoning arrives in the
`message.thinking` field and the answer in `message.content`; the two are
separated by the protocol, not by pattern-matching.

### `OpenAICompatibleChatModel`

`rag/generation/openai_compatible.py`. Posts to `/v1/chat/completions` (or
`/chat/completions` when the base URL already ends in `/v1`) and parses
server-sent events. Reasoning is read from a `reasoning_content` or `reasoning`
field when the service separates it; a service that instead inlines `<think>`
tags is handled by splitting on the tags, including when a tag is itself split
across two streamed fragments. The client carries per-stream state and resets
it at the start of each `stream()` call.

### Verified and not verified

The Ollama client is exercised end to end against a real server in
`tests/integration/test_ollama_chat.py`.

vLLM is supported by protocol and has not been run against a live vLLM server.
It is reached through the same client as every other OpenAI-compatible service,
and the unit tests pin the parts its behaviour depends on: the endpoint path,
the `/v1` base-URL convention, server-sent-event framing, bearer
authentication, and the `reasoning_content` field a reasoning parser emits. If
it misbehaves, the fault is in `openai_compatible.py` alone.

## Adding a service

Write a `BaseChatModel` — `model_name` and `stream()` — and add one line to
`CHAT_MODELS`. The HTTP plumbing is already shared in `transport.py`, so a new
client is usually the request body and the response shape and nothing else.

## The chat interface

`app/` is a standalone Chainlit client of the API — its own `src/` project with
its own `pyproject.toml` and no dependency on `rag`. `just serve` runs the API,
then `just ui` (`chainlit run app/src/rag_chat/main.py`) serves the chat at
`http://localhost:8000`. `just ask "your question"` runs the loop in-process,
without either, via `scripts/answer.py`. The full run and configuration is in
[api.md](api.md).

Per session the app calls `GET /health` for its banner, then for each question
streams `POST /answer` and:

- renders each cited passage as an inspectable side element, numbered from one
  to match the `[n]` markers in the answer, showing the page, section path, and
  reranker score. The passage *text* stays on the service — the app is given
  only provenance — so the panel says where a passage came from, not what it
  said;
- keeps the model's reasoning in a collapsed step above the answer;
- on a `no_context` event, states that nothing matched and does not show an
  answer — the service never called the model;
- with no service reachable, or `RAG_CHAT_DEMO=1`, streams a canned answer
  through the same UI so the interface can be demonstrated with no backend.
  Ratings are not recorded in that mode.

`scripts/answer.py` takes `--top-k`, `--document-id`, `--show-prompt`, and
`--show-reasoning`, and is the quickest way to see what the model was actually
sent.

## Feedback

`rag/feedback.py`. Each answer carries thumbs-up and thumbs-down buttons. The
app posts the rating and the fields the log records to `POST /feedback`; the
service builds the record and appends one JSON object to `RAG_FEEDBACK_PATH`
(default `results/feedback.jsonl`). No passage text crosses the wire — only the
provenance the log already keeps.

```json
{
  "recorded_at": "2026-09-05T10:31:22.481047+00:00",
  "vote": "up",
  "query": "How is the sampling frame defined?",
  "answer": "Part-time staff were excluded [1].",
  "model_name": "deepseek-r1:14b",
  "prompt_name": "retrieval_context",
  "prompt_version": "v1",
  "citations": [
    {"position": 1, "document_id": "…", "page_number": 12,
     "section": "2 Methods > 2.1 Sampling", "score": 0.88}
  ]
}
```

Two decisions are deliberate:

- **Citations record provenance, never passage text.** The provenance locates a
  passage exactly, and the corpus is not something to copy into a second place
  on disk.
- **The log is gitignored.** It holds question and answer text drawn from the
  corpus, so it is treated the way the corpus is.

The prompt version and model name are recorded with every rating, so ratings
collected across a prompt or model change stay distinguishable. This is the
seed for the evaluation harness in [future-work.md](future-work.md) §1.

Chainlit's own thumb icons are not used: they require a registered data layer,
which means a Chainlit schema created with DDL rights the application user
deliberately does not have. Explicit action buttons need none of that.
