# Generation and the chat interface

The package retrieves; it does not generate. This document covers the two
pieces on the far side of that boundary: `rag.generation`, which answers from a
`PromptContext`, and the Chainlit application in [`app/`](../app), which puts a
chat window in front of it.

Both are optional. Retrieval works, and is tested, with no model service
configured and the interface not installed (`pip install -e ".[ui]"` adds it).

## The boundary

```text
                    ┌───────────────────── src/rag ─────────────────────┐
  question ────────>│ retrieval ──> PromptContext ──> generation        │──> AnswerDelta stream
                    │  (unchanged)     (domain)        (sibling package) │
                    └───────────────────────────────────────────────────┘
                                                             │
                              app/main.py  ─────────────────┘   scripts/answer.py
                              (Chainlit, async)                  (terminal)
                                     │
                                     ▼
                          rag.feedback ──> results/feedback.jsonl
```

Three properties hold, two of them enforced by `tests/test_architecture.py`:

- **`generation` imports `domain` and `config`, and nothing else.** It cannot
  see `retrieval`, `ingestion`, or `storage`. It receives a `PromptContext` and
  has no way to reconstruct one.
- **The rendered prompt is sent verbatim.** `TemplatePromptAugmenter` builds a
  prompt with a structural boundary between application instructions and
  untrusted document content. A client that re-templated that text or split it
  across message roles would move a boundary it does not own, so none do: the
  prompt goes out as a single user message.
- **The interface lives outside the package.** `app/` assembles its own
  pipeline, exactly as `scripts/` does, through `rag.assembly`. Deleting it
  removes the interface and nothing else.

The interface is asynchronous and the package is not. `asyncio.to_thread`
bridges the two, in `app/main.py` alone: retrieval and generation run in a
worker thread, and each streamed fragment is pulled from the synchronous
generator the same way. No signature inside `src/rag` is coloured by it.

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

`app/main.py`. `chainlit run app/main.py`, or `just ui`, serves it at
`http://localhost:8000`. `just ask "your question"` runs the same loop without
a browser via `scripts/answer.py`. The database must be reachable and the model
service must already be running with the model available; neither is started
from here, and nothing is downloaded.

The interface loads the pipeline once per session, streams the answer, and:

- renders each cited passage as an inspectable side element, numbered from one
  to match the `[n]` markers in the prompt, showing the page, section path,
  reranker score, and document id;
- keeps the model's reasoning in a collapsed step above the answer;
- when retrieval found nothing, states so and does not call the model — an
  answer with no sources would be a guess;
- places passage text in a fenced code block, because it is untrusted document
  content and the interface renders Markdown: a document containing image or
  link syntax is displayed, not obeyed.

`scripts/answer.py` takes `--top-k`, `--document-id`, `--show-prompt`, and
`--show-reasoning`, and is the quickest way to see what the model was actually
sent.

## Feedback

`rag/feedback.py`. Each answer carries thumbs-up and thumbs-down buttons. A
click appends one JSON object to `RAG_FEEDBACK_PATH` (default
`results/feedback.jsonl`):

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
