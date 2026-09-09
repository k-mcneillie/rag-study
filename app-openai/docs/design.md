# Why this app is shaped the way it is

This is `app/` with one difference: it speaks the **OpenAI wire language**
instead of this repository's bespoke one. Everything `app/`'s
[`docs/design.md`](../../app/docs/design.md) argues — an HTTP API rather than an
in-process `rag` import, `async` because Chainlit is async and the work is I/O,
one responsibility per module — applies here unchanged and is not repeated. This
document covers only what is different.

---

## What "the OpenAI language" means

Not a vendor — a wire contract that most LLM servers now clone (vLLM, Ollama's
`/v1`, LM Studio, llama.cpp's server, OpenAI, Together, Groq, OpenRouter, and
hosted RAG services built on any of them):

- `POST /v1/chat/completions` with `{"model", "messages":[{"role","content"}], "stream": true}`.
- The response is `text/event-stream`: one unnamed `data: {chunk}` per line,
  each chunk `{"choices":[{"delta":{"content": "..."}}]}`, ending with a literal
  `data: [DONE]`.
- `Authorization: Bearer <key>`.
- `GET /v1/models` → `{"object":"list","data":[{"id": "..."}]}`.
- Errors: `{"error":{"message": "..."}}`, mid-stream as a chunk or before the
  stream opens with a 4xx/5xx.

`app/` instead consumes *named* SSE events (`event: citations` / `delta` /
`done` / `no_context` / `error`) from this repo's `service/`. The two clients are
otherwise the same program.

### Retrieval stays server-side

The provider **owns its index and retrieves internally through its own
endpoints**. This client sends only the question; it never receives an assembled
prompt or raw passage text. That is both best practice for a hosted RAG service
(OpenAI `file_search`, Azure "on your data", Bedrock `RetrieveAndGenerate` all
work this way) and the trust boundary this repository documents in
`docs/architecture.md`: nothing the client is handed lets it re-template the
prompt.

### Citations ride as a vendor extension

The OpenAI chat schema has no slot for `{document_id, page_number, section,
score}`. A RAG provider that speaks OpenAI therefore adds a `citations` array to
the stream — the same fields `_source_elements` renders — emitted on the first
chunk, before any answer content. `_StreamDecoder` reads it there. The one thing
deliberately **not** handled is bare OpenAI `annotations` (`file_citation` /
`url_citation`) with no `citations` extension: the provider contract commits to
the extension, and a branch for a shape we were told will not occur is
complexity for nothing. If a future provider only sends annotations, that is the
one place to extend `_StreamDecoder._maybe_citations`.

---

## One decoder, shared by the live stream and the demo

`client.py` owns `_StreamDecoder`. It turns one `data:` payload into zero or
more `StreamEvent`s — the app's own event model, identical to `app/`'s — and
keeps every raw payload in `.trace`.

```
live:  httpx aconnect_sse ─▶ sse.data ─┐
                                        ├─▶ _StreamDecoder.decode(data) ─▶ StreamEvent… ─▶ _consume ─▶ UI
demo:  demo.demo_lines()   ─▶ raw str ─┘        (each payload also appended to .trace)
```

The consequence: **demo mode exercises the exact translation code production
uses.** `demo.py` does not hand the UI pre-baked events the way `app/`'s does; it
emits canned `chat.completion.chunk` JSON — a citations chunk, two
`reasoning_content` chunks, three `content` chunks, a terminal chunk, then
`[DONE]` — and the real decoder turns them into the stream the renderer sees. A
bug in the `<think>` splitter or the citations mapping shows up in the offline
demo, not only against a live provider.

`_consume`, `_StreamState`, `_stream_answer`, `_source_elements` and the rating
callbacks are unchanged from `app/`: they still iterate a `StreamEvent` async
iterator and do not know or care where the events came from.

### The "OpenAI wire trace" element

After each answer, in demo mode or when `RAG_OPENAI_TRACE=1`, `main.py` renders
`decoder.trace` in a collapsed `cl.Step` — one pretty-printed
`chat.completion.chunk` per line, ending with `data: [DONE]`. It makes the demo
double as a readable specimen of what the provider must send, and gives a live
session a way to see the raw dialect when a translation looks wrong.

---

## The one provider-specific knob

`OpenAIRagClient._retrieval_config()` returns `{}`. A provider that owns its
index retrieves without being told how, so the chat request carries only
`model`, `messages` and `stream`. When a concrete provider needs
`tool_resources` / `vector_store_ids` / a `top_k` hint, this method is the single
place it goes; the values are already on the client (`RAG_OPENAI_VECTOR_STORE`,
`RAG_OPENAI_TOP_K`). Isolating it here keeps the rest of the client
provider-agnostic.

---

## What differs from `app/`, file by file

| File | Same as `app/`? | Difference |
|---|---|---|
| `models.py` | Almost | `DoneEvent.prompt` is `PromptId \| None` (OpenAI has no prompt-template id); `HealthInfo` fields beyond `model` are optional. The `StreamEvent` union is identical. |
| `client.py` | No | `_StreamDecoder` (OpenAI chunk → `StreamEvent`) replaces `_parse_event` (named event → `StreamEvent`). `Authorization: Bearer` not `X-API-Key`. `GET /v1/models` not `GET /health`. Adds `_retrieval_config`, keeps `ingest` / `feedback` / a standalone `search`. |
| `demo.py` | No | Emits raw `chat.completion.chunk` payloads for the shared decoder, not `StreamEvent`s directly. |
| `config.py` | Shape only | `RAG_OPENAI_*` variables: `BASE_URL`, `API_KEY`, `MODEL`, `VECTOR_STORE`, `TOP_K`, `FEEDBACK_URL`, `TRACE`, `DEMO`. |
| `main.py` | Nearly verbatim | Banner from `models()`; wire-trace step; a file-upload affordance for ingestion (the one intentional UI addition over `app/` — `app/` could adopt it); `done.prompt` and a no-op feedback endpoint tolerated. `_StreamState` and the render path are unchanged. |

### The cost of a separate directory

`main.py`, `models.py` and the demo scaffolding are near-copies of `app/`'s. This
is a deliberate choice: the directory is self-contained (own `pyproject.toml`, no
shared code, no `rag` import) so it can be lifted into another repository whole.
The price is that a change to the shared UX — the reasoning-step ordering, a
citation panel's layout, the rating flow — must be made in both `app/` and
`app-openai/`. That trade was made knowingly; if the two ever need to diverge
further, the duplication stops being a cost.
