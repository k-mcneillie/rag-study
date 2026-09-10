# Why this app is shaped the way it is

This is `app/` with one difference: it speaks the **OpenAI wire language**
instead of this repository's bespoke one. Everything `app/`'s
[`docs/design.md`](../../app/docs/design.md) argues — an HTTP API rather than an
in-process `rag` import, `async` because Chainlit is async and the work is I/O,
one responsibility per module — applies here unchanged and is not repeated. This
document covers only what is different.

---

## What "the OpenAI language" means

Not a vendor — a wire contract that most LLM servers now clone. The target here
is a **local** OpenAI-compatible server (vLLM, Ollama's `/v1`, LM Studio,
llama.cpp's server) or a local RAG service speaking that dialect — not the
hosted OpenAI platform, so nothing in this client touches Assistants,
`/v1/vector_stores`, or hosted `file_search`. The contract it relies on:

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
prompt or raw passage text. That is both the ordinary shape for a RAG service
and the trust boundary this repository documents in `docs/architecture.md`:
nothing the client is handed lets it re-template the prompt. So the chat request
carries only `model`, `messages` and `stream` — there is no retrieval-config
knob to add, because the provider is never told how to retrieve.

### Citations ride as a vendor extension

The OpenAI chat schema has no slot for `{document_id, page_number, section,
score}`. The local RAG provider — the one we will build — therefore adds a
`citations` array to the stream, the same fields `_source_elements` renders,
emitted on the first chunk before any answer content. `_StreamDecoder`
tolerates it at the top level or inside `choices[0].delta` until the provider
pins the shape. Bare OpenAI `annotations` (`file_citation` / `url_citation`)
are deliberately **not** handled: the provider contract commits to the
extension. If that ever changes, `_StreamDecoder._decode_chunk` is the one
place to extend.

### Reasoning comes from `reasoning_content` only

Reasoning is read from the `reasoning_content` delta field (the vLLM / DeepSeek
spelling; `reasoning` is also accepted). A provider that instead inlines
reasoning as `<think>` tags in `content` is **not** handled — an earlier
version split those out, and the code is easy to restore, but it is dead weight
against a server that separates the field. `_StreamDecoder._decode_chunk` is
where a `<think>` splitter would go back.

---

## One decoder, shared by the live stream and the demo

`client.py` owns `_StreamDecoder`. It turns one `data:` payload into zero or
more `Event`s — a single frozen dataclass (`kind`, `text`, `citations`), where
`app/` uses a five-type event union — and keeps every raw payload in `.trace`.

```
live:  httpx aconnect_sse ─▶ sse.data ─┐
                                        ├─▶ _StreamDecoder.decode(data) ─▶ Event… ─▶ _stream_answer ─▶ UI
demo:  demo.demo_lines()   ─▶ raw str ─┘        (each payload also appended to .trace)
```

The consequence: **demo mode exercises the exact translation code production
uses.** `demo.py` does not hand the UI pre-baked events the way `app/`'s does; it
emits canned `chat.completion.chunk` JSON — a citations chunk, two
`reasoning_content` chunks, three `content` chunks, a terminal chunk, then
`[DONE]` — and the real decoder turns them into the stream the renderer sees. A
bug in the reasoning split or the citations mapping shows up in the offline
demo, not only against a live provider.

`_StreamState`, `_source_elements` and the rating callbacks carry over from
`app/`: they iterate an `Event` async iterator and do not care where the events
came from. `app/`'s separate `_consume` step is folded into `_stream_answer`
here — one loop instead of two hops.

### The "OpenAI wire trace" element

After each answer, **in demo mode**, `main.py` renders `decoder.trace` in a
collapsed `cl.Step` — one pretty-printed `chat.completion.chunk` per line,
ending with `data: [DONE]`. It makes the demo double as a readable specimen of
what the provider must send. There is no live-answer trace toggle: a prototype
that needs to inspect a live stream can read the same `decoder.trace` from a
breakpoint.

---

## What differs from `app/`, file by file

| File | Same as `app/`? | Difference |
|---|---|---|
| `models.py` | No | `app/`'s five-type `StreamEvent` dataclass union collapses to one `Event(kind, text, citations)` dataclass, plus `Citation`. No `HealthInfo` (the banner shows only the model name), no prompt-template id (the OpenAI dialect has none). |
| `client.py` | No | `_StreamDecoder` (OpenAI chunk → `Event`) replaces `_parse_event` (named event → `StreamEvent`). `Authorization: Bearer` not `X-API-Key`. `GET /v1/models` not `GET /health`. Keeps `ingest` (just `POST /v1/files`) and `feedback`. |
| `demo.py` | No | Emits raw `chat.completion.chunk` payloads for the shared decoder, not events directly. |
| `config.py` | Shape only | `RAG_OPENAI_*` variables: `BASE_URL`, `API_KEY`, `MODEL`, `FEEDBACK_URL`, `DEMO`. |
| `main.py` | Mostly | Banner from `models()`; demo-only wire-trace step; a file-upload affordance for ingestion (the one intentional UI addition over `app/`); a no-op feedback endpoint tolerated. `_consume` is folded into `_stream_answer`. `_StreamState` and the render path otherwise match `app/`. |

### The cost of a separate directory

`main.py` and the demo scaffolding began as near-copies of `app/`'s, and the
directory is self-contained (own `pyproject.toml`, no shared code, no `rag`
import) so it can be lifted into another repository whole. The two have since
diverged more than the wire dialect alone forced — the `Event` `NamedTuple`, the
folded stream loop, the trimmed config — because for this client simplicity was
worth more than staying in lockstep with `app/`. A shared-UX change (the
reasoning-step ordering, a citation panel's layout, the rating flow) still has
to be made in both, but the two are no longer meant to be identical.
