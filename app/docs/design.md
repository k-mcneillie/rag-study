# Why this app is shaped the way it is

This is a chat window. A naive version is one file: a Chainlit handler that
imports `rag`, builds a retrieval pipeline, calls a model, and prints the
answer. That version exists in this project's history. This document explains
what each piece of the current shape buys, and why the naive version was not
kept.

The short answer: the naive version couples the UI to the package, cannot be
deployed or reused independently, blocks its own event loop, and mixes five
concerns into one file that cannot be tested without a database and model
weights. Each of those is fixed by one deliberate piece below.

---

## Why an HTTP API at all, instead of importing the package

The naive app does `from rag.assembly import build_retrieval_orchestrator` and
runs the whole pipeline in-process. That is simpler to write and wrong for
this project for four concrete reasons:

1. **The UI and the retrieval engine have different lifecycles.** The engine
   needs a database connection, ~100 MB of model weights on disk, and a GPU or
   a warm CPU. The UI needs a browser. Bundling them means every place that
   wants a chat window also has to provision MariaDB and the models. Splitting
   them means the engine is deployed once and any number of front-ends —
   this one, a Slack bot, a batch script — talk to it over HTTP.

2. **Portability was the explicit goal.** The requirement was "take the app
   and move it to any other repo." With an in-process pipeline that is
   impossible: the app *is* a second entry point of the `rag` package and
   drags the entire dependency tree (SQLAlchemy, PyMuPDF, sentence-transformers)
   with it. As an HTTP client its dependency list is four packages
   (`chainlit`, `httpx`, `httpx-sse`, `python-dotenv`) and it has no knowledge
   of how answers are produced.

3. **The trust boundary must not be reconstructible by a client.** The
   retrieval pipeline builds a prompt with a structural separator between
   application instructions and untrusted document text, and that prompt is
   sent to the model verbatim (see the main repo's `docs/security.md`). If the
   app could call `retrieve()` and get back the assembled context, it could
   re-template that text and move a security boundary it does not own. The API
   deliberately exposes only an *answer* endpoint — generation happens on the
   server — so there is no client-side seam to attack.

4. **The package's architecture rule forbids it.** `tests/test_architecture.py`
   in the parent repo asserts that nothing under `src/rag` imports a web
   framework, and (now) that nothing depends back on `app/` or `service/`. The
   API is the sanctioned way for a UI to reach the pipeline without either
   side importing the other.

The cost is real — a second process to run, a wire format to keep in step, a
network hop of latency — and it is paid deliberately. For a single-user local
demo the naive version would be fine; for something meant to be deployed and
reused it is not.

### Why FastAPI for that API (in `service/`, not here)

The server lives in the parent repo's `service/`, not in this package, but the
choice affects this app so it is worth stating. FastAPI was chosen over a
hand-rolled `http.server`, Flask, or Django because:

- **Streaming is first-class.** `sse-starlette`'s `EventSourceResponse` turns
  an async generator into a correct `text/event-stream` with keep-alives and
  client-disconnect handling. Doing SSE by hand — chunked encoding, framing,
  flush discipline — is exactly the kind of solved problem this project keeps
  behind a library.
- **The request/response types are the contract.** Pydantic models in
  `service/schemas.py` validate input and generate the OpenAPI schema, so the
  wire format is checked, not just documented.
- **`TestClient` needs no running server.** The API's tests exercise every
  route in-process against fakes — no socket, no database.
- **It is async-native**, which matters because the answer endpoint streams
  tokens for tens of seconds and the server must stay responsive to other
  callers meanwhile.

Flask would need an extension for each of those; a bare `http.server` would
need all of it written by hand and tested.

---

## Why `asyncio`, when the `rag` package is deliberately synchronous

The `rag` package is synchronous on purpose — its work is CPU-bound model
inference, where `async` would colour every signature for no throughput gain
(the parent repo's `docs/future-work.md` says so explicitly). So why is this
app async?

**Because Chainlit is async, and the thing it is waiting on is a network
stream.** An answer arrives as server-sent events over tens of seconds. The
event loop model is the right tool for "hold a connection open and react to
each fragment as it lands" — it is I/O-bound waiting, not computation. The
naive synchronous alternative is one of:

- **Block the handler until the whole answer is ready.** The user stares at a
  spinner for 30 seconds with no feedback, and a second user connecting in
  that window is stuck behind the first. Chainlit's whole value — streaming
  tokens, a live reasoning step — is thrown away.
- **Spawn a thread per request and marshal fragments back.** That is
  re-implementing an event loop, badly, with locking bugs to find.

With `async`, `_consume()` is a plain `async for` over the event stream, each
`await` yields the loop back to Chainlit to paint the partial UI, and a slow
model does not freeze the interface or other sessions.

The key point: **the app is async where the work is waiting on I/O, and the
package is sync where the work is computation.** The one place they would have
met — pulling fragments from a synchronous generator inside an async handler —
does not exist here, because the fragments arrive over HTTP. (In `service/`,
where a synchronous `rag.generation` stream *is* consumed inside async code,
`asyncio.to_thread` bridges the two in exactly one file.)

`httpx` (async) is used over `requests` for the same reason: `requests` has no
async API, so streaming an SSE response with it would block the loop and undo
the point.

---

## Why the package is split into five files

The naive app is one module. Splitting it into `models`, `client`, `demo`,
`config`, and `main` follows the same rule the parent repo uses everywhere:
**one responsibility per module, and a dependency that only points one way.**

```
config ─────────────┐
models ──┬── client ─┼── main
         └── demo ───┘
```

| File | Responsibility | Why it is separate |
|---|---|---|
| `models.py` | The wire shapes: `Citation`, `HealthInfo`, and the `StreamEvent` union (`CitationsEvent` / `DeltaEvent` / `DoneEvent` / `NoContextEvent` / `ErrorEvent`), as frozen dataclasses. | This is the contract with the service. It has zero imports beyond `dataclasses`, so both `client` (which parses into these) and `demo` (which produces them) can depend on it without a cycle, and a reader can see the entire protocol in one screen. Plain dataclasses, not Pydantic: the app validates nothing — it trusts its own service — so a second validation library would be dead weight. |
| `client.py` | `RagApiClient`: the four API calls over one `httpx.AsyncClient`, the SSE parser (`_parse_event`), and `ApiError`. | All knowledge of *HTTP* — headers, status codes, event framing, the `X-API-Key` header set once on the client — lives here and nowhere else. `main.py` never sees a status code. Swapping transport (a Python SDK, a WebSocket) would touch this file only. The `transport=` seam on the constructor lets the tests drive it with `httpx.MockTransport` and no network. |
| `demo.py` | `demo_events()`: a canned `StreamEvent` stream. | Demo mode is a real feature — the UI must be demonstrable with no backend — but it is not *chat* logic. Keeping it in its own file means `main.py`'s `_consume()` has exactly one code path: it iterates a `StreamEvent` async iterator and does not know or care whether the events came from `client.answer_stream()` or `demo_events()`. The two produce the identical type, so there is no `if demo:` branching in the rendering code. |
| `config.py` | `AppConfig.from_env()`: `RAG_API_URL`, `RAG_API_KEY`, `RAG_CHAT_TOP_K`, `RAG_CHAT_DEMO`, with `.env` loading. | Environment reads happen in exactly one place, at startup, into a frozen object. Nothing downstream calls `os.environ`, so the settings a session is using cannot drift, and a test constructs an `AppConfig` directly instead of monkeypatching the environment. This mirrors `rag.config` in the parent repo. |
| `main.py` | The Chainlit handlers: `start`, `on_message`, the rating callbacks, and the streaming-render state machine (`_StreamState`, `_consume`, `_stream_answer`). | This is the only file that imports `chainlit`. It is the composition point — it wires `config` → `client`/`demo` → the UI — and holds the genuinely tricky part: rendering a reasoning step and an answer message in the order the work happened. That logic is documented at length in `_StreamState` because Chainlit's step-ordering is timestamp-based and non-obvious. |

What the split buys, concretely:

- **`models`, `client`, and `demo` are tested without Chainlit.** `app/tests/`
  drives the client against a mock transport and checks the demo stream's
  shape; only one smoke test loads `main.py`, and it needs Chainlit.
- **A change has a blast radius of one file.** New endpoint → `client.py` and
  `models.py`. New env var → `config.py`. Restyle a citation panel →
  `main.py`. The naive single file has to be re-read in full for any of them.
- **The cycle-free dependency arrow is enforceable.** `models` importing
  nothing app-specific is what lets `client` and `demo` share it; if rendering
  logic leaked into `models` that would break.

### Why a `src/` layout and a `pyproject.toml`

`app/src/rag_chat/` rather than loose files at `app/`:

- **`src/` layout prevents "works on my machine".** The package is only
  importable after `pip install -e .`, so the tests and `chainlit run`
  exercise the *installed* package, not whatever happens to be in the working
  directory. This is standard modern packaging practice for exactly this
  reason.
- **`pyproject.toml` makes the dependency set explicit and installable.** It
  is what lets the directory be copied elsewhere and stood up with one
  command. It also carries this project's own Ruff and mypy config, mirroring
  the parent repo, so the code holds the same bar after it has been moved out.
- **`chainlit run app/src/rag_chat/main.py` with absolute `from rag_chat...`
  imports.** Chainlit executes the entry file outside `sys.modules`; relative
  imports would fail, absolute imports through the installed package do not.
  This is also why `_Turn` and `_StreamState` in `main.py` are hand-written
  classes, not `@dataclass` — the decorator resolves string annotations via
  `sys.modules` and raises under Chainlit's loader.

---

## What was rejected

| Naive choice | Why not |
|---|---|
| One `main.py` importing `rag` | Couples UI to the engine; not deployable or reusable separately; drags the whole dependency tree; lets a client rebuild the prompt. |
| Synchronous handler, block until done | 30-second dead spinner; one slow answer blocks every other session; discards streaming. |
| `requests` instead of `httpx` | No async API; streaming SSE would block the event loop. |
| Pydantic for the app-side models | The app validates nothing it receives from its own service; a second validation dependency for no gain. |
| `if demo:` branches in the render code | Two code paths to keep in step. Instead `demo_events()` yields the same `StreamEvent` type as the real client, and the renderer has one path. |
| Flat modules, no `pyproject.toml` | Not installable, not portable, no pinned dependency set, imports depend on the working directory. |

## What stayed simple

The justification cuts both ways. This app has **no** state store, **no**
retry/backoff layer, **no** auth beyond forwarding one header, **no**
per-message settings UI, and **no** WebSocket. Feedback correlation is
stateless — the app posts the fields back rather than the service holding
sessions in memory. Those are all things a "serious" app might add; none are
needed at the current scale, and the parent repo's priority order
(simplicity first) says not to build them until they are.
