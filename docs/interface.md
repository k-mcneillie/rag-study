# Generation and the chat interface

The package retrieves; it does not generate. This document covers the two
pieces that stand on the far side of that boundary: `rag.generation`, which
answers from an assembled `PromptContext`, and the Chainlit application in
[`app/`](../app), which puts a chat window in front of it.

Both are optional. Retrieval works, and is tested, with no model service
configured and the interface not installed.

---

## Where the boundary is

```
                    ┌─────────────────── src/rag ───────────────────┐
  question ────────>│ retrieval ──> PromptContext ──> generation    │──> AnswerDelta
                    │  (unchanged)     (domain)       (sibling)     │      stream
                    └───────────────────────────────────────────────┘
                                                             │
                              app/main.py  ─────────────────┘   scripts/answer.py
                              (Chainlit, async)                 (terminal)
                                     │
                                     v
                          rag.feedback ──> results/feedback.jsonl
```

Three properties hold, and two of them are enforced by
`tests/test_architecture.py` rather than by review:

- **`generation` imports `domain` and `config`, and nothing else.** It cannot
  see `retrieval`, `ingestion` or `storage`. It receives a `PromptContext` and
  has no way to reconstruct one.
- **The rendered prompt is sent verbatim.** `TemplatePromptAugmenter` builds a
  prompt with a structural boundary between application instructions and
  untrusted document content. A client that re-templated or split that text
  across message roles would be moving a security boundary it does not own, so
  none of them do. The prompt goes out as a single user message.
- **The interface lives outside the package.** `app/` assembles its own
  pipeline, exactly as `scripts/` does. Deleting it removes the interface and
  nothing else.

The interface is asynchronous and the package is not. The bridge is
`asyncio.to_thread`, used in `app/main.py` alone: retrieval runs in a worker
thread, and each streamed fragment is pulled from the synchronous generator the
same way. No signature inside `src/rag` is coloured by it.

---

## Running it

```bash
pip install -e ".[ui]"     # chainlit; not needed to retrieve
just ui                    # chainlit run app/main.py  →  http://localhost:8000
just ask "your question"   # the same thing without a browser
```

The database must be reachable and the model service must already be running
with the model available. Neither is started, and nothing is downloaded: a
missing model is an error naming what is missing.

`scripts/answer.py` takes `--top-k`, `--document-id`, `--show-prompt` and
`--show-reasoning`, and is the quickest way to see what the model was actually
sent.

---

## Choosing a model service

Set `RAG_LLM_PROVIDER`. It names a protocol, not a company.

| Provider | Speaks to | Notes |
| --- | --- | --- |
| `ollama` | Ollama's native `/api/chat` | Default. Preferred for a reasoning model: reasoning comes back in its own field rather than in `<think>` tags. Accepts `num_ctx`. |
| `vllm` | vLLM's OpenAI-compatible server | Point `RAG_LLM_BASE_URL` at the server's `/v1` root. Start vLLM with `--reasoning-parser` to get reasoning separated. |
| `openai` | Anything else speaking the OpenAI chat completions API | LM Studio, llama.cpp's server, OpenAI, Together, Groq, OpenRouter. |

The remaining variables are in `.env.example`: `RAG_LLM_BASE_URL`,
`RAG_LLM_MODEL`, `RAG_LLM_API_KEY`, `RAG_LLM_TIMEOUT_SECONDS`,
`RAG_LLM_TEMPERATURE`, `RAG_LLM_NUM_CTX`, `RAG_LLM_THINKING`.

`RAG_LLM_NUM_CTX` deserves attention. A prompt carrying several retrieved
passages is easily long enough to be truncated by a default context window, and
truncation is silent: the passages simply never reach the model. It is sent
only by the Ollama client, because the OpenAI protocol has no equivalent and a
service that sizes its own context is trusted to do so.

### Adding another service

Write a `BaseChatModel` — two members, `model_name` and `stream()` — and add
one line to `CHAT_MODELS` in `rag/generation/providers.py`. The HTTP plumbing
(authentication, error mapping, line streaming) is already shared in
`rag/generation/transport.py`, so a new client is usually the request body and
the response shape and nothing else.

### What is verified, and what is not

The Ollama client is exercised end to end against a real server in
`tests/integration/test_ollama_chat.py`.

**vLLM is supported by protocol and has not been run against a live vLLM
server.** It is reached through the same client as every other
OpenAI-compatible service, and the unit tests pin the parts vLLM's behaviour
depends on: the endpoint path, the `/v1` base-URL convention, server-sent event
framing, bearer authentication, and the `reasoning_content` field vLLM emits
with a reasoning parser enabled. If it misbehaves, the fault is in
`openai_compatible.py` alone.

---

## Reasoning

Reasoning models return their deliberation as well as their answer. The
interface streams it into a collapsed step above the answer, so it is
inspectable without being in the way, and `Answer` keeps the two apart.

How cleanly that separation happens depends on the service. Ollama returns
reasoning in a `thinking` field, and vLLM with a reasoning parser returns it in
`reasoning_content`; both are exact. A service that instead inlines `<think>`
tags in the content stream is handled by splitting on the tags, including when
a tag is itself split across two fragments — but that is pattern-matching on
model output, and it is the reason the native endpoint is preferred for a
reasoning model. Set `RAG_LLM_THINKING=false` to ask for none of it.

---

## Citations

Passages are numbered from one, matching the `[n]` markers the prompt template
asks the model to cite, so a `[2]` in an answer is the second source element
beside it. Each element shows the page, the section path, the reranker's score
and the document id.

Passage text is rendered inside a fenced code block. It is untrusted document
content and the interface renders Markdown: a document containing image or link
syntax should be displayed, not obeyed.

---

## Feedback

Each answer carries thumbs up and thumbs down buttons. A click appends one JSON
object to `RAG_FEEDBACK_PATH` (default `results/feedback.jsonl`):

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

Two decisions about that file are deliberate:

- **Citations record provenance, never passage text.** The provenance locates a
  passage exactly, and the document pool is not something to copy into a second
  place on disk.
- **The log is gitignored.** It still holds question and answer text drawn from
  the corpus, so it is treated the way the corpus itself is treated.

Chainlit's own thumb icons are not used. They require a registered data layer,
which means Chainlit's schema created with DDL rights the application user
deliberately does not have. Explicit action buttons need none of that and are
not tied to an interface that changes between Chainlit versions.

The prompt version and model name are recorded with every rating, so ratings
collected before and after a prompt or model change stay distinguishable —
which is what makes them usable later as the seed for the evaluation harness in
[future-work.md](future-work.md).
