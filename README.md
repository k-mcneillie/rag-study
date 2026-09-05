# rag-study

A simple, modular, offline-capable Retrieval-Augmented Generation package.
Documents go in; ranked, cited context comes out — and, through a chat
interface that sits outside the package, an answer with its sources. It runs
with **no internet connection**: model weights are local files, and nothing is
downloaded while the system is running.

---

# Quick start

Five steps from a clean checkout to a working query. Each one is checkable, so
if something goes wrong you will know which step it was.

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate rag-study-py3.12
```

Check it worked:

```bash
python -c "import rag; print(rag.__version__)"    # 0.1.0
```

## 2. Start MariaDB and create the databases

Requires **MariaDB 11.7 or later** — earlier versions have no `VECTOR` type.
Verified against 12.3.3.

```bash
brew services start mariadb          # macOS; use your platform's equivalent
mariadb -e "SELECT VERSION();"       # must be >= 11.7
```

Create the databases and a **least-privilege application user**. The
application needs only to read and write rows; it is deliberately not allowed
to change the schema, so a bug or an injected statement cannot alter it:

```sql
CREATE DATABASE rag_study      CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE DATABASE rag_study_test CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER 'rag_app'@'localhost' IDENTIFIED BY 'choose-a-strong-password';
GRANT SELECT, INSERT, UPDATE, DELETE ON rag_study.* TO 'rag_app'@'localhost';

-- The test database is disposable; the suite creates and drops its own tables.
GRANT ALL PRIVILEGES ON rag_study_test.* TO 'rag_app'@'localhost';
FLUSH PRIVILEGES;
```

## 3. Configure

```bash
cp .env.example .env
```

Edit `.env` and set `RAG_DB_PASSWORD` to the password you just chose. **Never
commit `.env`** — it is gitignored, and it should stay that way.

The `RAG_LLM_*` variables are optional: they configure the model that answers,
and retrieval works without any of them. Set them when you get to
[answering](#get-an-answer-not-just-context).

## 4. Fetch the model weights

This is the **only step that needs the internet**, and it happens now, not at
run time. Both models are downloaded as `safetensors`, never as pickle files.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

# Embedding model: turns text into vectors. Required.
snapshot_download(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_dir="models/embeddings/all-MiniLM-L6-v2",
    allow_patterns=["*.json", "vocab.txt", "model.safetensors", "1_Pooling/*"],
)

# Reranker: reorders results. Optional but recommended.
snapshot_download(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    local_dir="models/rerankers/ms-marco-MiniLM-L-6-v2",
    allow_patterns=["*.json", "vocab.txt", "model.safetensors"],
)
PY
```

Check it worked — this must print `384`, matching the database column width:

```bash
python -c "
from pathlib import Path
from rag.ingestion.embedding.local import LocalSentenceTransformerEmbedder
print(LocalSentenceTransformerEmbedder(Path('models/embeddings/all-MiniLM-L6-v2')).dimension)"
```

## 5. Create the schema

The application user cannot do this, by design, so use an administrative
account:

```bash
RAG_ADMIN_DB_URL='mysql+pymysql://root@localhost/rag_study?unix_socket=/tmp/mysql.sock' \
    python scripts/create_schema.py
```

Expected output: `Schema created. Tables: chunks, documents, embeddings, pages`

---

# Using it

## Ingest documents

```bash
python scripts/ingest.py path/to/your/pdfs
```

```text
Ingested 4 documents (487 chunks), skipped 1 duplicates, 0 failed.
```

One unreadable file cannot stop a run — failures are reported per document.
Documents whose content has already been stored are skipped, so re-running is
safe. A document is written in a single transaction: either all of it is
stored, or none of it is.

## Ask a question

```bash
python scripts/query.py "How does DPO avoid training a reward model?"
python scripts/query.py "..." --top-k 3 --show-prompt --document-id <id>
```

```text
prompt: retrieval_context v1   reranker: cross-encoder

[1] score=4.094 (initial 0.629)  page 2  2 Method > 2.1 Objective
    <the opening of the matching passage, from your own documents>
```

Both scores are shown: what the vector search thought, and what the reranker
thought. `--show-prompt` prints the full assembled prompt.

**The retrieval pipeline calls no language model.** It stops at assembled
context. Answering is a separate package on the far side of that contract.

## Get an answer, not just context

```bash
pip install -e ".[ui]"     # the chat interface; not needed to retrieve
just ui                    # http://localhost:8000
just ask "How does DPO avoid training a reward model?"    # no browser
```

The interface streams the answer, shows the passages it cited, keeps the
model's reasoning in a collapsed step, and offers thumbs up/down under each
answer. Ratings are appended to `results/feedback.jsonl` with the passages the
answer was given.

The model is a service, reached over HTTP and chosen by configuration:

```bash
RAG_LLM_PROVIDER=ollama                     # or vllm, or openai
RAG_LLM_BASE_URL=http://localhost:11434     # vLLM and OpenAI want the /v1 root
RAG_LLM_MODEL=deepseek-r1:14b
RAG_LLM_API_KEY=                            # only for a service that needs one
```

`openai` covers anything speaking the OpenAI chat completions API — LM Studio,
llama.cpp, OpenAI, Together, Groq, OpenRouter. Pointing at a different service
is an environment change, not a code change; adding one it cannot yet speak to
is a `BaseChatModel` and one line in `rag.generation.providers`.

The model is never downloaded — it must already exist on that service — and no
answer leaves the machine when the service is local. See
[docs/interface.md](docs/interface.md).

## Run the tests

```bash
just check-all                 # lint, format, types, tests — the full gate
pytest -m "not integration"    # no database or model weights needed
```

---

# How it works

## Two independent pipelines

Ingestion and retrieval **never import each other**. They share only data
contracts and storage:

```text
     config      domain      model_assets     (leaves: no internal deps)
                    ▲
                 storage                      (SQLAlchemy + MariaDB)
                    ▲
        ┌───────────┴───────────┐
    ingestion                retrieval
```

**Ingestion:** `PDF → extract → clean → chunk → embed → store`

**Retrieval:** `query → embed → rank → rerank → assemble context`

This is enforced, not merely intended: `tests/test_architecture.py` parses the
source and fails if a pipeline imports the other, if the domain gains a
framework dependency, or if anything outside `storage` imports SQLAlchemy.

## What the pipeline cleans, and why

Extraction is layout-aware, so two-column papers, journal templates and tables
come back in reading order. Beyond that, each of these was measured in a real
corpus before any code was written for it:

| Contaminant | Why it matters | Handling |
|---|---|---|
| Running heads, page numbers | Repeat on every page, crowding results | Three rules at page edges |
| Chart and axis label text | Fragments that match numbers but answer nothing | Figure text blocks discarded |
| Reference lists | Match on wording, answer nothing | Dropped (`drop_references=False` keeps them) |
| Duplicate documents | Fill top-k with the same passage | Skipped by content hash |
| Fragments | Bare headings displace real passages | Minimum chunk length |

Repeated sentences *inside* a page are deliberately kept: scientific writing
restates things legitimately, and removing them would destroy content.

## Chunking strategies

| Chunker | Cuts at | Use when |
|---|---|---|
| `MarkdownHeaderChunker` | Headings | You want section provenance (default) |
| `RecursiveChunker` | Character budget | You only need size control |
| `SemanticChunker` | Where the subject changes | Long discursive prose |

Semantic chunking is opt-in because it embeds every sentence:

```python
from rag.ingestion import ChunkingPipeline, SemanticChunker
from rag.ingestion.embedding.local import LocalSentenceTransformerEmbedder

embedder = LocalSentenceTransformerEmbedder(settings.embedding_model_path)
chunker = ChunkingPipeline(semantic_chunker=SemanticChunker(embedder))
```

## Replacing any component

Every stage is constructor-injected behind an interface. Write a class, pass
it in; nothing else changes:

| To replace | Subclass | Pass to |
|---|---|---|
| Document format | `BaseExtractor` | `IngestionOrchestrator(extractor=...)` |
| Chunking | `BaseChunker` | `IngestionOrchestrator(chunker=...)` |
| Embedding model | `BaseEmbedder` | `IngestionOrchestrator(embedder=...)` |
| Ranking | `BaseRanker` | `RetrievalOrchestrator(ranker=...)` |
| Reranking | `BaseReranker` | `RetrievalOrchestrator(reranker=...)` |
| Prompt assembly | `BasePromptAugmenter` | `RetrievalOrchestrator(prompt_augmenter=...)` |
| Database | `Repository` protocol | either orchestrator |
| Answering model | `BaseChatModel` | one line in `rag.generation.providers` |

## Prompt versioning and injection

Templates live in `src/rag/retrieval/prompts/` as `<name>_<version>.txt`.
Every result records the prompt that produced it, and the version is
configuration (`RAG_PROMPT_VERSION`), so a past result can be reproduced by
restoring the configuration that produced it.

Retrieved text is placed inside marked source material that the template
describes as untrusted data to be quoted, never obeyed. This is structural,
not a keyword filter — a filter can be rephrased around, a boundary cannot.
Marker forgeries in retrieved text *or in the query* are neutralised, and
citations come from the pipeline's own provenance, so a document cannot claim
to be a different source.

---

# Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Required environment variable RAG_DB_... is not set` | `.env` missing or incomplete — see step 3 |
| `No model at models/embeddings/...` | Weights not fetched — see step 4. It will never download them for you |
| `CREATE command denied to user 'rag_app'` | Working as intended: run step 5 as an admin |
| `error in your SQL syntax ... VECTOR` | MariaDB older than 11.7 |
| Integration tests skipped | No database or weights present; the unit suite still runs |
| Ingestion reports `0 chunks` | Scanned PDF with no text layer — OCR needs Tesseract installed |
| `Cannot reach the model service at ...` | The service is not running, or `RAG_LLM_BASE_URL` is wrong |
| `Model ... is not available at ...` | Not provisioned on that service. It is never pulled for you (`ollama pull <model>`) |
| `... rejected the credential (HTTP 401)` | `RAG_LLM_API_KEY` is missing or wrong for a service that requires one |
| `Unknown model provider '...'` | `RAG_LLM_PROVIDER` must be `ollama`, `vllm` or `openai` |
| `ModuleNotFoundError: chainlit` | The interface is an optional extra: `pip install -e ".[ui]"` |

---

# Further reading

- [docs/system-overview.md](docs/system-overview.md) — how every part fits together
- [docs/interface.md](docs/interface.md) — the answering layer, the chat UI, and feedback
- [docs/architecture.md](docs/architecture.md) — design decisions, threat model, open questions
- [docs/future-work.md](docs/future-work.md) — what is deliberately not built yet
- [CONTRIBUTING.md](CONTRIBUTING.md) — branch strategy and QA requirements
