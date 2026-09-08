# rag-study

A modular, offline-capable Retrieval-Augmented Generation package. Documents go
in; ranked, cited context comes out. It runs with no internet connection: model
weights are local files, and nothing is downloaded while the system is running.

The package stops at assembled context. Answering is a separate sibling
package, `rag.generation`. An HTTP API over the query side lives in `service/`,
and a standalone Chainlit chat app that talks only to that API lives in `app/`
— both outside the package. See [docs/api.md](docs/api.md).

## What it solves

Turning a corpus of PDFs into context a language model can answer from,
reliably and reproducibly, without trusting the documents. Retrieved text is
treated as untrusted data throughout: it cannot change a ranking, forge a
citation, or become an instruction. Every result records the prompt version
that produced it.

## Architecture

Two independent pipelines share only the domain contracts and the storage
interface. Neither imports the other; either could be deleted and the other
would still work. `tests/test_architecture.py` parses the source and fails the
build if that stops being true.

```text
        config      domain      model_assets      leaves: no internal dependencies
                       ▲
                    storage                       SQLAlchemy + MariaDB
                       ▲
           ┌───────────┴───────────┐
       ingestion                retrieval          never each other
                                    ╎
                                generation         domain and config only
```

**Ingestion:** `source → extract → clean → chunk → embed → store`

**Retrieval:** `query → embed → rank → rerank → assemble context`

Design priorities, in order when contested: simplicity, modularity,
independence, maintainability, correctness, testability, extensibility,
performance. Every abstraction is a real replacement point; nothing is built
for a need that does not yet exist.

## Main technologies

MariaDB 11.7+ (native `VECTOR` column and cosine vector index), SQLAlchemy,
PyMySQL, `sentence-transformers` for local embedding and cross-encoder models,
`pymupdf` / `pymupdf4llm` for layout-aware PDF extraction, `langchain-text-splitters`
and `ftfy` behind the project's own interfaces, and `httpx` for the answering
service. The optional HTTP API is FastAPI + `sse-starlette`; the optional chat
app is Chainlit talking to it over `httpx`.

## Quick start

Five steps from a clean checkout to a working query. Each is checkable.

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate rag-study-py3.12
python -c "import rag; print(rag.__version__)"    # 0.1.0
```

### 2. Start MariaDB and create the databases

Requires **MariaDB 11.7 or later**. Verified against 12.3.3.

```bash
brew services start mariadb          # or your platform's equivalent
mariadb -e "SELECT VERSION();"       # must be >= 11.7
```

```sql
CREATE DATABASE rag_study      CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE DATABASE rag_study_test CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER 'rag_app'@'localhost' IDENTIFIED BY 'choose-a-strong-password';
GRANT SELECT, INSERT, UPDATE, DELETE ON rag_study.* TO 'rag_app'@'localhost';
GRANT ALL PRIVILEGES ON rag_study_test.* TO 'rag_app'@'localhost';
FLUSH PRIVILEGES;
```

The application user holds no DDL rights by design. Schema creation is a
separate, privileged step. See
[docs/database-cheatsheet.md](docs/database-cheatsheet.md).

### 3. Configure

```bash
cp .env.example .env
```

Set `RAG_DB_PASSWORD` to the password chosen above. The `RAG_LLM_*` variables
are optional and configure the answering model; retrieval works without them.
Never commit `.env` — it is gitignored.

### 4. Fetch the model weights

The only step that needs the internet, and it happens now, not at run time.
Both models are fetched as `safetensors`.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_dir="models/embeddings/all-MiniLM-L6-v2",
    allow_patterns=["*.json", "vocab.txt", "model.safetensors", "1_Pooling/*"],
)
snapshot_download(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    local_dir="models/rerankers/ms-marco-MiniLM-L-6-v2",
    allow_patterns=["*.json", "vocab.txt", "model.safetensors"],
)
PY
```

Check it — this must print `384`, matching the database column width:

```bash
python -c "
from pathlib import Path
from rag.ingestion.embedding.local import LocalSentenceTransformerEmbedder
print(LocalSentenceTransformerEmbedder(Path('models/embeddings/all-MiniLM-L6-v2')).dimension)"
```

### 5. Create the schema

```bash
RAG_ADMIN_DB_URL='mysql+pymysql://root@localhost/rag_study?unix_socket=/tmp/mysql.sock' \
    python scripts/create_schema.py
```

Expected: `Schema present. Tables: chunks, documents, embeddings, pages`

## Usage

### Ingest documents

```bash
python scripts/ingest.py path/to/pdfs
```

```text
Ingested 4 documents (487 chunks), skipped 1 duplicates, 0 failed.
```

One unreadable file cannot stop a run; failures are reported per document.
Content already stored is skipped, so re-running is safe. A document is written
in a single transaction. Discovery is non-recursive and lowercase `.pdf` only.

### Retrieve context

```bash
python scripts/query.py "How does DPO avoid training a reward model?"
python scripts/query.py "..." --top-k 3 --document-id <id> --show-prompt
```

Both scores are shown — what the vector search thought, and what the reranker
thought. **The retrieval pipeline calls no language model.** It stops at
assembled context.

### Get an answer

```bash
just ask "How does DPO avoid training a reward model?"    # in-process, no browser
```

Or over the HTTP API, with the Chainlit chat app in front of it:

```bash
pip install -e ".[api,ui]" && pip install -e app/    # the API and the chat app
just serve                                           # http://localhost:8080
just ui                                              # http://localhost:8000
```

`app/` is its own project with no dependency on `rag` — it can be lifted into
another repository and pointed at a deployed service. With no service running
it starts in demo mode. Full contract, endpoints, and auth in
[docs/api.md](docs/api.md).

The model is a service reached over HTTP and chosen by configuration:

```bash
RAG_LLM_PROVIDER=ollama                     # or vllm, or openai
RAG_LLM_BASE_URL=http://localhost:11434     # vLLM and OpenAI want the /v1 root
RAG_LLM_MODEL=deepseek-r1:14b
RAG_LLM_API_KEY=                            # only for a service that needs one
```

`openai` covers anything speaking the OpenAI chat-completions API. The model is
never downloaded — it must already exist on that service. See
[docs/generation.md](docs/generation.md).

## Configuration

Every recognised variable is in [`.env.example`](.env.example), with prose. The
ones a first deployment sets:

| Variable | Default | Purpose |
|---|---|---|
| `RAG_DB_HOST` / `RAG_DB_USER` / `RAG_DB_PASSWORD` / `RAG_DB_NAME` | — | database connection (required) |
| `RAG_EMBEDDING_MODEL_PATH` / `RAG_EMBEDDING_MODEL_NAME` | — | the local embedding model (required) |
| `RAG_RERANKER_MODEL_PATH` | unset | cross-encoder directory; unset keeps the vector ranking |
| `RAG_TOP_K` / `RAG_MAX_TOP_K` | 5 / 100 | passages returned by default, and the ceiling |
| `RAG_PROMPT_NAME` / `RAG_PROMPT_VERSION` | `retrieval_context` / `v1` | which prompt template renders context |
| `RAG_SEMANTIC_CHUNKING` | `false` | split oversized sections at topic boundaries (costs an extra embedding pass) |
| `RAG_LLM_*` | see `.env.example` | the optional answering service |
| `RAG_API_KEY` / `RAG_API_HOST` / `RAG_API_PORT` | unset / `127.0.0.1` / `8080` | the HTTP API; an unset key means it runs open ([docs/api.md](docs/api.md)) |

## Offline operation

No component reaches the network at run time. Model weights are local files,
fetched once at setup (step 4). Model loading passes `local_files_only=True`
and refuses remote code. The guarantee is verified by running the suite with
the network disabled:

```bash
HF_HUB_OFFLINE=1 pytest
```

## Model requirements

An embedding model (required) and a cross-encoder re-ranker (optional), both
`sentence-transformers` models held in local directories. Defaults are
`all-MiniLM-L6-v2` (384 dimensions) and `cross-encoder/ms-marco-MiniLM-L-6-v2`.
See [docs/model-deployment-cheatsheet.md](docs/model-deployment-cheatsheet.md).

## Database requirements

MariaDB 11.7 or later, for the native `VECTOR` type and vector index.
Similarity runs in the database via `VEC_DISTANCE_COSINE`. The application user
is least-privilege and holds no DDL rights; schema creation uses a separate
administrative account. See
[docs/database-cheatsheet.md](docs/database-cheatsheet.md).

## Testing

```bash
just check-all                 # ruff, bandit, mypy, pytest — the full gate
pytest -m "not integration"    # no database or model weights required
HF_HUB_OFFLINE=1 pytest        # proves nothing reaches the network
```

Integration tests need MariaDB running and the model weights present; they skip
rather than fail when either is missing.

## Security

Retrieved content and metadata are untrusted throughout. The instruction/data
boundary in the prompt is structural, not a keyword filter. Provenance and
scores are pipeline-set on frozen dataclasses and cannot be written from
document content. SQL is built from bound parameters only, with an allow-list
for filterable fields. The application database user holds DML rights only.
Model loading is local-only with remote code refused. The full model — what is
implemented, what is assumed, and what is not — is in
[docs/security.md](docs/security.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | boundaries, contracts, dependency rules, weaknesses |
| [docs/ingestion.md](docs/ingestion.md) | the ingestion pipeline, stage by stage |
| [docs/retrieval.md](docs/retrieval.md) | the retrieval pipeline, and prompt versioning |
| [docs/generation.md](docs/generation.md) | `rag.generation`, the generation contract, feedback |
| [docs/api.md](docs/api.md) | the HTTP API (`service/`) and the standalone chat app (`app/`) |
| [docs/security.md](docs/security.md) | the security model and its mitigations |
| [docs/database-cheatsheet.md](docs/database-cheatsheet.md) | operating MariaDB for this project |
| [docs/model-deployment-cheatsheet.md](docs/model-deployment-cheatsheet.md) | provisioning local model assets |
| [docs/codebase-cheatsheet.md](docs/codebase-cheatsheet.md) | repository map and extension points |
| [docs/future-work.md](docs/future-work.md) | what is deliberately not built yet |
| [CONTRIBUTING.md](CONTRIBUTING.md) | branch strategy and QA requirements |
