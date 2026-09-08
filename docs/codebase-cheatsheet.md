# Codebase cheat sheet

A map of the repository and where to make each kind of change.

## Repository layout

```text
src/rag/
├── __init__.py            package docstring; __version__
├── config.py              Settings dataclasses, loaded from RAG_* environment variables
├── model_assets.py        the one audited way to load a local model (both pipelines)
├── feedback.py            human ratings appended to a JSON Lines file
├── assembly.py            composition root for the query side (retrieval + generation)
│
├── domain/
│   └── models.py          the eleven frozen dataclasses every component exchanges
│
├── storage/               the only package that imports SQLAlchemy
│   ├── repository.py       Repository protocol + ALLOWED_SEARCH_FILTERS
│   ├── orm.py              DocumentRow, PageRow, ChunkRow, EmbeddingRow, EMBEDDING_DIMENSION
│   ├── vector.py           Vector SQLAlchemy type for MariaDB VECTOR(n)
│   ├── engine.py           build_url, build_engine, build_session_factory
│   └── mariadb_repository.py   MariaDBRepository
│
├── ingestion/             documents → stored, embedded chunks; never imports retrieval
│   ├── interfaces.py       BaseExtractor, BaseChunker, BaseEmbedder, ExtractionError
│   ├── orchestrator.py     IngestionOrchestrator, IngestionResult
│   ├── extraction/
│   │   ├── pdf.py           PDFExtractor
│   │   └── cleaning.py      ftfy repair + page-furniture / figure-text removal
│   ├── chunking/
│   │   ├── pipeline.py      ChunkingPipeline (the default BaseChunker)
│   │   ├── splitters.py     MarkdownHeaderChunker, RecursiveChunker
│   │   └── semantic.py      SemanticChunker + TextEmbedder protocol
│   └── embedding/
│       └── local.py         LocalSentenceTransformerEmbedder
│
├── retrieval/             a query → context with provenance; never imports ingestion
│   ├── interfaces.py       BaseQueryEmbedder, BaseRanker, BaseReranker, BasePromptAugmenter
│   ├── orchestrator.py     RetrievalOrchestrator
│   ├── embedding/local.py  LocalSentenceTransformerQueryEmbedder
│   ├── ranking/cosine.py   CosineSimilarityRanker
│   ├── reranking/
│   │   ├── cross_encoder.py CrossEncoderReranker
│   │   └── passthrough.py   PassthroughReranker
│   ├── prompting/augmenter.py   TemplatePromptAugmenter, neutralise_markers
│   └── prompts/
│       └── retrieval_context_v1.txt
│
└── generation/            context → an answer, from a swappable service; domain + config only
    ├── interfaces.py       BaseChatModel, GenerationError
    ├── providers.py        CHAT_MODELS registry + chat_model_for
    ├── transport.py        shared HTTP plumbing
    ├── ollama.py           OllamaChatModel
    └── openai_compatible.py    OpenAICompatibleChatModel

service/                   the HTTP API over the query side, an entry point like scripts/
├── app.py                 create_app(); lifespan builds the pipeline via rag.assembly
├── routes.py              GET /health, POST /answer (SSE), /feedback, /documents
├── schemas.py             the pydantic wire contract
├── streaming.py           AnswerDelta generator → server-sent events
├── security.py            optional X-API-Key check (RAG_API_KEY)
└── ingest_wiring.py       ingestion pipeline for /documents (rag.assembly can't import ingestion)

app/                       standalone Chainlit client of the API; no rag import; its own pyproject
└── src/rag_chat/          main.py (Chainlit), client.py (httpx), models.py, demo.py, config.py

scripts/                   ingest.py, query.py, answer.py, create_schema.py
tests/                     unit, integration (marked), architecture; tests/service/ covers the API
```

### Package roles

| Package | Role | Import rule |
|---|---|---|
| `domain` | the eleven data contracts; no third-party imports | imports nothing internal |
| `config` | `Settings` from the environment | imports nothing internal |
| `model_assets` | safe local model loading | imports nothing internal |
| `storage` | persistence; the only SQLAlchemy importer | `domain`, `config` |
| `feedback` | rating log | `domain` |
| `ingestion` | the ingestion pipeline | `domain`, `storage`, `model_assets`, `config` |
| `retrieval` | the retrieval pipeline | `domain`, `storage`, `model_assets`, `config` |
| `generation` | answering from a `PromptContext` | `domain`, `config` |
| `assembly` | query-side composition root | `domain`, `config`, `storage`, `retrieval`, `generation`, `model_assets` |

This table is `ALLOWED_IMPORTS` in `tests/test_architecture.py`, which parses
the source and fails the build on a violation. When two components need the
same thing, it becomes a leaf module rather than an import between them.

## Entry points

| Command | Does |
|---|---|
| `python scripts/ingest.py <dir> [--reingest]` | ingest a directory of PDFs |
| `python scripts/query.py "<q>" [--top-k N] [--document-id ID] [--show-prompt]` | retrieve and print context; no model called |
| `python scripts/answer.py "<q>" [--top-k N] [--document-id ID] [--show-prompt] [--show-reasoning]` | retrieve and answer from a model service |
| `python scripts/create_schema.py` | create the schema / report drift (needs `RAG_ADMIN_DB_URL`) |
| `just serve` | the HTTP API at `http://localhost:8080` (`uvicorn service.app:create_app --factory`) |
| `just ui` | the chat interface at `http://localhost:8000` (`chainlit run app/src/rag_chat/main.py`; needs `pip install -e app/`) |
| `just ask "<q>"` | `scripts/answer.py` without a browser |

`scripts/query.py`, `scripts/answer.py`, and `service/app.py` compose the
query pipeline through `rag.assembly`. `scripts/ingest.py` and
`service/ingest_wiring.py` each have their own ingestion wiring, because
ingestion is not part of the query-side composition root. `app/` composes
nothing — it is an HTTP client of `service/`.

## Running the checks

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate rag-study-py3.12

just check-all                 # ruff lint + format-check, bandit, mypy, pytest
pytest -m "not integration"    # no database or model weights required
HF_HUB_OFFLINE=1 pytest        # proves nothing reaches the network
```

Individual gates: `just lint`, `just format-check`, `just type-check`
(`mypy src/ service/ app/src/ app/tests/ scripts/`), `just security` (bandit),
`just test`. Integration tests need MariaDB running and the model weights
present; they skip rather than fail otherwise.

## Where to make a change

Each extension is one class implementing one interface, injected at
composition. Nothing downstream changes.

```text
New document format
    → subclass BaseExtractor (rag/ingestion/interfaces.py)
    → inject into IngestionOrchestrator in scripts/ingest.py:build_orchestrator
      and service/ingest_wiring.py:build_ingestion_orchestrator
    → no chunking, embedding, storage, retrieval, or generation change

New API endpoint
    → add a route in service/routes.py, a shape in service/schemas.py
    → depend on _KEYED unless it is a health/liveness check
    → the app in app/ is a separate client; add the call in app/src/rag_chat/client.py

New chunking strategy
    → subclass BaseChunker
    → inject as the chunker, or compose into ChunkingPipeline
    → no other stage changes

New embedding model
    → subclass BaseEmbedder (ingestion) and/or BaseQueryEmbedder (retrieval)
    → the same model must serve both; loading goes through rag.model_assets
    → a different output width is a schema change (see model-deployment-cheatsheet.md)

New ranking strategy (e.g. hybrid keyword + vector)
    → subclass BaseRanker (rag/retrieval/interfaces.py)
    → inject into RetrievalOrchestrator via rag.assembly
    → the Repository protocol is unchanged

New re-ranker
    → subclass BaseReranker
    → inject via rag.assembly:build_reranker

New prompt augmenter
    → subclass BasePromptAugmenter
    → inject via rag.assembly

New prompt version
    → add src/rag/retrieval/prompts/<name>_<version>.txt (package data in pyproject.toml)
    → set RAG_PROMPT_NAME / RAG_PROMPT_VERSION
    → old versions stay as files; nothing selects them unless configured

New answering service
    → subclass BaseChatModel (rag/generation/interfaces.py): model_name, stream()
    → add one line to CHAT_MODELS in rag/generation/providers.py
    → shared HTTP plumbing is already in rag/generation/transport.py

New storage backend
    → implement the Repository protocol (rag/storage/repository.py)
    → only rag/storage/ may import a database library
```

Storage behaviour — transactions, bulk insert, the similarity query,
replace-on-reingest — lives in `rag/storage/mariadb_repository.py`. The schema
lives in `rag/storage/orm.py`.

## Tests

| Path | Covers |
|---|---|
| `tests/test_architecture.py` | the import rules, by parsing the source |
| `tests/test_config.py` | environment parsing; credential non-disclosure |
| `tests/domain/` | frozen models; provenance not taken from metadata |
| `tests/storage/` | the `Vector` type, engine URL, the protocol against an in-memory fake |
| `tests/ingestion/` | extraction, cleaning, chunking, semantic chunking, the orchestrator |
| `tests/retrieval/` | ranking, reranking, prompt assembly and its boundary |
| `tests/generation/` | the two clients, provider selection |
| `tests/service/` | the API routes, against fakes and a `TestClient`; no DB, no weights |
| `tests/integration/` | real MariaDB, real weights (marked `integration`) |
| `app/tests/` | the chat app's client, demo stream, and Chainlit-loader smoke test |

`pyproject.toml` sets `pythonpath = ["src", ".", "app/src"]`, collects
`tests` and `app/tests`, and defines the `integration` marker. CI runs
`pytest -m "not integration"` and, separately,
`pytest tests/test_architecture.py --no-cov`.

## Configuration files

| File | Holds |
|---|---|
| `.env.example` | every recognised `RAG_*` variable, with prose (including `RAG_API_*`) |
| `pyproject.toml` | the package; deps (`api`, `ui` extras), ruff / mypy / bandit / pytest / coverage |
| `app/pyproject.toml` | the chat app as its own project — deps and mirrored ruff / mypy config |
| `environment.yml` | the conda environment; installs `-e .[dev,api,ui]` and `-e ./app` |
| `justfile` | the check and run recipes (`serve`, `ui`, …) |
| `.github/workflows/ci.yml` | the CI job |
