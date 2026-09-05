# rag-study

A simple, modular, offline-capable Retrieval-Augmented Generation package,
built as a foundation to extend rather than a finished platform.

## Purpose

The package turns source documents into retrievable, embedded chunks, and
turns a query into ranked context with full provenance. It is designed to run
with **no internet connection**: model weights are local assets, and no
component downloads anything at runtime.

## Architectural philosophy

> Simplicity. Modularity. Independence. Maintainability.

Abstractions exist only where they buy **replaceability** (swap an embedding
model, a chunking strategy, a ranker) or **separation of responsibility**
(processing versus persistence). When a decision is contested, it is settled in
this order: simplicity, modularity, independence, maintainability, correctness,
testability, extensibility, and only then performance.

## The two-pipeline architecture

The central rule is that **ingestion and retrieval know nothing about each
other**. They are separate subpackages that never import one another, sharing
only data contracts and storage.

```
                 domain          (data contracts, no dependencies)
                    ▲
                    │
                 storage         (SQLAlchemy + MariaDB)
                    ▲
        ┌───────────┴───────────┐
        │                       │
    ingestion                retrieval
```

**Ingestion:** `Document → Extraction → Chunking → Embedding → Storage`

**Retrieval:** `Query → Storage → Ranking → Re-ranking → Prompt augmentation →
Top-k context + metadata`

Deleting either pipeline directory leaves the other working. `storage` is
infrastructure beneath both, not part of either.

The full design — domain contracts, interface responsibilities, database
schema, threat model, and open questions — is in
[docs/architecture.md](docs/architecture.md).

## Package structure

```
src/rag/
├── domain/models.py            # Document, Page, Chunk, EmbeddedChunk,
│                               # RankedChunk, RerankedChunk, PromptContext
├── config.py                   # environment-driven settings
├── storage/
│   ├── vector.py               # MariaDB VECTOR column type
│   ├── orm.py                  # SQLAlchemy models (the only ORM mapping)
│   ├── engine.py               # engine / session construction
│   ├── repository.py           # the storage contract both pipelines use
│   └── mariadb_repository.py   # MariaDB implementation
└── ingestion/
    ├── interfaces.py           # BaseExtractor, BaseChunker, BaseEmbedder
    ├── extraction/
    │   ├── pdf.py              # layout-aware PDF extraction
    │   └── cleaning.py         # encoding repair, page-furniture removal
    ├── chunking/
    │   ├── splitters.py        # heading-aware and size-aware chunkers
    │   └── pipeline.py         # the default strategy + contamination filters
    ├── embedding/local.py      # offline sentence-transformers embedder
    └── orchestrator.py         # workflow only
```

`retrieval/` arrives in Phase 4.

## Running ingestion

```python
from pathlib import Path

from rag.config import load_settings
from rag.ingestion import ChunkingPipeline, IngestionOrchestrator, PDFExtractor
from rag.ingestion.embedding.local import LocalSentenceTransformerEmbedder
from rag.storage import MariaDBRepository, build_engine, build_session_factory

settings = load_settings()
sessions = build_session_factory(build_engine(settings.database))

orchestrator = IngestionOrchestrator(
    extractor=PDFExtractor(
        max_bytes=settings.max_document_bytes,
        max_pages=settings.max_document_pages,
    ),
    chunker=ChunkingPipeline(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ),
    embedder=LocalSentenceTransformerEmbedder(
        settings.embedding_model_path,
        model_name=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
    ),
    repository=MariaDBRepository(
        sessions, embedding_dimension=settings.embedding_dimension
    ),
)

for result in orchestrator.ingest_all(sorted(Path("documents").glob("*.pdf"))):
    print(result.source.name, result.chunk_count, result.error or "")
```

Failures are reported per document rather than raised, so one unreadable file
cannot end a run. Documents whose content has already been ingested are
skipped.

## What ingestion cleans, and why

Extraction is layout-aware, so two-column papers, templated journal articles,
and tables all come back in reading order, with tables as Markdown tables and
mathematics preserved as readable text. On top of that, these contaminants are
removed because they were measured in a real corpus of papers:

| Contaminant | Why it matters | How it is handled |
|---|---|---|
| Running heads, folios, page numbers | Repeat on every page and crowd results | Structural, repetition, and known-artefact rules at page edges |
| Figure and chart label text | Fragments that match numeric queries but answer nothing | The whole figure-text block is discarded |
| Reference lists | Match on wording while answering nothing | Dropped by default, `drop_references=False` to keep |
| Duplicate documents | Fill the top-k with the same passage | Skipped by content hash |
| Fragments | Bare headings displace real passages | Minimum chunk length |

Repeated sentences *within* a page's body are deliberately left alone:
scientific writing legitimately restates definitions, and removing them would
destroy content to solve a problem that only exists at the margins.

## Replacing a component

Every stage is constructor-injected behind an interface, so replacing one means
writing a class and passing it in — no other stage changes:

| To replace | Subclass | Then pass it to |
|---|---|---|
| Document format | `BaseExtractor` | `IngestionOrchestrator(extractor=...)` |
| Chunking strategy | `BaseChunker` | `IngestionOrchestrator(chunker=...)` |
| Embedding model | `BaseEmbedder` | `IngestionOrchestrator(embedder=...)` |
| Database | the `Repository` protocol | either orchestrator |

A `DocxExtractor` or `LatexExtractor` needs only to return an
`ExtractedDocument`; chunking, embedding, storage, and retrieval are unaffected.

## Domain contracts

Components exchange typed domain objects, never loose dictionaries:

```
PDFExtractor → list[Page] → Chunker → list[Chunk] → Embedder → list[EmbeddedChunk]
```

Every domain model is a frozen dataclass with no framework dependency — none of
them import SQLAlchemy. Immutability is deliberate: a chunk's provenance (its
identifiers, page number, and scores) is written once by the component
responsible for it, so document content can never overwrite it. Untrusted,
document-derived values live only in `metadata` mappings, which no component
treats as instructions or control values.

## Installation

```bash
conda env create -f environment.yml
conda activate rag-study-py3.12
```

## MariaDB configuration

Requires **MariaDB 11.7 or later** for native `VECTOR` support (verified
against 12.3.3). Create a database and a dedicated least-privilege user — the
application needs only DML on its schema, never `root` and never DDL:

```sql
CREATE DATABASE rag_study CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE DATABASE rag_study_test CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER 'rag_app'@'localhost' IDENTIFIED BY '<generated-password>';
GRANT SELECT, INSERT, UPDATE, DELETE ON rag_study.* TO 'rag_app'@'localhost';
-- The test schema is disposable; the suite creates and drops its own tables.
GRANT ALL PRIVILEGES ON rag_study_test.* TO 'rag_app'@'localhost';
```

Then copy `.env.example` to `.env` and fill it in. **Never commit `.env`.**

## Local model assets

Python dependencies and model weights are provisioned separately. Weights live
outside the package and are never downloaded at runtime:

```
models/
├── embeddings/all-MiniLM-L6-v2/
└── rerankers/
```

`RAG_EMBEDDING_MODEL_PATH` points at the model directory. Loading fails with an
explicit error if it is missing — it never falls back to a network download.
The `models/` directory is gitignored.

## Running the tests

```bash
just check-all          # lint, format check, type check, tests
pytest                  # tests only
pytest -m "not integration"   # skip tests needing a live MariaDB
```

Integration tests are skipped, not failed, when no database is reachable.

## Contributing

Branch strategy, QA requirements, and the changelog convention are documented
in [CONTRIBUTING.md](CONTRIBUTING.md).
