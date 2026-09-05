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
└── storage/
    ├── vector.py               # MariaDB VECTOR column type
    ├── orm.py                  # SQLAlchemy models (the only ORM mapping)
    ├── engine.py               # engine / session construction
    ├── repository.py           # the storage contract both pipelines use
    └── mariadb_repository.py   # MariaDB implementation
```

`ingestion/` and `retrieval/` arrive in Phases 3 and 4.

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
