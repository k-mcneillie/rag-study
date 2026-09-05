# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Phase 1 architecture document (`docs/architecture.md`) covering package
  structure, domain contracts, interfaces, dependency graph, database
  schema, and dependency plan for the RAG package.
- Git workflow: protected `main`, `dev` as the primary working branch,
  feature branches per issue/ticket (see `CONTRIBUTING.md`).
- Phase 2 foundation:
  - Domain contracts in `rag.domain.models` — `Document`, `Page`, `Chunk`,
    `EmbeddedChunk`, `RankedChunk`, `RerankedChunk`, `PromptContext`, and
    `PromptMetadata`, as frozen dataclasses with no framework dependency.
  - Environment-driven configuration in `rag.config`, with credentials kept
    out of `repr` output and required variables reported by name only.
  - `rag.storage`: a `Vector` type for MariaDB's native `VECTOR` column,
    SQLAlchemy ORM models, engine/session construction, the `Repository`
    protocol, and the MariaDB implementation with database-side cosine
    ranking via `VEC_DISTANCE_COSINE`.
  - Allow-listed search filters, bounded `top_k`, and embedding-dimension
    validation, so untrusted input cannot widen a query or exhaust resources.
  - `environment.yml` for the `rag-study-py3.12` Conda environment, and
    `.env.example` documenting every recognised variable.
  - Test suite: 44 tests covering the domain contracts, configuration, the
    vector type, an in-memory repository proving the storage contract is
    satisfiable without a database, and live MariaDB integration tests
    including SQL-injection and resource-limit cases.

- Phase 3 ingestion pipeline:
  - `PDFExtractor`, using `pymupdf4llm` for layout-aware extraction, so
    two-column papers, templated journal articles, and tables come back in
    reading order. Its layout models ship inside the package, so extraction
    needs no network. PDFs are validated by signature rather than extension,
    with size and page-count limits, and a malformed document fails alone.
  - Text cleaning built on `ftfy` for encoding repair, plus removal of page
    furniture through three complementary rules: structural (bare page
    numbers, folios, publisher stamps), repetition-based with digit masking
    (running heads that carry a page number), and a short list of known
    artefacts.
  - `MarkdownHeaderChunker` and `RecursiveChunker`, thin adapters over
    `langchain_text_splitters`, returning domain chunks that carry their own
    page and section provenance. Heading trails survive page breaks,
    including the ancestors of a subsection that starts a new page.
  - `ChunkingPipeline`, which chunks by structure then by size and removes
    contamination measured in the real corpus: reference lists, figure and
    chart label text, and fragments.
  - `LocalSentenceTransformerEmbedder`, loading weights only from a local
    directory with remote code refused, failing loudly rather than falling
    back to a download.
  - `scripts/create_schema.py`, an administrative operation, because the
    application user deliberately holds no DDL privileges and so cannot
    create its own tables.
  - `IngestionOrchestrator`, holding workflow only, which reports per-document
    failures instead of raising so one bad file cannot end a batch, and skips
    documents whose content has already been ingested.

### Changed

- `pyproject.toml` now describes this project rather than the upstream
  template: real dependencies, Ruff configured to enforce Google-style
  docstrings (pydocstyle) and security lints (bandit), and Mypy and Ruff both
  targeting Python 3.12 (they were previously mismatched at 3.10).
- The embedding column is named `embedding` rather than `vector`, because
  `vector` is a reserved word in MariaDB 11.7+ that SQLAlchemy's MySQL dialect
  does not quote automatically. `docs/architecture.md` records the change.
- `.gitignore` excludes local model assets under `models/`.
- `Repository` gained `document_exists`, so ingestion can skip content it has
  already stored. The corpus contained a byte-identical duplicate pair, which
  would otherwise have appeared twice in every result set.
- `BaseExtractor` returns an `ExtractedDocument` rather than a bare list of
  pages: the extractor reads the file, so it is the only component positioned
  to derive the content hash and the document's metadata.

### Removed

- Template scaffolding that did not apply to this project: the `src/package`
  placeholder, the `sesh` and `torch` dependencies, the placeholder baseline
  tests, and `example.environment.yml`.
