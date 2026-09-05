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

### Changed

- `pyproject.toml` now describes this project rather than the upstream
  template: real dependencies, Ruff configured to enforce Google-style
  docstrings (pydocstyle) and security lints (bandit), and Mypy and Ruff both
  targeting Python 3.12 (they were previously mismatched at 3.10).
- The embedding column is named `embedding` rather than `vector`, because
  `vector` is a reserved word in MariaDB 11.7+ that SQLAlchemy's MySQL dialect
  does not quote automatically. `docs/architecture.md` records the change.
- `.gitignore` excludes local model assets under `models/`.

### Removed

- Template scaffolding that did not apply to this project: the `src/package`
  placeholder, the `sesh` and `torch` dependencies, the placeholder baseline
  tests, and `example.environment.yml`.
