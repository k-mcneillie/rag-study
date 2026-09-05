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

- Phase 4 retrieval pipeline:
  - `CosineSimilarityRanker`, which delegates the comparison to MariaDB's
    vector index so only the top-k rows return to the application, behind an
    interface that keeps the choice replaceable.
  - `PassthroughReranker`, a real stage that preserves the ranker's ordering,
    so the pipeline is already shaped for the cross-encoder that will replace
    it.
  - `TemplatePromptAugmenter` with file-based prompt versioning. Retrieved
    text is placed inside marked source material the template describes as
    untrusted data, and markers occurring in retrieved content or in the query
    are neutralised so a document cannot forge the boundary and pose as the
    application. Templates resolve by identifier against the package's own
    directory, never by a caller-supplied path.
  - `LocalSentenceTransformerQueryEmbedder`, behind a `BaseQueryEmbedder`
    declared in the retrieval package so that retrieval never imports
    ingestion.
  - `RetrievalOrchestrator`, holding workflow only, which retrieves more
    candidates than it returns so a reranker has room to work, and bounds
    both `top_k` and the widened candidate pool.
  - `rag.model_assets`, one audited implementation of local model loading
    shared by both pipelines.
  - `tests/test_architecture.py`, which parses the source to enforce the
    dependency rules — that the pipelines never import each other, that the
    domain stays framework-free, and that only storage imports SQLAlchemy.

- Phase 5 integration and reranking:
  - `CrossEncoderReranker`, which scores each query/passage pair directly
    rather than comparing pre-computed vectors, applied to the ranker's
    shortlist because it is too slow to run over a store. Both scores are
    kept on every result, so a change in ordering can be traced to the stage
    that caused it.
  - `scripts/ingest.py` and `scripts/query.py`, runnable demonstrations of
    each pipeline. Both are assembly only; the behaviour lives in the package.
  - A full-cycle integration test that ingests a generated PDF with one
    pipeline and retrieves it with the other, through the storage contract.
  - Optional reranker configuration. An unset `RAG_RERANKER_MODEL_PATH`
    means the stage is skipped; a path that is set but unloadable is an
    error, so a misconfiguration is never mistaken for a deliberate choice.

- Prompt name and version are now configuration (`RAG_PROMPT_NAME`,
  `RAG_PROMPT_VERSION`) rather than constructor defaults, so changing the
  prompt in use is not a code change and a past result can be reproduced by
  restoring the configuration that produced it.

- Review remediation:
  - `SemanticChunker`, a standalone chunker that cuts where sentence-to-
    sentence similarity drops rather than at a character count. It depends on
    `TextEmbedder`, a two-method description of the capability it needs, so it
    imports no model or model-loading library and is tested with fixed
    vectors. Composable into `ChunkingPipeline` and opt-in, since it embeds
    every sentence.
  - `docs/system-overview.md`, a complete account of the system including its
    weaknesses, and `docs/future-work.md`, covering what is deliberately not
    built.
  - A rewritten README: five checkable setup steps, worked examples, and a
    troubleshooting table.

- Phase 6 answering and the chat interface:
  - `rag.generation`, which turns a `PromptContext` into an answer. It is a
    sibling of `retrieval`, not a stage inside it: it imports `domain` and
    `config` and nothing else, and `tests/test_architecture.py` enforces that
    it can reach neither pipeline. The retrieval pipeline still calls no
    language model and does not know that this package exists.
  - `BaseChatModel`, two members wide, with two implementations:
    `OllamaChatModel` for Ollama's native API — preferred for a reasoning
    model, because reasoning arrives in its own field rather than in `<think>`
    tags — and `OpenAICompatibleChatModel` for vLLM, LM Studio, llama.cpp,
    OpenAI, Together, Groq and OpenRouter. Shared HTTP plumbing, including
    bearer authentication and error mapping, lives in
    `rag.generation.transport`.
  - Credentials for hosted services: `RAG_LLM_API_KEY` is sent as a bearer
    token when set and omitted entirely when not, and is excluded from
    `repr` for the same reason the database password is. Base URLs that
    already end in `/v1`, as vLLM's and OpenAI's conventionally do, are
    handled without repeating the segment — the alternative is a bare 404
    that reads exactly like a missing model.
  - Which service answers is configuration (`RAG_LLM_PROVIDER` and the
    `RAG_LLM_*` variables), resolved by `rag.generation.providers`. This is
    the one place constructor injection was not enough: the service that
    answers is a deployment decision, and requiring a code edit to change it
    would have made a swappable interface swappable only in principle.
  - `Answer` and `AnswerDelta` domain contracts, keeping a model's reasoning
    separate from its answer rather than concatenated into it.
  - A Chainlit interface in `app/`, outside the package, assembling its own
    pipeline exactly as the scripts do. It streams the answer, renders each
    cited passage as an inspectable side element numbered to match the `[n]`
    markers in the prompt, keeps reasoning in a collapsed step, and refuses to
    call the model at all when retrieval found nothing. The package stays
    synchronous: `asyncio.to_thread` bridges the two worlds in the interface
    alone.
  - Human feedback: thumbs up/down under every answer, appended by
    `rag.feedback` to a JSON Lines file with the model, the prompt version,
    and the provenance of every passage the answer was given. Citations record
    provenance and never passage text, and the log is gitignored, because it
    carries text drawn from the document pool.
  - `scripts/answer.py`, the same loop in the terminal, for seeing what the
    model was actually sent.
  - `docs/interface.md`, covering the boundary, provider selection, reasoning,
    citations, and the feedback format — including a plain statement that vLLM
    is supported by protocol and has not been run against a live vLLM server.

- Bandit security scanning, in `just check-all` and in CI, over `src/`, `app/`
  and `scripts/`. Ruff's flake8-bandit (`S`) rules already ran over every
  file; the two do not overlap completely, and Bandit reports severity and
  confidence, so a finding can be triaged rather than only suppressed. Tests
  are excluded, because they deliberately contain dummy credentials used to
  prove real ones never escape. The scan is clean at the time of writing.

### Fixed

- `ftfy` and `langchain-text-splitters` were imported by `rag.ingestion` but
  never declared in `[project] dependencies`. They were present in the
  development Conda environment, so the omission was invisible locally while
  a clean install of the package — including CI's `pip install .[dev]` —
  lacked both. Mypy reported them as missing imports and, with `ftfy`
  unresolved, `repair_text` as returning `Any`; at run time the ingestion
  pipeline would have failed at import. Both are now declared.

- The chat interface raised `AttributeError` from `dataclasses` on startup.
  Chainlit executes `app/main.py` without registering it in `sys.modules`, and
  a `@dataclass` there resolves the string annotations produced by
  `from __future__ import annotations` by looking its defining module up
  there. An ordinary `import` succeeds where Chainlit fails, so
  `tests/test_app_module.py` reproduces Chainlit's loader rather than
  importing the module the usual way.

- The README's example output quoted a passage extracted from a document in
  `docs/pool/`. The document is a public paper, so nothing confidential was
  exposed, but copying corpus text into a tracked file is the pattern the
  pool rule exists to prevent. Replaced with a synthetic sample.

- The reasoning step rendered *below* the answer it preceded. Chainlit stamps
  a step's start time only when the step is entered as a context manager, so a
  step created and sent by hand carried no start time and was ordered by its
  last update instead. The step is now entered through its context manager,
  and the answer message is not constructed until the first answer fragment
  arrives, so neither can be timestamped ahead of the reasoning it followed.

- The OpenAI-compatible client leaked reasoning into the answer when a
  `<think>` tag arrived split across two streamed fragments. Text that might
  still become a tag is now held back until it is resolved, and flushed when
  the stream ends.

- `scripts/create_schema.py` now reports schema drift. `CREATE TABLE` only
  adds missing tables, so a column added to the models after a database was
  created was silently absent until a query failed at run time with an
  unhelpful "unknown column" error — which is exactly what happened when
  `ocr_extracted` was added. The script now compares the models against the
  live database and prints the `ALTER` statements needed. Columns are read
  from `information_schema` because SQLAlchemy's MySQL reflection cannot
  parse MariaDB's `VECTOR` type.

- **Ingestion was not atomic, and could silently lose a document.** The
  document, its pages, and its chunks were written in three transactions, so
  a failure during chunk writing left a document row carrying its content
  hash but no chunks. That document was invisible to retrieval, while its
  hash made every retry look like a duplicate to skip — lost permanently and
  silently. The three repository methods are replaced by one atomic
  `save_ingested_document`, with a regression test reproducing the original
  scenario.

### Changed

- `just type-check` and CI now run mypy over `app/` and `scripts/` as well as
  `src/`. The entry points hold real logic — assembly, streaming, session
  state — and CI does not install the interface, so this also proves the app
  type-checks against a bare dependency set.

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
- Connection URL construction moved from `DatabaseSettings` into
  `rag.storage.engine`, so configuration is plain data with no SQLAlchemy
  dependency. The architecture tests caught the leak.
- `BaseExtractor` returns an `ExtractedDocument` rather than a bare list of
  pages: the extractor reads the file, so it is the only component positioned
  to derive the content hash and the document's metadata.

- Chunk and page rows are bulk-inserted rather than merged one at a time,
  which previously cost a database round trip per chunk. A 272-chunk paper
  now ingests end to end in about eight seconds.
- `ocr_extracted` moved from the untrusted `metadata` mapping onto `Chunk` as
  a first-class field. The pipeline sets it, not the document, so it belongs
  with the provenance rather than in the bucket documented as untrusted.
- Re-ingesting a document now replaces what is stored rather than failing,
  including when the same content arrives under a different filename.
- CI now runs on `main` and `dev`, deselects integration tests rather than
  letting them skip silently, and checks the architecture rules as their own
  step so a dependency violation is reported on its own terms.

### Removed

- Template scaffolding that did not apply to this project: the `src/package`
  placeholder, the `sesh` and `torch` dependencies, the placeholder baseline
  tests, and `example.environment.yml`.
