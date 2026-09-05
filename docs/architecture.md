# Architecture — Phase 1

This document is the Phase 1 ("Architecture") deliverable for the RAG
package: package structure, domain contracts, interfaces, dependency graph,
database schema, and dependency plan. No implementation code is written in
this phase — that begins in Phase 2 ("Foundation").

## 1. Philosophy and priority order

> Simplicity. Modularity. Independence. Maintainability.

When a decision has to be made, it is made in this order:

1. Simplicity
2. Modularity
3. Independence
4. Maintainability
5. Correctness
6. Testability
7. Extensibility
8. Performance optimisation

Every abstraction introduced below is justified by **replaceability** (a
model, a chunking strategy, a ranking algorithm) or **separation of
responsibility** (processing vs. persistence). Nothing is introduced "for
the future" — see §11 for places where a future need is deliberately left
as an open question rather than pre-built.

## 2. Package structure

```text
src/
└── rag/
    ├── domain/
    │   ├── __init__.py
    │   └── models.py            # Document, Page, Chunk, EmbeddedChunk,
    │                             # RankedChunk, RerankedChunk,
    │                             # PromptContext, PromptMetadata
    │
    ├── storage/
    │   ├── __init__.py
    │   ├── orm.py                # SQLAlchemy declarative models
    │   ├── engine.py             # MariaDB engine/session factory
    │   ├── repository.py         # Repository protocol + interface
    │   └── mariadb_repository.py # SQLAlchemy/MariaDB implementation
    │
    ├── ingestion/
    │   ├── __init__.py
    │   ├── interfaces.py         # BaseExtractor, BaseChunker, BaseEmbedder
    │   ├── extraction/
    │   │   ├── pdf.py            # PDFExtractor (layout-aware, via pymupdf4llm)
    │   │   └── cleaning.py       # encoding repair + page-furniture removal
    │   ├── chunking/
    │   │   ├── splitters.py      # MarkdownHeaderChunker, RecursiveChunker
    │   │   └── pipeline.py       # ChunkingPipeline + contamination filters
    │   ├── embedding/
    │   │   └── local.py          # LocalSentenceTransformerEmbedder
    │   └── orchestrator.py       # IngestionOrchestrator
    │
    └── retrieval/
        ├── __init__.py
        ├── interfaces.py         # BaseRanker, BaseReranker, BasePromptAugmenter
        ├── ranking/
        │   └── cosine.py         # CosineSimilarityRanker (DB-side or app-side)
        ├── reranking/
        │   └── noop.py           # initial no-op/pass-through reranker
        ├── prompting/
        │   └── template_augmenter.py
        ├── prompts/
        │   └── retrieval_context_v1.txt
        └── orchestrator.py       # RetrievalOrchestrator

models/                            # local model assets, NOT Python package
├── embeddings/
│   └── all-MiniLM-L6-v2/          # local copy of the embedding model
└── rerankers/
    └── ...                        # populated only if/when a real reranker lands

tests/
├── domain/
├── storage/
├── ingestion/
├── retrieval/
└── integration/
```

**Deliberate deviation from the earlier sketch:** `domain/` and `storage/`
are *not* nested inside `ingestion/` or `retrieval/`, and there is no
top-level `orchestration/` package above the pipelines. `domain` and
`storage` are shared infrastructure that both pipelines sit on top of. This
is what makes the "delete `retrieval/` and `ingestion/` still works" test
(§35) literally true: deleting either pipeline directory removes only files
that import `domain`/`storage` downward, never files that something else
imports upward.

## 3. Dependency graph

```text
            config      domain      model_assets
          (no internal dependencies; leaves)
                    ▲
                    │
                 storage
       (depends on: domain, config)
                    ▲
        ┌───────────┴───────────┐
        │                       │
    ingestion                retrieval
(depends on: domain,     (depends on: domain,
 storage, config,         storage, config,
 model_assets)            model_assets)
```

**Phase 4 addition — `model_assets`.** Both pipelines load a model from disk:
ingestion embeds chunks, retrieval embeds queries. The rules for doing that
safely (directory must exist, no network, no remote code execution) are
identical, and duplicating them would mean two copies of a security-critical
routine that must be kept in agreement. `rag/model_assets.py` holds that one
implementation. It is infrastructure for a model asset in the same way
`storage` is infrastructure for a database: a leaf module that neither
pipeline owns, and through which neither can reach the other.

These rules are enforced by `tests/test_architecture.py`, which parses the
source rather than trusting review — it is what caught `storage` reaching into
`config` and `config` importing SQLAlchemy.

Rules enforced by this graph and checked in CI/lint (an import-linter or
equivalent rule belongs in Phase 2, not this doc):

- `domain` imports nothing from this project.
- `storage` imports only `domain`.
- `ingestion` imports only `domain` and `storage`.
- `retrieval` imports only `domain` and `storage`.
- `ingestion` never imports `retrieval`; `retrieval` never imports
  `ingestion`.

## 4. Domain contracts (`domain/models.py`)

Plain dataclasses. No SQLAlchemy, no I/O, no framework dependency. Fields
marked **[provenance]** are set once by the producing component and are
never derived from, or overwritable by, document content — this is the
concrete expression of the "content vs. control metadata" security
principle (see §10).

```text
Document
  id                 [provenance] internal identifier (generated, not the filename)
  source_filename     original filename, treated as untrusted display data only
  content_hash        for dedup / integrity, not used as a path
  metadata            dict[str, Any] — untrusted, arbitrary

Page
  document_id         [provenance]
  page_number         [provenance]
  text                 extracted, cleaned text
  ocr_extracted        bool
  metadata             dict[str, Any] — untrusted

Chunk
  id                  [provenance] generated
  document_id         [provenance]
  page_number         [provenance]
  text                 chunk text
  section              str | None  — structural heading path, e.g. "2 > 2.1"
  headers              list[str]
  metadata             dict[str, Any] — untrusted

ExtractedDocument
  document            Document
  pages               tuple[Page, ...]

EmbeddedChunk
  chunk               Chunk
  vector               tuple[float, ...] — immutable, keeps domain numpy-free
  model_name           [provenance] e.g. "all-MiniLM-L6-v2"
  model_dimension      [provenance] derived from the vector, so it cannot drift

RankedChunk
  chunk               Chunk
  score                [provenance] set only by the Ranker, never by document content
  rank                 [provenance]

RerankedChunk
  ranked_chunk         RankedChunk
  rerank_score         [provenance] set only by the Reranker

PromptContext
  chunks               list[RerankedChunk]
  rendered_text        str — the assembled context block
  prompt_metadata      PromptMetadata

PromptMetadata
  prompt_name          [provenance]
  prompt_version       [provenance]
```

`Document`/`Page`/`Chunk` carry a `metadata: dict` bucket for whatever the
extractor/chunker legitimately wants to surface (e.g. PDF author field,
detected language). That bucket is explicitly documented as **untrusted
data** and is never read by the ranker, reranker, or prompt augmenter as if
it were a control value (score, scope, trust level) — those live only in
the `[provenance]` fields above, which document-derived code paths never
write to.

## 5. Interfaces

Each interface answers: input, output, responsible for, does not know about.

### Ingestion (`ingestion/interfaces.py`)

**`BaseExtractor`**
```text
INPUT:     Path
OUTPUT:    ExtractedDocument (the Document record plus its Pages)
RESPONSIBLE FOR:  document-specific extraction + cleaning, page boundaries, metadata
DOES NOT KNOW ABOUT: chunking, embedding, ranking, storage, prompting
```

**`BaseChunker`**
```text
INPUT:     list[Page]
OUTPUT:    list[Chunk]
RESPONSIBLE FOR:  document structure detection, chunk construction, size/overlap
DOES NOT KNOW ABOUT: embedding, storage, ranking, prompting
```

**`BaseEmbedder`**
```text
INPUT:     list[Chunk]   (batched)
OUTPUT:    list[EmbeddedChunk]
RESPONSIBLE FOR:  loading a local model, producing vectors, batching
DOES NOT KNOW ABOUT: extraction, storage, ranking, prompting
```

### Retrieval (`retrieval/interfaces.py`)

**`BaseRanker`**
```text
INPUT:     query embedding, candidate source (via Repository), top_k
OUTPUT:    list[RankedChunk]
RESPONSIBLE FOR:  producing an initial relevance ordering
DOES NOT KNOW ABOUT: extraction, chunking, embedding generation, reranking, prompting
```

**`BaseReranker`**
```text
INPUT:     list[RankedChunk]
OUTPUT:    list[RerankedChunk]
RESPONSIBLE FOR:  refining an existing ranking
DOES NOT KNOW ABOUT: extraction, chunking, storage, prompting
```

**`BasePromptAugmenter`**
```text
INPUT:     list[RerankedChunk]
OUTPUT:    PromptContext
RESPONSIBLE FOR:  assembling top-k context + provenance, applying a versioned
                  template, maintaining instruction/data separation
DOES NOT KNOW ABOUT: extraction, chunking, storage, ranking algorithms, and
                     it never calls an LLM
```

### Shared (`storage/repository.py`)

A single small `Repository` protocol, not a generic ORM-exposing CRUD
layer. It only has the methods each pipeline actually needs:

```python
class Repository(Protocol):
    # used by ingestion
    def document_exists(self, content_hash: str) -> bool: ...
    def save_document(self, document: Document) -> None: ...
    def save_pages(self, pages: list[Page]) -> None: ...
    def save_chunks_with_embeddings(self, chunks: list[EmbeddedChunk]) -> None: ...

    # used by retrieval
    def similarity_search(
        self, query_vector: Sequence[float], top_k: int
    ) -> list[RankedChunk]: ...
```

`similarity_search` returns `RankedChunk` (score already attached) rather
than raw rows, so the `CosineSimilarityRanker` can either delegate scoring
to the database (`VEC_DISTANCE`, see §6) or compute it in Python over
vectors the repository returns — the caller (the `Ranker`) doesn't need to
know which. This is what keeps §19's "don't force DB-side ranking" option
open without changing the interface later.

## 6. Database schema

Targeting MariaDB 11.7+, which supports a native `VECTOR` column type and
vector indexes — confirm the installed server version before Phase 2 (see
§11); if it predates 11.7, `embeddings.vector` becomes a `BLOB` of packed
floats and `CosineSimilarityRanker` computes similarity in Python instead of
via SQL. Either way, only the `Ranker` implementation and this one column
type change — no other component is affected.

```text
documents
  id              CHAR(36) PK   -- generated UUID, not derived from filename
  source_filename VARCHAR(512)  -- untrusted, display-only
  content_hash    CHAR(64)
  metadata        JSON
  created_at      DATETIME

pages
  id              CHAR(36) PK
  document_id     CHAR(36) FK -> documents.id
  page_number     INT
  text            LONGTEXT
  ocr_extracted   BOOLEAN
  metadata        JSON
  UNIQUE (document_id, page_number)

chunks
  id              CHAR(36) PK
  document_id     CHAR(36) FK -> documents.id
  page_number     INT
  section         VARCHAR(512) NULL
  headers         JSON
  text            TEXT
  metadata        JSON
  created_at      DATETIME

embeddings
  id              CHAR(36) PK
  chunk_id        CHAR(36) FK -> chunks.id
  embedding       VECTOR(384)        -- dimension matches model_dimension below
  model_name      VARCHAR(255)       -- e.g. "all-MiniLM-L6-v2"
  model_dimension INT                -- e.g. 384; guards against silent mismatch
  created_at      DATETIME
  VECTOR INDEX ix_embeddings_embedding (embedding) DISTANCE=cosine
```

Notes:

- The vector column is named `embedding`, not `vector`: `vector` is a reserved
  word in MariaDB 11.7+, and SQLAlchemy's MySQL dialect does not yet quote it
  automatically. The ORM attribute is still `vector`; only the column name
  differs. This was found during Phase 2 implementation.

- `384` matches the default `all-MiniLM-L6-v2` model chosen for this phase,
  but is a config value, not a hard-coded assumption baked into the schema
  design — a differently-sized model requires a migration, and `model_name`
  + `model_dim` on every row make a mismatch detectable rather than a silent
  bug.
- `documents.id`/`chunks.id`/etc. are generated identifiers, never derived
  from user-supplied filenames — this is what keeps the storage layer safe
  from path-traversal-style attacks when file storage is added in Phase 3
  (see §10).
- `metadata` JSON columns are explicitly the untrusted bucket; provenance
  (ids, page numbers, section path, model identity) lives in real typed
  columns, mirroring the domain model split in §4.

## 7. Prompt versioning

```text
retrieval/prompts/
    retrieval_context_v1.txt
```

`PromptMetadata{prompt_name, prompt_version}` is attached to every
`PromptContext` the augmenter produces. The augmenter resolves a template
only by `(prompt_name, prompt_version)` against this known local directory
— never by a caller-supplied filesystem path — so prompt selection can't be
turned into arbitrary file inclusion (see §10).

## 8. Dependency plan (Conda / pip)

The current `pyproject.toml` and `example.environment.yml` are leftover
generic-template boilerplate (`my-project-name`, a `sesh` git dependency,
`torch` with no RAG-related use, `src/package/`). This phase only records
what should replace them; the actual edit is Phase 2 ("Foundation") work,
done alongside creating `src/rag/`.

Planned `environment.yml` dependencies:

```text
python=3.12
pymupdf              # PDF extraction
pymupdf4llm          # markdown-oriented PDF extraction for chunking
sqlalchemy           # ORM
mariadb              # MariaDB Python connector
sentence-transformers  # local embedding model loading (offline mode)
numpy                # vector math (fallback ranking path, general use)
pytest, pytest-cov   # testing
ruff                 # lint + format
mypy                 # type checking
```

Every entry has a concrete reason tied to a component in §2/§5. Nothing is
added speculatively. `torch` is a transitive dependency of
`sentence-transformers`, not a direct one — it is not listed as a top-level
project dependency.

## 9. Local model asset convention (offline requirement)

```text
models/
├── embeddings/
│   └── all-MiniLM-L6-v2/     # local copy of the model files
└── rerankers/
    └── ...                    # added only when a real reranker is implemented
```

- Model directory path comes from configuration (an environment variable),
  never hard-coded.
- `LocalSentenceTransformerEmbedder` loads with the relevant "local files
  only" / offline flag set, and raises a clear, specific error if the
  configured path doesn't exist or doesn't contain a valid model — it never
  silently falls back to fetching from the network.
- This directory holds binary model weights, not Python source; it stays
  out of `src/rag` entirely, keeping "Python package dependencies" and
  "model/data assets" (§32 of the original spec) visibly distinct.

## 10. Security / threat model (condensed)

Full mitigations follow the "Secure by design, simple by implementation"
philosophy from the spec: established mitigations, not bespoke security
infrastructure. One row per component:

| Component | Input | Trust level | Realistic attacks | Mitigation |
|---|---|---|---|---|
| PDF ingestion | file path | untrusted file content | malformed/oversized PDF, decompression bombs, pathological page counts | PyMuPDF only (no custom parser); configurable max file size / max page count; catch and skip malformed documents without crashing the run |
| File storage | filename/path | untrusted | path traversal (`../../etc/passwd`), filename collisions | generated internal IDs used as storage keys/filenames; all paths resolved and checked to stay under a configured storage root |
| Extraction/cleaning | raw PDF text | untrusted | embedded control characters, adversarial formatting | output treated as plain data downstream; no execution/interpretation of content |
| Chunking | Page.text | untrusted | pathological input sizes driving semantic chunking cost | configurable max chunk count / max chunk size |
| Embedding | Chunk.text | untrusted | oversized batch causing memory exhaustion | configurable embedding batch size cap |
| Database | all persisted fields | mixed (provenance = trusted-once-set, metadata = untrusted) | SQL injection, credential leakage | SQLAlchemy parameter binding everywhere, no f-string SQL; least-privilege DB user; credentials via env vars, never logged |
| Ranking | query + stored vectors | query is external input | requesting an unbounded `top_k` | configurable max `top_k`; score always computed by the Ranker, never read from document metadata |
| Reranking | RankedChunk list | same as above | none new | reranker only reorders/rescopes what the ranker already returned |
| Prompt augmentation | RerankedChunk list (i.e. retrieved document content) | **untrusted** | prompt injection / indirect prompt injection embedded in document text | template keeps a hard structural separation between application instructions and retrieved content; retrieved text is never concatenated into an "instructions" section; no string-matching "injection filter" is relied on as the actual defense |
| Configuration | env vars / `.env` | trusted (operator-controlled) | secrets committed to source control | `.env.example` with placeholders only, real `.env` gitignored |
| Logging | pipeline internals | n/a | leaking full document text, retrieved context, or credentials into logs | log identifiers/sizes/durations, not full content or connection strings |
| Model loading | local file path | trusted (operator-provisioned assets) | loading an unexpected/untrusted model path | path comes only from configuration, not from any request-time input; offline mode set explicitly; fail closed if missing |

The most important structural point carried through from the domain model
in §4: **a chunk's provenance and control metadata (id, ranking score,
model identity) are set exactly once, by the component responsible for
them, and document content can never write to those fields.** That single
rule is what prevents a malicious document from inflating its own retrieval
score or spoofing which document it came from — it is a design property,
not a runtime check.

## 11. Open questions / suggested improvements

Flagged rather than silently decided, per the instruction to surface
architectural ambiguities before implementation:

1. ~~**MariaDB version.**~~ **Resolved in Phase 2.** The installed server is
   MariaDB 12.3.3, so native `VECTOR`, `VEC_DISTANCE_COSINE`, and
   `VECTOR INDEX ... DISTANCE=cosine` are all available and verified working.
   The BLOB + numpy fallback is not needed; if a future deployment targets an
   older server, only the column type and the ranker implementation change.
2. ~~**Semantic chunking.**~~ **Deferred in Phase 3, deliberately.** Structural
   (Markdown heading) plus recursive splitting produces well-formed chunks with
   full section provenance on the real corpus. Semantic chunking would require
   the chunker to hold an embedding function, weakening the rule that a chunker
   knows nothing about embeddings, and would embed the corpus twice. It remains
   a drop-in behind `BaseChunker` if retrieval quality later shows a need.

3. **Contamination handling is evidence-based, not exhaustive.** Phase 3
   measured the real corpus and handled what actually occurs: duplicate
   documents, figure/chart label text, reference lists, page furniture, and
   fragments. Mid-sentence line breaks were measured at zero occurrences —
   `pymupdf4llm` already rejoins wrapped lines — so no code was written for
   them. Two known limitations are accepted for now: a single table larger than
   `chunk_size` will be split across chunks and lose its header row on the
   second piece, and OCR'd pages of poor scans yield low-quality text that is
   flagged via `ocr_extracted` rather than filtered.

4. ~~**Initial `BaseReranker` implementation.**~~ **Resolved in Phase 5.**
   `PassthroughReranker` remains the default when no reranker is provisioned,
   and `CrossEncoderReranker` is available when one is. Measured against the
   real corpus, the cross-encoder removes the clearest failure of vector-only
   ranking — an "Author Contributions" section appearing in the top three for
   a question about method — but its ordering of genuinely relevant academic
   passages is not consistently better. The model is trained on web search
   relevance, not academic prose. Judging this properly needs a labelled query
   set, which is the recommended next step rather than further tuning.

5. **Superseded — original reranker note.** The spec allows a literal
   no-op (pass the ranked list through unchanged) as an acceptable Phase 4
   starting point. Recommended: implement it as an explicit
   `PassthroughReranker`, not by skipping the abstraction — this keeps the
   orchestrator wiring identical to what it'll look like once a real
   cross-encoder reranker is added.
5. **Scope/filter parameter on `similarity_search`.** No access-control or
   multi-document-scope requirement exists yet. Recommended: give
   `similarity_search` an optional `filters: dict | None = None` parameter
   now (unused, always `None` in Phase 1–4) so a future scope/tenant filter
   doesn't require changing the `Repository` protocol shape later — this is
   the one place a small amount of forward-looking surface seems justified
   by "clean interfaces that make future complexity possible without
   requiring it today" (spec §34), rather than actually implementing
   access control now.
6. **`content_hash` on `Document`.** Included in §4/§6 for basic dedup, but
   no dedup *behavior* is specified anywhere in the original spec.
   Resolved in Phase 3: the corpus contained a byte-identical duplicate pair,
   so `Repository.document_exists` and an orchestrator skip were added.

## Architectural review (spec §35, answered)

1. Can each module be replaced independently? Yes — each concrete class
   sits behind one of the interfaces in §5.
2. Can ingestion exist without retrieval? Yes — ingestion depends only on
   `domain`/`storage`.
3. Can retrieval exist without ingestion, given existing data? Yes, same
   reasoning in reverse.
4. Are database concerns isolated from processing? Yes — only `storage/`
   imports SQLAlchemy.
5. Are domain models independent of SQLAlchemy? Yes — `domain/models.py`
   has no ORM imports; `storage/orm.py` is a separate mapping layer.
6. Can embedding models be swapped? Yes, behind `BaseEmbedder`.
7. Can chunking strategies be swapped? Yes, behind `BaseChunker`.
8. Can ranking strategies be swapped? Yes, behind `BaseRanker`.
9. Can reranking strategies be swapped? Yes, behind `BaseReranker`.
10. Can prompt versions change without changing retrieval logic? Yes — the
    augmenter resolves templates by name/version identifier only.
11. Can the system operate offline? Yes, given local model assets per §9;
    no runtime component fetches anything over the network.
12. Can the system be understood without understanding every module? Yes —
    §5's per-interface input/output/responsibility/non-responsibility table
    is designed to make each piece legible on its own.
13. Have we introduced any abstraction without a real benefit? The one
    borderline case is the `filters` parameter in §11.3, explicitly called
    out as unused-but-reserved rather than snuck in silently.
14. Is there a simpler way to implement each component? Not currently
    identified beyond what's already noted in §11.
