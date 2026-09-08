# Architecture

This document describes the structure of the package as it is implemented: its
boundaries, the contracts its components exchange, the direction of its
dependencies, and the rules that keep those dependencies in order. It also
records the design priorities that decide contested questions, and the known
weaknesses that follow from them.

The dependency rules stated here are enforced by `tests/test_architecture.py`,
which parses the source. The document and the code cannot silently diverge on
that point.

## 1. Priority order

When a design decision is contested, it is resolved in this order:

1. Simplicity
2. Modularity
3. Independence
4. Maintainability
5. Correctness
6. Testability
7. Extensibility
8. Performance

Every abstraction in the package is justified either by replaceability — a
model, a chunking strategy, a ranking algorithm — or by separation of
responsibility, chiefly processing from persistence. Nothing is introduced for
a need that does not yet exist; deliberately deferred work is listed in
[future-work.md](future-work.md).

## 2. System boundaries

The package accepts source documents and a query. It produces ranked, cited
context: a block of retrieved text with the provenance of every passage in it.
It does not call a language model.

Answering is provided by a separate package, `rag.generation`, which consumes
the retrieval output from outside. An HTTP API over the query side lives in
`service/`, and a chat interface that talks only to that API lives in `app/` —
both outside the package entirely, both entry points like `scripts/`. All of
it is optional: retrieval works, and is tested, with no model service
configured, the API not run, and the interface not installed.

The prompt is built and consumed entirely inside the service. No endpoint
returns an assembled `PromptContext` to a caller, so the boundary between
application instructions and untrusted document content cannot be moved by a
client (§4).

Model weights are local files, provisioned once at setup. No component reaches
the network at run time.

## 3. The two pipelines

Ingestion and retrieval are independent. They share the domain contracts and
the storage interface, and nothing else. Neither imports the other. Either
directory could be deleted and the other would continue to work.

```text
Ingestion
    Source
      ↓
    Extraction
      ↓
    Chunking
      ↓
    Embedding
      ↓
    Storage


Retrieval
    Query
      ↓
    Ranking
      ↓
    Re-ranking
      ↓
    Prompt Augmentation
      ↓
    Context + Metadata
```

Ingestion writes through the storage contract; retrieval reads through it. That
contract is the only point at which the two meet.

## 4. The generation sibling

`rag.generation` turns a `PromptContext` into an answer. It is a sibling of the
retrieval pipeline, not a stage inside it. It imports `rag.domain` and
`rag.config` and nothing else: it cannot see `retrieval`, `ingestion`, or
`storage`, and `tests/test_architecture.py` enforces that.

The boundary between application instructions and untrusted document content is
established once, by the prompt augmenter, and the rendered prompt is sent to
the model verbatim. A client that could reconstruct the prompt from parts would
be able to move that boundary without owning it, so none can: the client
receives a finished `PromptContext` and has no way to rebuild one. The HTTP API
holds to the same line — it exposes an *answer* endpoint that runs generation
server-side, and no endpoint that returns a `PromptContext` (see
[api.md](api.md)).

## 5. Dependency graph

```text
        config      domain      model_assets      leaves: no internal dependencies
                       ▲
                    storage                       depends on domain, config
                       ▲
           ┌───────────┴───────────┐
       ingestion                retrieval          never each other
                                    ╎
                                generation         domain and config only
```

`rag.model_assets` is a leaf shared by both pipelines. Both load a model from
disk — ingestion to embed chunks, retrieval to embed queries — and the rules
for doing so safely are identical and security-critical. Keeping them in one
audited place is preferable to two copies that must be held in agreement.

`rag.assembly` sits above the query side: it depends on `retrieval`, `storage`,
and `generation`, and composes them for the entry points that answer or
retrieve. It never imports `ingestion`, so the pipeline boundary is unaffected.
Ingestion is composed separately, in `scripts/ingest.py`.

The rules enforced by `tests/test_architecture.py` are, verbatim, the
`ALLOWED_IMPORTS` table it parses the source against:

| Submodule | May import from within the package |
|---|---|
| `config` | (nothing) |
| `domain` | (nothing) |
| `model_assets` | (nothing) |
| `feedback` | `domain` |
| `storage` | `domain`, `config` |
| `ingestion` | `domain`, `storage`, `model_assets`, `config` |
| `retrieval` | `domain`, `storage`, `model_assets`, `config` |
| `generation` | `domain`, `config` |
| `assembly` | `domain`, `config`, `storage`, `retrieval`, `generation`, `model_assets` |

Four further checks in the same file assert that the pipelines never import
each other, that `generation` imports neither pipeline nor `storage`, that no
`domain` module imports a third-party framework, that only `storage` imports
SQLAlchemy, and that nothing under `src/rag` imports an entry point
(`service/`, `app/`) or a web framework (`fastapi`, `starlette`, `chainlit`) —
the package must not depend on what depends on it.

## 6. Domain contracts

`rag/domain/models.py` holds eleven frozen dataclasses and no third-party
imports. Every component speaks these and nothing else, so a change of database
or model provider cannot reach the data contract.

```text
Document ─► Page ─► Chunk ─► EmbeddedChunk ─► RankedChunk ─► RerankedChunk ─► PromptContext ─► Answer
```

| Contract | Carries |
|---|---|
| `Document` | generated id, source filename, content hash, metadata, timestamp |
| `Page` | document id, page number, text, OCR flag, metadata |
| `ExtractedDocument` | a `Document` and its `Page` tuple — the extractor's output |
| `Chunk` | document id, page number, text, section path, heading trail, OCR flag, metadata |
| `EmbeddedChunk` | the chunk, its vector, the model that produced it; `model_dimension` is derived from the vector |
| `RankedChunk` | the chunk, a relevance score, a rank |
| `RerankedChunk` | the ranked chunk, a rerank score, a rank; `initial_score` reads the pre-rerank score |
| `PromptContext` | the reranked chunks, the rendered context, the prompt's identity |
| `PromptMetadata` | prompt name, prompt version |
| `AnswerDelta` | one streamed fragment of text, and whether it is reasoning |
| `Answer` | the answer text, the reasoning, the model name, the prompt's identity |

### Provenance versus metadata

Every model is frozen, and its fields divide in two.

- **Provenance and control** — identifiers, page numbers, section paths, scores,
  rank, model identity, the OCR flag. Written once, by the component
  responsible, and never derived from document content.
- **`metadata`** — a mapping of untrusted, document-derived values, such as a
  PDF's author field. No component reads it as an instruction, an identifier,
  or a score.

A document therefore cannot inflate its own ranking, claim to be a different
source, or forge a citation, because nothing that reads a score or an
identifier reads it from document-supplied data. This is a structural property
of the frozen models, not a runtime check.

## 7. Interfaces

Each interface is an abstract base class stating what it receives, what it
returns, what it is responsible for, and what it knows nothing about.

### Ingestion — `rag/ingestion/interfaces.py`

| Interface | Input → output | Responsible for |
|---|---|---|
| `BaseExtractor` | `Path` → `ExtractedDocument` | reading the source, page boundaries, document-specific cleaning, metadata |
| `BaseChunker` | `list[Page]` → `list[Chunk]` | detecting document structure, deciding chunk boundaries, size, overlap |
| `BaseEmbedder` | `Sequence[Chunk]` → `list[EmbeddedChunk]` | loading a local model, producing one vector per chunk, batching; exposes `model_name` and `dimension` |

### Retrieval — `rag/retrieval/interfaces.py`

| Interface | Input → output | Responsible for |
|---|---|---|
| `BaseQueryEmbedder` | `str` → `tuple[float, ...]` | loading a local model, encoding a single query; exposes `dimension` |
| `BaseRanker` | query vector, `top_k`, optional filters → `list[RankedChunk]` | producing an initial relevance ordering |
| `BaseReranker` | query, `Sequence[RankedChunk]`, `top_k` → `list[RerankedChunk]` | refining an ordering; may only reorder or narrow, never fetch more |
| `BasePromptAugmenter` | query, `Sequence[RerankedChunk]` → `PromptContext` | rendering a versioned template, keeping instructions separate from retrieved content, recording the template's identity; never calls a model |

### Generation — `rag/generation/interfaces.py`

`BaseChatModel` exposes `model_name` and an abstract `stream(context)` that
yields `AnswerDelta` fragments. A concrete `answer(context)` consumes that
stream into an `Answer`; consuming a stream is not a decision each client makes
differently.

### Storage — `rag/storage/repository.py`

```python
class Repository(Protocol):
    def document_exists(self, content_hash: str) -> bool: ...
    def save_ingested_document(
        self,
        document: Document,
        pages: Sequence[Page],
        embedded_chunks: Sequence[EmbeddedChunk],
    ) -> None: ...
    def similarity_search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[RankedChunk]: ...
```

The protocol exposes no sessions, tables, or query objects. Ingestion uses only
`document_exists` and `save_ingested_document`; retrieval uses only
`similarity_search`. `save_ingested_document` is one operation rather than
three because an ingested document is only useful complete: a document row
without its chunks is invisible to retrieval, yet its content hash makes every
later attempt look like a duplicate to skip. Writing everything in one
transaction means the store holds either the whole document or none of it.

`similarity_search` returns `RankedChunk` with the score already attached, so
the ranker does not know whether the score was computed in the database or in
Python. `filters` keys are restricted to `ALLOWED_SEARCH_FILTERS`, which
currently contains only `document_id`; it is the seam for future access
control, and no access rules are applied today.

## 8. Current implementations

| Interface | Implementations |
|---|---|
| `BaseExtractor` | `PDFExtractor` (layout-aware, via `pymupdf4llm`) |
| `BaseChunker` | `ChunkingPipeline` (structure then size then contamination filters — the default), `MarkdownHeaderChunker`, `RecursiveChunker`, `SemanticChunker` |
| `BaseEmbedder` | `LocalSentenceTransformerEmbedder` |
| `BaseQueryEmbedder` | `LocalSentenceTransformerQueryEmbedder` |
| `BaseRanker` | `CosineSimilarityRanker` (delegates to `Repository.similarity_search`) |
| `BaseReranker` | `CrossEncoderReranker` (when a model is configured), `PassthroughReranker` (otherwise) |
| `BasePromptAugmenter` | `TemplatePromptAugmenter` |
| `BaseChatModel` | `OllamaChatModel`, `OpenAICompatibleChatModel` |
| `Repository` | `MariaDBRepository` |

The ingestion and retrieval pipelines are documented in detail in
[ingestion.md](ingestion.md) and [retrieval.md](retrieval.md); generation in
[generation.md](generation.md).

## 9. Storage layer

`rag/storage/` is the only package that imports SQLAlchemy.

- `orm.py` — the declarative models `DocumentRow`, `PageRow`, `ChunkRow`,
  `EmbeddingRow`. It is the only place the package maps to tables.
- `vector.py` — a `Vector` SQLAlchemy type for MariaDB's native `VECTOR(n)`
  column, converting to and from plain tuples through `VEC_FromText` and
  `VEC_ToText` inside bound parameters, and decoding with `json.loads`.
- `engine.py` — `build_url`, `build_engine`, `build_session_factory`. URL
  construction lives here rather than on the settings object, so configuration
  stays free of SQLAlchemy.
- `mariadb_repository.py` — `MariaDBRepository`, which implements the protocol.
  Similarity is computed in the database: `VEC_DISTANCE_COSINE` runs against the
  vector index, so only the top-k rows return to the application.

The schema and its operational details are in
[database-cheatsheet.md](database-cheatsheet.md).

## 10. Composition

Every stage is injected through a constructor. There is no factory, registry,
or plugin discovery, with one deliberate exception: `rag.generation.providers`
maps a configured provider name to a chat-model class, because which model
service answers is a deployment decision and requiring a code edit to change it
would make the swappable interface swappable only in principle. It is a
dictionary and a lookup.

`rag.assembly` holds the composition for the query side —
`build_retrieval_orchestrator`, `build_reranker`, `build_chat_model` — used by
`scripts/query.py`, `scripts/answer.py`, and `service/app.py`. The ingestion
pipeline is composed in `scripts/ingest.py`, and again in
`service/ingest_wiring.py` for the upload endpoint (a few lines of
construction: `rag.assembly` must never import `ingestion`, so the wiring
cannot live there). The chat app in `app/` composes nothing from `rag` — it is
an HTTP client of `service/` and imports the package not at all. Configuration
selects
implementations: an unset `RAG_RERANKER_MODEL_PATH` yields the passthrough
reranker, `RAG_SEMANTIC_CHUNKING` adds the semantic chunker, `RAG_LLM_PROVIDER`
selects the chat-model client, and `RAG_PROMPT_NAME` / `RAG_PROMPT_VERSION`
select the template.

A configured model that cannot be loaded is an error, never a silent downgrade,
so a misconfiguration is not mistaken for a deliberate choice to skip a stage.

## 11. Extension points

Adding a capability means implementing one interface and injecting it. Nothing
downstream changes. For example:

```text
New extractor
    → subclass BaseExtractor
    → inject into IngestionOrchestrator (scripts/ingest.py:build_orchestrator)
    → no retrieval, storage, or generation change
```

The same pattern applies to chunkers, embedders, rankers, rerankers, prompt
augmenters, the repository, and — with one added line in `CHAT_MODELS` — chat
models. [codebase-cheatsheet.md](codebase-cheatsheet.md) gives the concrete
location for each.

## 12. What holds it correct

| Layer | What it establishes |
|---|---|
| Unit tests | each component in isolation, against fakes |
| Architecture tests | the dependency rules, by parsing the source |
| Integration tests | real MariaDB, real model weights, real vector search |
| Full-cycle test | a generated PDF ingested by one pipeline and retrieved by the other |

Integration tests skip, rather than fail, when the database or the weights are
absent, so the suite runs on any machine. CI deselects them explicitly rather
than letting them skip unnoticed, and runs the architecture tests as their own
step so a dependency violation is reported on its own terms.

Two tests are load-bearing. `test_architecture.py` makes the central
architectural claim executable. `test_a_failed_write_leaves_nothing_behind`
reproduces a data-loss scenario found during review — a document written
without its chunks, permanently skipped on every retry — so it cannot return.

## 13. Known weaknesses

Stated plainly, because they follow from the priorities in §1.

1. **Retrieval quality is unmeasured.** There is no evaluation set, so "better"
   is a matter of reading results. This blocks judging the reranker and is the
   largest gap. See [future-work.md](future-work.md) §1.
2. **The cross-encoder reranker is unproven on this corpus.** It removes the
   clearest failures of vector-only ranking, but on some academic queries it
   ranks a genuinely answering passage below where vector similarity placed it.
   The model is trained on web-search relevance, not academic prose.
3. **Retrieval is purely dense.** There is no keyword or hybrid search, so
   exact identifiers, symbols, and rare terms retrieve less reliably than
   prose.
4. **There are no schema migrations.** Changing the schema on a populated
   database has no supported path. `scripts/create_schema.py` reports drift and
   prints the required `ALTER` statements as a stopgap.
5. **Poor scans produce poor text.** Pages are flagged with `ocr_extracted`,
   and nothing yet acts on the flag.
6. **A table larger than `chunk_size` is split, and the second piece loses its
   header row.**
7. **Single-node assumptions.** There is no connection-pool tuning, no batching
   across documents, and no concurrency model. This is adequate at the current
   scale and unexamined beyond it.
8. **No external security review.** The mitigations in [security.md](security.md)
   are implemented and tested, but no adversarial review by anyone other than
   the author has taken place.
9. **The API has one shared key and no more.** `service/` checks a single
   `RAG_API_KEY` against `X-API-Key`, and runs open when the key is unset.
   There is no per-user identity, rate limiting, or TLS; those belong to a
   reverse proxy. See [api.md](api.md) and [future-work.md](future-work.md).

## 14. Documentation map

| For | See |
|---|---|
| Installing and running the system | [../README.md](../README.md) |
| The ingestion pipeline | [ingestion.md](ingestion.md) |
| The retrieval pipeline | [retrieval.md](retrieval.md) |
| Answering and the chat interface | [generation.md](generation.md) |
| The HTTP API and the chat app | [api.md](api.md) |
| The security model | [security.md](security.md) |
| Operating the database | [database-cheatsheet.md](database-cheatsheet.md) |
| Provisioning model assets | [model-deployment-cheatsheet.md](model-deployment-cheatsheet.md) |
| Navigating the codebase | [codebase-cheatsheet.md](codebase-cheatsheet.md) |
| What is deliberately not built | [future-work.md](future-work.md) |
| Contributing | [../CONTRIBUTING.md](../CONTRIBUTING.md) |
