# Future work

What the package deliberately does not do yet, why, and what building each thing
would involve. Every entry is a decision, not an oversight: the foundation was
built so these could be added without rework, and each names the seam it would
attach to. Ordered by the value it would add.

## 1. Evaluation harness

The highest-value item, and the one that unblocks judging everything else.

Retrieval quality is currently assessed by reading results. That catches
something obviously wrong — an "Author Contributions" section ranked above a
paper's method — but not whether a change helped. Two questions are unresolved
for want of measurement: whether the cross-encoder reranker earns its cost on
academic prose, and what `chunk_size` is right.

**What it would involve.** A set of 30–50 questions over a known corpus, each
paired with the passages that should be retrieved, and a runner reporting
recall@k, precision@k, and mean reciprocal rank. No new architecture: it reads
through the existing `Repository` and drives the existing orchestrator.

**Intended approach.** An [Inspect AI](https://inspect.aisi.org.uk) harness in
`evals/`, outside `src/rag/` for the same reasons `app/` is outside it —
independent deletability, no widening of the dependency graph the architecture
tests enforce, and freedom to use the async and registry patterns declined
inside the package. Golden labels key on `(content_hash, page_number)` rather
than on `Document.id` or `Chunk.id`, which are fresh UUID4 values on every
ingestion run. Question and answer drafts would be generated from ingested
chunks with the local model and then reviewed by hand, with the retrievability
bias that introduces recorded rather than glossed over. The retrieval task runs
with the reranker on and off, which settles the reranker question with numbers.

**Why it is not built.** It needs a corpus whose right answers are agreed, and
that is a judgement about the documents rather than about the code. The chat
interface's thumbs-up / thumbs-down ratings
([generation.md](generation.md#feedback)) are where the questions will come
from; they cost nothing to accumulate in the meantime.

## 2. Additional document types

Only PDF is supported. `DocxExtractor`, `PowerPointExtractor`, and
`LatexExtractor` were scoped out deliberately.

**What it would involve.** Each is one class implementing `BaseExtractor`,
returning an `ExtractedDocument`. Nothing downstream changes: chunking,
embedding, storage, and retrieval never learn that a new format exists.

**To decide first.** Where format detection lives — probably a small registry
mapping a file's signature to an extractor, not extension-based dispatch, which
is attacker-controlled. And whether the `Page` contract still fits: slides and
LaTeX sources paginate differently from PDFs, and a more general "location"
concept may serve better than a page number.

## 3. Access control and multi-tenancy

The system assumes every caller may see every document.

**What it would involve.** The seam exists and is tested: `similarity_search`
takes a `filters` mapping validated against an explicit allow-list, and the
orchestrator passes it through untouched. Adding scope means adding a column to
`documents`, extending the allow-list, and resolving a caller's permitted scope
before the query — not restructuring retrieval.

**What would need care.** Filtering must happen in the query, not after results
return, or the ranking silently degrades as filtered results are discarded. And
the prompt augmenter would need to refuse to render a passage outside the
caller's scope, as a second check rather than the first line of defence.

## 4. Schema migrations

`scripts/create_schema.py` creates tables and reports drift. There is no path
to change the schema once a database holds data worth keeping.

**What it would involve.** Alembic, with the migration step run by the same
administrative account that owns schema creation, since the application user
holds no DDL rights. The existing schema becomes the initial revision.

**Why it matters sooner than it looks.** A change of embedding model changes
the `VECTOR` column width, which is a migration and a re-embedding of every
stored chunk. Better to have the mechanism before that is needed.

## 5. Hybrid retrieval

Retrieval is purely dense. Dense retrieval is weak exactly where papers are
strongest — exact identifiers, symbols, statistical notation, rare terms —
because an embedding blurs a token into its neighbourhood.

**What it would involve.** A `BaseRanker` implementation combining MariaDB
full-text search with the existing vector search and merging the two orderings
(reciprocal rank fusion is the usual choice). It slots in as one more ranker;
the orchestrator does not change.

## 6. Known limitations worth revisiting

Small, understood, and each with a cost that has not yet justified fixing.

- **A table larger than `chunk_size` is split, and the second piece loses its
  header row.** Fixing it means table-aware chunking — treating a table as an
  atomic unit and splitting by rows with the header repeated.
- **Poor scans produce poor text.** Pages are flagged with `ocr_extracted`, and
  nothing yet uses the flag. A quality heuristic could drop or down-weight
  them.
- **The reranker truncates long passages for scoring.** The full passage still
  reaches the prompt, so a very long chunk is judged on its opening only.
- **Sentence splitting is a regular expression.** Abbreviations and decimal
  points occasionally produce a boundary in the wrong place. A real tokeniser
  would cost a dependency and a model for a small gain.
- **No per-document chunk-count cap.** A well-formed PDF within the size and
  page limits can still produce an unbounded number of chunk rows. See
  [security.md](security.md#resource-exhaustion).

## Deliberately not planned

Stated so that their absence reads as a decision.

- **Generation inside the retrieval pipeline.** This position was tested by
  building the thing it was about, and it held. `rag.generation` answers from a
  `PromptContext`, but it is a sibling of `retrieval`, not a stage inside it: it
  imports `domain` and `config` and nothing else, and `tests/test_architecture.py`
  enforces that. Retrieval still stops at assembled context and still does not
  know a model exists.
- **A plugin registry or factory layer.** Constructor injection is how
  components are wired, with one deliberate exception:
  `rag.generation.providers` maps a configured name to a model client, because
  which service answers is a deployment decision. It is a dictionary and a
  lookup — not a registry, not discovery.
- **Async.** Ingestion is throughput-bound on model inference, not on waiting.
  Async would add colour to every function signature for no measured gain. The
  chat interface bridges to async in `app/main.py` alone.
