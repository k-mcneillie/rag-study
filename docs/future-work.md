# Future work

What this package deliberately does not do yet, why, and what building each
thing would involve. Everything here is a decision, not an oversight: the
foundation was built so these could be added without rework, and each entry
names the seam it would attach to.

Ordered by the value it would add.

---

## 1. Evaluation harness

**The highest-value item, and the one that unblocks judging everything else.**

Retrieval quality is currently assessed by reading results. That is enough to
catch something obviously wrong — an "Author Contributions" section ranked
above a paper's method — but not enough to answer whether a change helped.

The chat interface now collects thumbs up and down against real questions,
recording the passages each answer was given
([interface.md](interface.md)). That is not a substitute for this item — a
rating judges an answer, not a ranking, and the sample is whatever happened to
be asked — but it is where the questions for a harness will come from, and it
costs nothing to accumulate in the meantime.
The cross-encoder reranker is the clearest example: it removes obvious noise,
yet on some academic queries it ranks a genuinely answering passage lower than
vector similarity did. Without measurement there is no way to settle that.

**What it would involve.** A set of 30–50 questions over a known corpus, each
paired with the passages that should be retrieved, and a runner reporting
recall@k, precision@k and mean reciprocal rank. No new architecture: it reads
through the existing `Repository` and drives the existing orchestrator.

**Why it is not built.** It needs a corpus whose right answers are agreed, and
that is a judgement about the documents rather than about the code.

**What it would immediately settle:** whether the reranker earns its cost;
whether semantic chunking beats structural chunking; the best `chunk_size`;
whether a larger embedding model is worth the memory.

---

## 2. Additional document types

Only PDFs are supported. `DocxExtractor`, `PowerPointExtractor` and
`LatexExtractor` were scoped out of the original plan deliberately.

**What it would involve.** Each is one class implementing `BaseExtractor`,
returning an `ExtractedDocument`. Nothing downstream changes: chunking,
embedding, storage and retrieval never learn that a new format exists. The
work is per-format extraction and cleaning, not integration.

**Two things to decide first.** Where format detection lives — probably a
small registry mapping a file's signature to an extractor, rather than
extension-based dispatch, which is attacker-controlled. And whether the
`Page` contract still fits: slides and LaTeX sources paginate differently from
PDFs, and forcing them into page numbers may be less useful than a more
general "location" concept.

---

## 3. Access control and multi-tenancy

The system assumes every caller may see every document.

**What it would involve.** The seam exists and is tested: `similarity_search`
takes a `filters` mapping validated against an explicit allow-list, and the
orchestrator passes it through untouched. Adding scope means adding a column
to `documents`, extending the allow-list, and resolving a caller's permitted
scope before the query — not restructuring retrieval.

**What would need care.** Filtering must happen *in the query*, not after
results return, or the ranking silently degrades as filtered results are
discarded. And the prompt augmenter would need to refuse to render a passage
outside the caller's scope, as a second check rather than a first line of
defence.

---

## 4. Schema migrations

`scripts/create_schema.py` creates tables. There is no path to change them
once a database holds data worth keeping.

**What it would involve.** Alembic, with the migration step run by the same
administrative account that owns schema creation, since the application user
holds no DDL rights. The existing schema becomes the initial revision.

**Why it matters sooner than it looks.** A change of embedding model changes
the `VECTOR` column width, which is a migration *and* a re-embedding of every
stored chunk. Better to have the mechanism before that is needed.

---

## 5. Hybrid retrieval

Retrieval is purely dense: everything is found by vector similarity. Dense
retrieval is weak exactly where papers are strongest — exact identifiers,
symbols, statistical notation, rare terms — because an embedding blurs a token
into its neighbourhood.

**What it would involve.** A `BaseRanker` implementation combining MariaDB
full-text search with the existing vector search, merging the two orderings
(reciprocal rank fusion is the usual choice). It slots in as one more ranker;
the orchestrator does not change.

---

## 6. Known limitations worth revisiting

Small, understood, and each with a cost that has not yet justified fixing:

- **A table larger than `chunk_size` is split, and the second piece loses its
  header row.** Fixing it means table-aware chunking — treating a table as an
  atomic unit and splitting by rows with the header repeated.
- **Poor scans produce poor text.** Pages are flagged with `ocr_extracted`, and
  nothing yet uses that flag. A quality heuristic could drop or down-weight
  them; the flag is there so the decision can be made later.
- **The reranker truncates long passages for scoring.** The full passage still
  reaches the prompt, so a very long chunk is judged on its opening only.
- **Sentence splitting is a regular expression.** Abbreviations and decimal
  points occasionally produce a boundary in the wrong place. A real tokeniser
  would cost a dependency and a model for a small gain.

---

## What is deliberately *not* planned

Worth stating, so that absence reads as a decision rather than an omission:

- **Generation inside the retrieval pipeline.** This position has been tested
  by building the thing it was about, and it held. `rag.generation` answers
  from a `PromptContext`, but it is a sibling of `retrieval`, not a stage
  inside it: it imports `domain` and `config` and nothing else, and
  `tests/test_architecture.py` enforces that. Retrieval still stops at
  assembled context and still does not know that a model exists. What changed
  is only that something now stands on the far side of the contract. See
  [interface.md](interface.md).
- **A plugin registry or factory layer.** Constructor injection is still how
  components are wired, with one deliberate exception:
  `rag.generation.providers` maps a configured name to a model client. Which
  model service answers is a deployment decision, so requiring a code edit to
  change it would have made the swappable interface swappable only in
  principle. It is a dictionary and a lookup — not a registry, not discovery,
  and it exists for one kind of component.
- **Async.** Ingestion is throughput-bound on model inference, not on waiting.
  Async would add colour to every function signature for no measured gain.
