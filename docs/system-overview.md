# System overview

A complete account of what this package is, how its parts fit together, what
holds it correct, and where it is weak. For step-by-step instructions see the
[README](../README.md); for the reasoning behind individual design decisions
see [architecture.md](architecture.md).

---

## 1. What it does

Documents go in; ranked, cited context comes out.

```text
   PDFs                                              a question
     │                                                    │
     ▼                                                    ▼
  extract ──► clean ──► chunk ──► embed ──┐        embed query
                                          │             │
                                          ▼             ▼
                                       MariaDB ◄── vector search
                                                        │
                                                        ▼
                                                    rerank
                                                        │
                                                        ▼
                                            context + provenance
                                                        ╎
                                            ╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌ the contract
                                                        ▼
                                                    generation
                                                        │
                                                        ▼
                                             answer, cited, rated
```

**The retrieval pipeline stops at assembled context and calls no language
model.** That has not changed. What sits below the dashed line is
`rag.generation`, a sibling package that consumes the `PromptContext` contract
from outside: retrieval cannot import it, does not know it exists, and is still
tested without it. The separation is what keeps the boundary between untrusted
document text and application instructions in one auditable place — the prompt
is built once, by the augmenter, and sent verbatim.

A chat interface over that pipeline lives in `app/`, outside the package
entirely. See [interface.md](interface.md).

**Everything runs offline.** Model weights are local files, fetched once at
setup. No component reaches the network at run time — verified by running the
whole suite with `HF_HUB_OFFLINE=1`.

---

## 2. Shape of the system

Roughly 4,200 lines of source across four packages plus three leaf modules.

```text
src/rag/
├── config.py          settings from the environment; plain data, no frameworks
├── model_assets.py    the one audited way to load a local model
├── feedback.py        human ratings, appended to a JSON Lines file
├── domain/            the contracts every component speaks
├── storage/           the only place SQLAlchemy appears
├── ingestion/         documents → stored, embedded chunks
├── retrieval/         a query → context with provenance
└── generation/        context → an answer, from a swappable service

app/                   the Chainlit chat interface, outside the package
scripts/               ingest, query, answer: one entry point each
```

### The dependency rule

```text
     config      domain      model_assets     leaves: no internal dependencies
                    ▲
                 storage                      depends on domain, config
                    ▲
        ┌───────────┴───────────┐
    ingestion                retrieval        never each other
                                 ╎
                             generation       domain and config only;
                                              never any pipeline
```

**Ingestion and retrieval never import one another.** They meet only at the
storage contract: one writes through it, the other reads through it, and
neither knows the other exists. Delete either directory and the other still
works.

This is not a convention anyone has to remember. `tests/test_architecture.py`
parses the source and fails the build if a pipeline imports the other, if the
domain gains a third-party dependency, or if anything outside `storage`
imports SQLAlchemy. It has already caught two real violations — `storage`
reaching into `config`, and `config` importing SQLAlchemy — both of which were
fixed by moving code rather than by adding an exception.

---

## 3. The contracts

Eight frozen dataclasses in `domain/models.py`, with no third-party imports at
all. Every component speaks these and nothing else.

```text
Document ──► Page ──► Chunk ──► EmbeddedChunk ──► RankedChunk ──► RerankedChunk ──► PromptContext
```

| Contract | Carries |
|---|---|
| `Document` | generated id, source filename, content hash, metadata |
| `Page` | document id, page number, text, whether OCR produced it |
| `Chunk` | text, page, section path, heading trail, OCR provenance |
| `EmbeddedChunk` | the chunk, its vector, the model that produced it |
| `RankedChunk` | the chunk, relevance score, rank |
| `RerankedChunk` | the ranked chunk, rerank score — **both are kept** |
| `PromptContext` | the passages, the rendered text, the prompt's identity |

### Content versus control

The single most important property. Every model is frozen, and its fields
split in two:

- **Provenance and control** — identifiers, page numbers, section paths,
  scores, model identity. Written once, by the component responsible.
- **`metadata`** — untrusted, document-derived. Never read as an instruction,
  an identifier, or a score.

A document therefore cannot inflate its own ranking, claim to be a different
source, or forge a citation, because nothing that reads a score or an
identifier ever reads it from document-supplied data. This is a structural
property, not a runtime check.

`ocr_extracted` was moved out of `metadata` and onto `Chunk` as a real field
during review, for exactly this reason: the pipeline sets it, not the
document, so it belongs with the provenance.

---

## 4. Ingestion

`PDF → extract → clean → chunk → embed → store`

### Extraction

Layout analysis is delegated to `pymupdf4llm`, whose models ship inside the
installed package, so two-column papers, journal templates and tables come
back in reading order with nothing fetched. Tables arrive as Markdown tables;
mathematics is preserved as readable text rather than guessed back into LaTeX,
because PDFs record an equation's glyphs, not its structure.

Every PDF is untrusted input: validated by file signature rather than
extension, bounded by size and page count, and a malformed document fails on
its own without ending a run.

### Cleaning

Encoding repair is `ftfy`. Page furniture removal uses three rules, because no
single rule covers what real papers contain:

| Rule | Catches |
|---|---|
| Structural shape | Bare page numbers, folios, publisher stamps |
| Repetition with digit masking | Running heads carrying a page number |
| Known artefacts | Figure-text blocks, publisher markers |

Every contaminant handled here was **measured in a real corpus first**.
Mid-sentence line breaks, the classic PDF complaint, were measured at zero
occurrences — the extractor already rejoins wrapped lines — so no code was
written for them.

Repeated sentences inside a page body are deliberately kept: scientific
writing restates definitions legitimately, and only page edges are filtered.

### Chunking

| Chunker | Cuts at | Cost |
|---|---|---|
| `MarkdownHeaderChunker` | Headings, tracking the trail across page breaks | Free |
| `RecursiveChunker` | A character budget, backing off through paragraphs | Free |
| `SemanticChunker` | Where sentence-to-sentence similarity drops | An embedding pass per sentence |

The default composes the first two: structure first, size second, so every
chunk knows its section. Semantic chunking is opt-in because of its cost.

`SemanticChunker` depends on `TextEmbedder`, a two-method description of the
capability it needs, rather than on an embedder. It never imports a model or a
model-loading library, and it is tested with a handful of fixed vectors. That
is how semantic chunking coexists with the rule that a chunker knows nothing
about embeddings.

Contamination is then removed: reference lists (matched on the heading trail,
dropped by default), and fragments too short to be worth retrieving.

### Embedding and storage

`all-MiniLM-L6-v2`, loaded from disk with remote code refused and network
access closed off. Chunks are embedded in batches.

**A document is written in one transaction.** This was a defect found during
review: writing the document, pages and chunks separately meant a failure part
way through left a document row carrying its content hash but no chunks. That
document was invisible to retrieval, and its hash made every retry look like a
duplicate to skip — lost permanently and silently. It is now one atomic unit
of work, with a regression test that reproduces the original scenario.

Rows are bulk-inserted. A 272-chunk paper stores in 8.2 seconds end to end.

---

## 5. Retrieval

`query → embed → rank → rerank → assemble`

### Ranking

Cosine similarity runs **inside MariaDB**, against a native `VECTOR(384)`
column with a vector index, so only the top-k rows return to the application
rather than every embedding in the store. The repository returns scored
candidates without saying how it scored them, so moving the calculation into
Python later would change nothing above it.

### Reranking

The initial ranker compares a query vector against passage vectors computed
without ever seeing the query. That makes it cheap enough for a whole corpus,
and also limits it: a passage on the right subject scores well whether or not
it answers the question.

`CrossEncoderReranker` reads query and passage together. It runs over the
shortlist, never the store — a cheap wide filter followed by an expensive
narrow one, with the orchestrator retrieving more candidates than it returns
so the reranker has room to work.

**Measured honestly:** it reliably removes the clearest failure of vector-only
ranking, and on some academic queries it ranks a genuinely answering passage
lower than vector similarity did. The model is trained on web-search
relevance, not academic prose. Settling this needs the evaluation harness in
[future-work.md](future-work.md), not more tuning.

Both scores survive on every result, so any change in ordering can be traced
to the stage that caused it.

### Prompt assembly

Templates are files named `<name>_<version>.txt`, resolved by identifier
against the package's own directory — never by a caller-supplied path. The
version is configuration, so a past result can be reproduced by restoring the
configuration that produced it.

Retrieved text is placed inside marked source material that the template
describes as untrusted data to be quoted, never obeyed. **This is structural,
not a keyword filter**: a filter can be rephrased around, a boundary cannot.
Marker forgeries in retrieved text *or in the query* are neutralised, so
nothing can appear to close the quoted region and continue as application
instructions. Citations render from the pipeline's provenance, so a document
claiming to be another source cannot forge one.

---

## 6. Security posture

| Concern | How it is handled |
|---|---|
| SQL injection | SQLAlchemy bound parameters throughout; no string-built SQL anywhere |
| Least privilege | Application user holds DML only — it cannot alter its own schema |
| Path traversal | Generated identifiers; filenames are display data, never paths |
| Prompt injection | Structural instruction/data boundary, with forgery neutralised |
| Retrieval poisoning | Scores and provenance are pipeline-set and unforgeable |
| Resource exhaustion | Bounded document size, page count, chunk count, `top_k`, batch sizes |
| Unsafe deserialisation | `safetensors` only; `json.loads`, never `pickle` or `eval` |
| Supply chain | Minimal pinned dependencies; nothing downloaded at run time |
| Credential leakage | Passwords excluded from `repr`; errors name variables, never values |
| Query scope | Allow-listed filters; arbitrary fields rejected |

Each has a test that exercises the attack, not merely the happy path: an SQL
payload round-trips as data, an injection string in a filter matches nothing,
a document attempting to close the prompt boundary fails to.

**This is not a claim that the system is secure.** The mitigations are in
place and tested, but no adversarial review by anyone other than its author
has taken place. Treat that as open.

---

## 7. What holds it correct

**205 tests, 98.4% coverage**, with ruff and mypy clean under strict settings.

| Layer | What it proves |
|---|---|
| Unit tests | Each component in isolation, with fakes |
| Architecture tests | The dependency rules, by parsing the source |
| Integration tests | Real MariaDB, real models, real vector search |
| Full-cycle test | A PDF ingested by one pipeline, retrieved by the other |

Integration tests skip when the database or weights are absent, so the suite
runs anywhere; CI deselects them explicitly rather than letting them skip
unnoticed, so a real failure cannot hide behind a skip.

Two tests are worth singling out. `test_architecture.py` makes the central
architectural claim executable rather than aspirational. And
`test_a_failed_write_leaves_nothing_behind` reproduces the exact data-loss
scenario found during review, so it cannot return.

**Verified working end to end:** 4 documents, 134 pages, 487 chunks and 487
embeddings stored, with a byte-identical duplicate correctly skipped, and
queries returning relevant, cited passages.

---

## 8. Strengths

1. **The pipeline separation is real and enforced.** Not documented and
   trusted — parsed and failed on. It has already caught violations.
2. **Provenance cannot be forged.** Content and control are separated
   structurally, so a malicious document cannot influence its own ranking or
   claim a different source.
3. **The offline guarantee holds under test.** The full suite passes with the
   network disabled, including model loading.
4. **Cleaning is evidence-based.** Every contaminant was measured before code
   was written, and one plausible-sounding problem was measured at zero and
   deliberately not solved.
5. **Every stage is genuinely replaceable.** Each is one class behind one
   interface, injected through a constructor. No factories, no registry, no
   framework.
6. **Failures are contained.** One bad document cannot end a run; one bad
   write cannot leave the store inconsistent.

## 9. Weaknesses

Stated plainly, because a review that finds nothing is not a review:

1. **Retrieval quality is unmeasured.** There is no evaluation set, so
   "better" is currently a matter of reading results. This is the single
   biggest gap, and it blocks judging the reranker.
2. **The reranker's value is unproven on this corpus.** It removes obvious
   noise; whether it improves ordering among relevant passages is unknown.
3. **Retrieval is purely dense.** No keyword or hybrid search, so exact
   identifiers, symbols and rare terms retrieve less reliably than prose.
4. **No schema migrations.** Changing the schema on a populated database has
   no supported path yet.
5. **Poor scans produce poor text.** Flagged via `ocr_extracted`, but nothing
   yet acts on the flag.
6. **A table larger than `chunk_size` loses its header row** on the second
   piece.
7. **Single-node assumptions.** No connection-pool tuning, no batching across
   documents, no concurrency story. Fine at this scale; unexamined beyond it.

---

## 10. Where to look

| For | See |
|---|---|
| Running it | [README](../README.md) |
| Why it is built this way | [architecture.md](architecture.md) |
| What is deliberately missing | [future-work.md](future-work.md) |
| How to contribute | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| What changed and when | [CHANGELOG.md](../CHANGELOG.md) |
