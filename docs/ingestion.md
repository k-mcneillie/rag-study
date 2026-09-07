# Ingestion

Ingestion turns source documents into embedded, persisted chunks:

```text
Source → Extraction → Chunking → Embedding → Storage
```

The pipeline is coordinated by `IngestionOrchestrator` in
`rag/ingestion/orchestrator.py`, which holds workflow only: the order of the
stages, what happens when a document fails, and when a document can be skipped.
Each stage is injected through the constructor. The orchestrator contains no
extraction, chunking, or embedding logic and never names a file format or a
model.

`scripts/ingest.py` assembles the pipeline from configuration and runs it over
a directory of PDFs.

```bash
python scripts/ingest.py path/to/pdfs        # skips content already stored
python scripts/ingest.py path/to/pdfs --reingest
```

Discovery is `sorted(directory.glob("*.pdf"))`: non-recursive, and lowercase
`.pdf` only.

## Supported input

Only PDF is supported. `DocxExtractor` and similar are deliberately deferred;
each would be one class implementing `BaseExtractor` with no change downstream
(see [future-work.md](future-work.md) §2).

## Extraction — `PDFExtractor`

`rag/ingestion/extraction/pdf.py`.

- **Responsibility.** Read a PDF, recover its pages in reading order, clean each
  page, and record the document's identity and metadata.
- **Input.** A filesystem `Path`.
- **Output.** An `ExtractedDocument`: a `Document` record and a tuple of
  `Page` objects.
- **Dependencies.** `pymupdf` for the document object and text-layer inspection;
  `pymupdf4llm` for layout-aware Markdown extraction. Both ship their layout and
  table models inside the installed package, so extraction fetches nothing.
- **Configuration.** `max_bytes` (`RAG_MAX_DOCUMENT_BYTES`, default 100 MiB),
  `max_pages` (`RAG_MAX_DOCUMENT_PAGES`, default 2000), `use_ocr` (default on).
- **Side effects.** None. The file is read, never written or moved.
- **Does not control.** Chunking, embedding, storage. It does not resolve or
  validate the directory it is handed; the caller supplies paths.

### Validation

Every PDF is untrusted input. Before the file is opened:

- it must be a readable regular file;
- its size must be within `max_bytes` and greater than zero;
- its first bytes must be the signature `%PDF-`. The extension is not trusted,
  because it is attacker-controlled and says nothing about the bytes on disk.

After opening, a document whose page count exceeds `max_pages` is rejected. A
file that fails to parse raises `ExtractionError` with a one-line summary; the
underlying exception's class name is logged, and nothing else. One malformed
document does not end a run.

### Page boundaries and OCR

Extraction is per page, so every chunk can be traced to a page number.
`pymupdf4llm.to_markdown(..., page_chunks=True)` returns one Markdown block per
page, with tables rendered as Markdown tables and mathematics preserved as
readable text rather than reconstructed as LaTeX.

Whether a page carries its own text layer is decided before extraction: a page
whose raw text is shorter than 32 characters is treated as having none, and any
text later recovered from it is marked `ocr_extracted=True` on the `Page`. OCR
of such pages requires the Tesseract binary, which is a system dependency, not
a Python one. When it is absent, scanned pages yield no text rather than
failing.

### Metadata

Only `title`, `author`, `subject`, and `creationDate` are read from the PDF's
metadata dictionary, coerced to strings, and stored in `Document.metadata`.
Everything else is discarded. These values are untrusted document content and
are stored as data only.

The content hash is a streamed SHA-256 of the file's bytes, read in 1 MiB
blocks so a large document is never held in memory in full. It is used for
duplicate detection, never as a path.

## Cleaning

`rag/ingestion/extraction/cleaning.py`. Applied per page during extraction,
except boilerplate detection, which needs every page at once.

- **Encoding repair** is delegated to `ftfy`, with NFKC normalisation, so
  ligatures and full-width forms embed identically to their ordinary
  equivalents. Line de-hyphenation and paragraph rejoining are already done by
  the layout-aware extractor and are not repeated.
- **Figure text** recovered from inside charts — axis labels, legend entries,
  tick values — arrives as disconnected fragments delimited by the extractor's
  own markers. The whole block is discarded: it matches numeric queries without
  answering anything.
- **Page furniture** — running heads, footers, page numbers, publisher stamps —
  is removed by three rules acting only on the three lines at each edge of a
  page:

  | Rule | Catches |
  |---|---|
  | Structural shape | a line retaining almost no letters once digits and punctuation are stripped (bare page numbers, folios, `Vol.:(0123456789)`) |
  | Digit-masked repetition | a line whose form, with digit runs masked, recurs at the edges of most pages (running heads carrying a page number) |
  | Known artefacts | a short explicit list of publisher and extractor markers |

Headings and table rows are exempt from every furniture rule. Repeated
sentences inside a page body are kept: scientific writing restates definitions
legitimately, and only page edges are filtered. Furniture removal never empties
a page; a page short enough for every line to look like furniture is more
likely a sparse title or figure page, and its text is kept.

Every contaminant handled here was measured in the project's own corpus before
code was written for it. Mid-sentence line breaks, a common PDF complaint, were
measured at zero occurrences and are not handled.

## Chunking — `ChunkingPipeline`

`rag/ingestion/chunking/pipeline.py`. The default `BaseChunker`.

- **Responsibility.** Divide pages into retrievable chunks, each carrying its
  page number and section path, and drop chunks that would only add noise.
- **Input.** A sequence of `Page`.
- **Output.** A list of `Chunk`.
- **Dependencies.** `MarkdownHeaderChunker` and `RecursiveChunker`, both thin
  adapters over `langchain_text_splitters`; optionally a `SemanticChunker`.
- **Configuration.** `chunk_size` (`RAG_CHUNK_SIZE`, default 1200 characters),
  `chunk_overlap` (`RAG_CHUNK_OVERLAP`, default 150), `min_chunk_chars`
  (default 40, not environment-configurable), `drop_references` (default true).
- **Side effects.** None.
- **Does not control.** Embedding, storage, ranking.

The pipeline runs in order:

1. **Structure.** `MarkdownHeaderChunker` splits at Markdown headings (levels
   one to six), recording the heading trail. The trail in force at the end of a
   page carries onto the next, so a section spanning a page break keeps its
   provenance. Text before the first heading becomes its own chunk, so
   abstracts and front matter are not discarded.
2. **Size.** Any section over `chunk_size` is split by `RecursiveChunker`,
   which backs off through the separators `["\n\n", "\n", ". ", " ", ""]` —
   paragraph breaks first, so a Markdown table is only broken into when a
   single table exceeds the budget on its own. Each piece keeps the original's
   page and section.
3. **Contamination filters.** A chunk whose heading trail names a reference list
   (`references`, `bibliography`, `works cited`, `literature cited`) is tagged
   and, by default, dropped; `drop_references=False` keeps it. A chunk shorter
   than `min_chunk_chars` is dropped as a fragment.

Both filters run after chunking rather than during cleaning, because both need
a chunk's section, which does not exist until the document is split.

### Semantic chunking — `SemanticChunker`

`rag/ingestion/chunking/semantic.py`. Opt-in, enabled by
`RAG_SEMANTIC_CHUNKING=true`.

When enabled, `ChunkingPipeline` divides oversized sections at points where the
subject changes rather than at a character count, with `RecursiveChunker`
running afterwards as a size backstop. The chunker splits a section into
sentences (a regular-expression boundary; table rows and headings are kept
whole), embeds them, and places a boundary where the distance between
consecutive sentences exceeds a percentile of the distances observed in that
section. Defaults: `breakpoint_percentile` 85, `max_chunk_chars` 1200,
`min_chunk_chars` 100. None of these is environment-configurable.

`SemanticChunker` depends on `TextEmbedder`, a protocol with a single
`embed_texts` method. It imports no model and no model-loading library, and is
tested with fixed vectors. `scripts/ingest.py` passes it the same
`LocalSentenceTransformerEmbedder` instance already loaded for the pipeline's
own embedding step, so enabling it costs no second model load — but it does
embed every sentence, on top of embedding every finished chunk, which is why it
is not the default.

## Embedding — `LocalSentenceTransformerEmbedder`

`rag/ingestion/embedding/local.py`.

- **Responsibility.** Turn chunks into vectors.
- **Input.** A sequence of `Chunk`.
- **Output.** A list of `EmbeddedChunk`, one per input chunk, in order.
- **Dependencies.** `rag.model_assets.load_sentence_transformer`, which loads
  the model from a local directory with network access closed and remote code
  refused.
- **Configuration.** `RAG_EMBEDDING_MODEL_PATH` (required),
  `RAG_EMBEDDING_MODEL_NAME` (required, recorded on every vector),
  `embedding_batch_size` (`RAG_EMBEDDING_BATCH_SIZE`, default 32).
- **Side effects.** Loads a model into memory at construction.
- **Does not control.** Which chunks it is given, or where the vectors are
  stored.

Vectors are L2-normalised at encoding time. The default model is
`all-MiniLM-L6-v2` (384 dimensions). Provisioning is covered in
[model-deployment-cheatsheet.md](model-deployment-cheatsheet.md).

## Storage

The orchestrator calls `Repository.save_ingested_document(document, pages,
embedded_chunks)`. The MariaDB implementation:

- validates that every vector's width matches the configured embedding
  dimension **before** opening a transaction, so a bad batch never partially
  writes;
- opens one transaction, deletes any existing copy of the document — matched by
  identifier **or** by content hash, so re-ingesting a renamed file replaces it
  rather than duplicating it — and bulk-inserts the document, its pages, its
  chunks, and its embeddings;
- commits, so the store holds either the whole document or none of it.

Bulk insertion replaces a per-row round trip; a paper of a few hundred chunks
would otherwise spend most of its ingestion time waiting on the database.

## Failure behaviour

`IngestionOrchestrator.ingest` returns an `IngestionResult` per document rather
than raising:

| Field | Meaning |
|---|---|
| `source` | the document's path |
| `document_id` | assigned id, or `None` if skipped or failed |
| `chunk_count` | chunks stored |
| `skipped` | the content was already present and left alone |
| `error` | a safe description of the failure, or `None` |

`ingest_all` returns one result per source, in order, and continues past any
failure. `scripts/ingest.py` exits non-zero if any document failed or if
configuration or a model was missing at startup.

A document that yields no chunks — a scanned PDF with no text layer and no
Tesseract, for instance — is reported as a failure, not stored as an empty
document.

## Resource limits

| Limit | Default | Environment variable | Enforced by |
|---|---|---|---|
| Document size | 100 MiB | `RAG_MAX_DOCUMENT_BYTES` | `PDFExtractor` (before open) |
| Page count | 2000 | `RAG_MAX_DOCUMENT_PAGES` | `PDFExtractor` (after open) |
| Chunk size | 1200 chars | `RAG_CHUNK_SIZE` | `RecursiveChunker` |
| Chunk overlap | 150 chars | `RAG_CHUNK_OVERLAP` | `RecursiveChunker` (must be `< chunk_size`) |
| Minimum chunk | 40 chars | — | `ChunkingPipeline` |
| Embedding batch | 32 | `RAG_EMBEDDING_BATCH_SIZE` | `LocalSentenceTransformerEmbedder` |

There is no cap on the number of chunks a single document may produce, on total
extracted text length, or on the number of documents in one batch. See
[security.md](security.md) for the full treatment of ingestion as an attack
surface.
