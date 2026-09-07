# Retrieval

Retrieval turns a query into ranked context with provenance:

```text
Query → Ranking → Re-ranking → Prompt Augmentation → Context + Metadata
```

The pipeline is coordinated by `RetrievalOrchestrator` in
`rag/retrieval/orchestrator.py`, which holds workflow only: it decides the order
of the stages and how many candidates each is given. It contains no similarity
calculation, no database query, no reranking rule, and no prompt text. Every
stage is injected through the constructor.

`RetrievalOrchestrator.retrieve` returns a `PromptContext` and calls no
language model. The pipeline terminates at assembled context. Answering is
`rag.generation`, a separate package; see [generation.md](generation.md).

`scripts/query.py` runs the pipeline and prints the assembled context with its
provenance:

```bash
python scripts/query.py "How does DPO avoid training a reward model?"
python scripts/query.py "..." --top-k 3 --document-id <id> --show-prompt
```

`rag.assembly.build_retrieval_orchestrator` composes the pipeline from
configuration, and is shared by `scripts/query.py`, `scripts/answer.py`, and
`app/main.py`.

## Query embedding — `LocalSentenceTransformerQueryEmbedder`

`rag/retrieval/embedding/local.py`.

- **Responsibility.** Encode one query string into one vector.
- **Input.** A query `str`. An empty or whitespace-only query is rejected.
- **Output.** A `tuple[float, ...]`, L2-normalised.
- **Dependencies.** `rag.model_assets.load_sentence_transformer`.
- **Configuration.** `RAG_EMBEDDING_MODEL_PATH`.
- **Does not control.** Storage, ranking, reranking, prompting.

The query must be embedded by the same model that embedded the chunks, or the
vectors occupy different spaces and their similarity is meaningless. The
configured model path is what keeps them in step, and a dimension check turns a
mismatch into an error rather than silently poor results.

`BaseQueryEmbedder` is declared in the retrieval package rather than reused
from ingestion. Retrieval must not import ingestion, and the two need different
shapes: ingestion embeds many chunks in batches, retrieval embeds one query.
The shared part — loading the model safely — lives in `rag.model_assets`.

## Ranking — `CosineSimilarityRanker`

`rag/retrieval/ranking/cosine.py`.

- **Responsibility.** Produce an initial relevance ordering of candidate
  chunks.
- **Input.** The query vector, a candidate count, optional `filters`.
- **Output.** A list of `RankedChunk`, most relevant first.
- **Dependencies.** A `Repository`, supplied through the storage contract. The
  ranker never touches a session or a table.
- **Does not control.** Extraction, chunking, embedding generation, reranking,
  prompting.

The class is a one-line delegate to `Repository.similarity_search`. In the
MariaDB implementation the comparison runs in the database:
`VEC_DISTANCE_COSINE` is evaluated against the native `VECTOR(384)` column and
its cosine vector index, ordered, and limited, so only the top-k rows return to
the application rather than every embedding in the store. The repository
returns `RankedChunk` with the score already attached — `score = 1.0 -
distance`, since `VEC_DISTANCE_COSINE` returns a distance where smaller is
closer while the contract expects higher to be better — so moving the
calculation into Python later would change nothing above the repository.

`filters` keys are checked against `ALLOWED_SEARCH_FILTERS`, which contains
only `document_id`. An unknown key is rejected. The value is bound as a
parameter. This restricts a query to one document; it is not an access-control
mechanism, and no access rules are applied. See
[security.md](security.md#data-isolation).

## Candidate count

`RetrievalOrchestrator` asks the ranker for more candidates than it will
return: `candidate_count = min(top_k * 4, max_top_k)`. A reranker can only
reorder what it is given, so it is given a wider pool; without this it could
only confirm the ranker's order. `top_k` itself must be between 1 and
`max_top_k` (`RAG_MAX_TOP_K`, default 100).

## Re-ranking

The initial ranker compares a query vector against passage vectors that were
computed without ever seeing the query. That makes it cheap enough for a whole
corpus and also limits it: a passage on the right subject scores well whether
or not it answers the question.

### `CrossEncoderReranker`

`rag/retrieval/reranking/cross_encoder.py`. Used when `RAG_RERANKER_MODEL_PATH`
is set.

- **Responsibility.** Rescore each candidate by reading the query and the
  passage together, then reorder and narrow to `top_k`.
- **Input.** The query, a sequence of `RankedChunk`, `top_k`.
- **Output.** A list of `RerankedChunk`, most relevant first.
- **Dependencies.** `rag.model_assets.load_cross_encoder`.
- **Configuration.** `reranker_batch_size` (`RAG_RERANKER_BATCH_SIZE`, default
  16), `max_text_chars` (default 2000, not environment-configurable).
- **Does not control.** It runs only over the shortlist the ranker returned; it
  never reaches back to the store.

A passage longer than `max_text_chars` is truncated for scoring only. The full
passage still reaches the prompt. Cross-encoder scores are unbounded logits,
not similarities in `[0, 1]`, and are not comparable with the ranker's cosine
scores; both are kept on every result (`rerank_score` and `initial_score`), so
a change in ordering can be traced to the stage that caused it.

Measured against the project's corpus, the cross-encoder removes the clearest
failure of vector-only ranking — an "Author Contributions" section appearing
above a paper's method for a question about method — but its ordering of
genuinely relevant academic passages is not consistently better. The model is
trained on web-search relevance. Settling this needs the evaluation harness in
[future-work.md](future-work.md) §1.

### `PassthroughReranker`

`rag/retrieval/reranking/passthrough.py`. The default when no reranker model is
configured. It keeps the ranker's ordering, carries the initial score forward
as the rerank score, and narrows to `top_k`. It is a real stage rather than an
absent one, so swapping in a cross-encoder later changes one constructor
argument and nothing else.

## Prompt augmentation — `TemplatePromptAugmenter`

`rag/retrieval/prompting/augmenter.py`.

- **Responsibility.** Assemble the top-k passages and their provenance into a
  rendered prompt, applying a versioned template, and record which template
  produced it.
- **Input.** The query, a sequence of `RerankedChunk`.
- **Output.** A `PromptContext`.
- **Dependencies.** None beyond the standard library and the domain contracts.
- **Does not control.** Extraction, chunking, storage, ranking. It never calls
  a language model.

Each passage is rendered as `[n] (document=<id> page=<n> [section=<...>])`
followed by its text. The number and the provenance line are built from
pipeline fields, never from the document, so a passage claiming to be a
different source cannot forge a citation. When no passages were retrieved, the
source material reads `(no relevant source material was found)`.

### The instruction/data boundary

Retrieved text is untrusted: a PDF can contain a sentence addressed to a
language model, and it reaches the augmenter looking like any other passage.
The defence is structural. Retrieved text is placed inside marked source
material that the template's instructions describe as data to be quoted, never
obeyed. No attempt is made to detect malicious wording, because a filter can be
rephrased around and a boundary cannot.

A passage or a query containing a marker-like line could appear to close the
quoted region and continue as application text. `neutralise_markers` replaces
any near-miss of the begin or end marker — matched loosely, on its distinctive
wording rather than exact punctuation — with `[marker removed]`, in every
passage's text, its section path, and the query. The template is rendered with
`string.Template.substitute`, which performs no format-string interpretation.

The template is `retrieval_context_v1.txt`. Its full text and the security
analysis are in [security.md](security.md#prompt-injection).

## Output contract

`RetrievalOrchestrator.retrieve` returns:

```python
@dataclass(frozen=True)
class PromptContext:
    chunks: tuple[RerankedChunk, ...]
    rendered_text: str
    prompt_metadata: PromptMetadata  # prompt_name, prompt_version
```

`rendered_text` is the finished prompt. `chunks` carries each passage with its
full provenance and both scores. `prompt_metadata` identifies the template.
This is the hand-off point to a model; the consumer must treat `rendered_text`
as final and send it unchanged.

## Prompt versioning and reproducibility

Templates are files in `src/rag/retrieval/prompts/`, named
`<name>_<version>.txt` and shipped as package data. A template is resolved by
`(prompt_name, prompt_version)` through `importlib.resources` against that
package's own directory — never by a caller-supplied path. Both identifiers
must match `^[a-z0-9_]+$`; anything else is rejected before a filename is
built, so prompt selection cannot become arbitrary file inclusion.

The version in use is configuration: `RAG_PROMPT_NAME` (default
`retrieval_context`) and `RAG_PROMPT_VERSION` (default `v1`). Every
`PromptContext`, every `Answer`, and every `FeedbackRecord` records the name
and version that produced it. A past result can therefore be reproduced by
restoring the configuration that produced it, and feedback collected before and
after a prompt change stays distinguishable.

Changing a prompt means adding a new file with a new version. Obsolete versions
are left in place as additional files; nothing selects them unless the
configuration names them.
