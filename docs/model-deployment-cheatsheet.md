# Model deployment cheat sheet

How local model assets are installed, located, validated, and loaded. Loading
is handled by `src/rag/model_assets.py`, shared by both pipelines.

## Three distinct things

| | What it is | Where it comes from | When |
|---|---|---|---|
| **Python dependencies** | `sentence-transformers`, `torch` (transitive), etc. | `pip` / `conda` from an index | environment setup |
| **Model assets** | weight and tokenizer files | `huggingface_hub.snapshot_download`, once | setup, before first use |
| **Application configuration** | paths, names, dimensions, batch sizes | environment variables (`.env`) | per deployment |

Runtime execution must never download a model weight or any external resource.
A missing model fails loudly; it never falls back to a download.

## Supported model types

| Role | Kind | Required | Default |
|---|---|---|---|
| Embedding | sentence-transformers bi-encoder | yes | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| Re-ranking | cross-encoder | no | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

The same embedding model must be used for ingestion and for queries; otherwise
the vectors occupy different spaces.

## Directory layout

```text
models/
├── embeddings/
│   └── all-MiniLM-L6-v2/
│       ├── config.json
│       ├── config_sentence_transformers.json
│       ├── modules.json
│       ├── sentence_bert_config.json
│       ├── 1_Pooling/config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       ├── vocab.txt
│       └── model.safetensors
└── rerankers/
    └── ms-marco-MiniLM-L-6-v2/
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        ├── vocab.txt
        └── model.safetensors
```

`models/` is gitignored and holds binary assets only, never Python source. The
directory name is arbitrary; configuration points at it.

## Provisioning

The one step that needs the internet, done at setup, not at run time. Weights
are fetched as `safetensors`, never as pickle files.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

# Embedding model. Required.
snapshot_download(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_dir="models/embeddings/all-MiniLM-L6-v2",
    allow_patterns=["*.json", "vocab.txt", "model.safetensors", "1_Pooling/*"],
)

# Re-ranker. Optional but recommended.
snapshot_download(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    local_dir="models/rerankers/ms-marco-MiniLM-L-6-v2",
    allow_patterns=["*.json", "vocab.txt", "model.safetensors"],
)
PY
```

The `allow_patterns` restriction is deliberate: it fetches the safetensors
weights and the JSON and vocabulary files, and nothing else — no `pytorch_model.bin`,
no arbitrary scripts.

## Configuration

| Variable | Required | Meaning |
|---|---|---|
| `RAG_EMBEDDING_MODEL_PATH` | yes | directory holding the embedding model |
| `RAG_EMBEDDING_MODEL_NAME` | yes | identity recorded on every stored vector |
| `RAG_EMBEDDING_DIMENSION` | no (default 384) | must equal the model's output width and the `VECTOR` column width |
| `RAG_RERANKER_MODEL_PATH` | no | cross-encoder directory; unset means the passthrough reranker is used and no reranking model is loaded |
| `RAG_EMBEDDING_BATCH_SIZE` | no (default 32) | chunks encoded per batch; bounds peak memory |
| `RAG_RERANKER_BATCH_SIZE` | no (default 16) | query/passage pairs scored per batch |

A path may be relative to the project root. An unset optional path means the
feature it configures is not used — no default location is searched.

## Loading

`rag/model_assets.py`:

```python
SentenceTransformer(str(model_path), local_files_only=True, trust_remote_code=False)
CrossEncoder(str(model_path), local_files_only=True, trust_remote_code=False)
```

Before either is constructed, the directory must exist. Every route to the
network is closed: `local_files_only=True` prevents a fetch, and
`trust_remote_code=False` refuses to execute code shipped alongside the
weights. A failure logs the exception's class name only and raises
`ModelUnavailableError` naming the configured path.

The embedding dimension is read from the loaded model via
`get_embedding_dimension`, falling back to the older
`get_sentence_embedding_dimension` accessor across `sentence-transformers`
versions.

## Offline operation

Runtime code must not fetch weights or remote code. The mechanism is
`local_files_only=True` plus `trust_remote_code=False` plus the
directory-exists check. The guarantee is verified by running the test suite
with the network disabled:

```bash
HF_HUB_OFFLINE=1 pytest
```

`HF_HUB_OFFLINE` is a check used in development and is **not** set by the
application. Similarly, "safetensors only" is enforced by the provisioning
`allow_patterns` above, not by a load-time inspection of the files.

## Device, memory, versioning

- **Device.** Left to `sentence-transformers` auto-detection. There is no
  `device=` argument, and no CUDA or MPS code, anywhere in `src/`.
- **Memory.** `RAG_EMBEDDING_BATCH_SIZE` and `RAG_RERANKER_BATCH_SIZE` bound
  peak memory during encoding and scoring. `all-MiniLM-L6-v2` and the
  `ms-marco` cross-encoder are small; both run comfortably on CPU.
- **Versioning.** Every `embeddings` row stores `model_name` and
  `model_dimension`, so a change of model is detectable rather than silent.

## Changing the embedding model

A model with a different output width is a schema change and a full re-embed,
because the `VECTOR` column width is fixed:

1. Provision the new model into `models/embeddings/<name>/`.
2. Set `RAG_EMBEDDING_MODEL_PATH`, `RAG_EMBEDDING_MODEL_NAME`,
   `RAG_EMBEDDING_DIMENSION`.
3. Update `EMBEDDING_DIMENSION` in `src/rag/storage/orm.py` to match.
4. Drop and recreate the schema (there is no migration path), then re-ingest.

`scripts/create_schema.py` refuses to run if `RAG_EMBEDDING_DIMENSION` and the
ORM constant disagree, naming which to change.

## Validating availability

```bash
python -c "
from pathlib import Path
from rag.ingestion.embedding.local import LocalSentenceTransformerEmbedder
print(LocalSentenceTransformerEmbedder(Path('models/embeddings/all-MiniLM-L6-v2')).dimension)"
# expect: 384
```

## Failure modes

| Condition | Result |
|---|---|
| Model directory missing | `ModelUnavailableError`, naming the path; no download attempted |
| Directory present but unloadable | `ModelUnavailableError`; exception class name logged |
| `RAG_EMBEDDING_DIMENSION` ≠ ORM constant | `scripts/create_schema.py` exits with an error before creating tables |
| Query/chunk vector width ≠ column width | `ValueError` from the repository before any write |
| `RAG_RERANKER_MODEL_PATH` set but missing | `ModelUnavailableError` — a hard error, not a silent downgrade to passthrough |
| Generation model absent from the service | `GenerationError` (HTTP 404); never pulled automatically |

## Trust assumptions

A model directory is trusted infrastructure configured by an operator. Its path
comes only from configuration, never from a request or a document. Remote code
execution is refused at load time. Weights are provisioned as `safetensors`.
The directory's other contents are trusted; there is no per-file inspection.
