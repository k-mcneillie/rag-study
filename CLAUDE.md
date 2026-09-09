# CLAUDE.md

Working notes for Claude Code on this repository. Facts and constraints, not a
procedure — where something is a hard rule it says so, and everything else is
context to reason from.

## What this is

A modular, offline-capable RAG package: PDFs in, ranked and cited context out.
The retrieval pipeline deliberately stops before calling a language model;
`rag.generation` is a separate sibling package that answers from that context.
`service/` is an HTTP API over the query side (FastAPI, an entry point like
`scripts/`), and `app/` is a standalone Chainlit chat client of that API — its
own `src/` project with its own `pyproject.toml` and no `rag` import, so it can
be lifted into another repository. `app-openai/` is a second such client, the
same Chainlit UI written against the OpenAI wire language (`/v1/chat/completions`
streaming) for a hosted RAG provider instead of `service/`; it is independent of
both `rag` and `app/`.

Start with [docs/architecture.md](docs/architecture.md) for how the parts fit
together and why, [docs/ingestion.md](docs/ingestion.md) and
[docs/retrieval.md](docs/retrieval.md) for the pipelines,
[docs/api.md](docs/api.md) for the HTTP API and the chat app, and
[docs/future-work.md](docs/future-work.md) for what is deliberately absent.

## Environment

Everything runs in a conda environment; the tools are not on the base path.

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate rag-study-py3.12
```

```bash
just check-all                 # lint, format, types, tests — the full gate
pytest -m "not integration"    # no database or model weights required
HF_HUB_OFFLINE=1 pytest        # proves nothing reaches the network
```

Integration tests need MariaDB running (`brew services start mariadb`) and the
model weights present. They skip rather than fail when either is missing.

## Hard rules

These are enforced by `tests/test_architecture.py`, which parses the source. If
one seems to be in the way, the design is wrong, not the rule.

- **`ingestion` and `retrieval` must never import each other.** They meet only
  at the storage contract. When both need something, it becomes a leaf module
  (`domain`, `config`, `model_assets`) — as happened with model loading.
- **Only `storage` may import SQLAlchemy.** Processing components depend on the
  `Repository` protocol.
- **`domain` imports nothing third-party.** The contracts stay framework-free.
- **`src/rag` never imports an entry point or a web framework.** No module
  under the package may import `service/`, `app/` (`rag_chat`), `app-openai/`
  (`rag_chat_openai`), `fastapi`, `starlette`, or `chainlit`. The package must
  not depend on what depends on it — this is what keeps the apps liftable and
  `service/` deletable.

Three further rules are not automated but matter as much:

- **Nothing is downloaded at run time.** Model weights are local assets, fetched
  at setup. A missing model fails loudly; it never falls back to a download.
- **Provenance is never taken from document content.** Identifiers, scores,
  page numbers and section paths are pipeline-set. Untrusted, document-derived
  values live only in `metadata`, which nothing reads as a score, an identifier
  or an instruction.
- **`safetensors` only, never pickle**, for any model weights.

## Never commit

- `docs/pool/` — the document corpus. Gitignored, and it stays that way.
  Never `git add -f` it, and never copy its content into tracked files,
  including test fixtures or example output.
- `.env` — real credentials. `.env.example` carries placeholders.
- `models/` — weights.

Before any commit touching `docs/`, verify with `git check-ignore -v`.

## Code standards

Python 3.12+, enforced by ruff and mypy, both configured in `pyproject.toml`:

- **Google-style docstrings** on every module, class and function — pydocstyle
  is on, so this is checked, not merely encouraged.
- Modern typing: `list[str]`, `X | None`. No `typing.List`, no `Optional`.
- `disallow_untyped_defs` is on. Tests included.
- Ruff runs bandit (`S`) and annotation (`ANN`) rules.

Prefer an established, maintained library over hand-written code for a solved
problem — but keep it *behind* one of this project's interfaces, so it stays a
swappable implementation detail. `ftfy` and `langchain_text_splitters` are both
used this way.

## Git workflow

`main` and `dev` are protected. All work happens on a branch off `dev`:

```bash
git checkout dev && git pull
git checkout -b feature/<short-description>
```

Commit regularly. Before pushing: `just check-all` must pass **and**
`CHANGELOG.md` must be updated. Merge into `dev` with `--no-ff`.

## Things that will surprise you

Learned the hard way; each cost real time.

- **`vector` is a reserved word in MariaDB 11.7+**, and SQLAlchemy's MySQL
  dialect does not quote it. The column is named `embedding`; the ORM attribute
  is still `vector`.
- **The application user has no DDL rights** — by design. Schema creation is
  `scripts/create_schema.py`, run with `RAG_ADMIN_DB_URL` set. A
  `CREATE command denied` error is the security control working.
- **Bulk-insert dicts use ORM attribute names, not column names**
  (`vector`, not `embedding`; `extra_metadata`, not `metadata`).
- **`ruff format` may reformat code between edits**, so a multi-line string
  match can silently stop matching. Verify substitutions applied rather than
  assuming.
- **Adding a column to the models does not add it to an existing database.**
  There are no migrations yet. Run `scripts/create_schema.py`, which reports
  drift and prints the `ALTER` needed; otherwise the first symptom is an
  "unknown column" error during retrieval.
- **SQLAlchemy's MySQL reflection cannot parse `VECTOR` columns** and raises
  while trying. Read column metadata from `information_schema` instead.
- **`pymupdf4llm` invokes Tesseract** for pages with no text layer. It is a
  system binary, not a Python dependency; without it, scanned pages yield
  nothing rather than failing.

## How to approach work here

The priority order, when a decision is contested: simplicity, modularity,
independence, maintainability, correctness, testability, extensibility, and
only then performance.

Measure before building. Several features here exist because a real corpus was
examined first, and at least one plausible-sounding problem was measured at
zero occurrences and deliberately left unsolved. When a claim can be checked
against `docs/pool/`, check it rather than reasoning about it.

Say what is uncertain. The reranker's value on academic prose is genuinely
unproven, and the documentation says so. That is preferable to a confident
claim that would not survive measurement.
