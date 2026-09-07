---
name: Bug report
about: Something does not behave as documented.
title: "[BUG] "
labels: bug
assignees: ""
---

### What happened

The observed behaviour, and what you expected instead. Quote the exact error
message if there is one, but redact any credential or connection string.

### Where

- [ ] `scripts/ingest.py`
- [ ] `scripts/query.py`
- [ ] `scripts/answer.py`
- [ ] `scripts/create_schema.py`
- [ ] the chat interface (`app/`)
- [ ] the library, used directly

### Reproduction

Steps to reproduce. A minimal generated PDF is preferable to a corpus document.
Do not attach anything from `docs/pool/`.

### Environment

- OS and MariaDB version (`SELECT VERSION();`)
- Output of `python -c "import rag; print(rag.__version__)"`
- Relevant `RAG_*` settings, with secrets removed
- Whether integration tests pass: `pytest -m integration`
