### Summary

What changed and why. Link the issue it addresses, if any.

### Scope

- [ ] Ingestion
- [ ] Retrieval
- [ ] Generation
- [ ] Storage
- [ ] Configuration / operations
- [ ] Documentation only

### Checklist

- [ ] `just check-all` passes (ruff, bandit, mypy, pytest)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] New behaviour is covered by a test
- [ ] `HF_HUB_OFFLINE=1 pytest` still passes if model loading or dependencies changed
- [ ] `scripts/create_schema.py` run, or drift reported, if the ORM models changed
- [ ] Documentation updated if a contract, command, setting, or security property changed

### Architecture

If this touches module boundaries: confirm `tests/test_architecture.py` still
passes, and that ingestion and retrieval still do not import each other.
