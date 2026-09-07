---
name: Feature or enhancement
about: Propose a new capability or a change to an existing one.
title: "[FEATURE] "
labels: enhancement
assignees: ""
---

### Problem

What is missing or awkward today? If it relates to a documented gap, link the
relevant section of `docs/future-work.md`.

### Proposed change

What you want the package to do, and roughly how it fits the architecture. If it
is a new pipeline stage, name the interface it would implement (`BaseExtractor`,
`BaseChunker`, `BaseEmbedder`, `BaseRanker`, `BaseReranker`,
`BasePromptAugmenter`, `BaseChatModel`, `Repository`).

### Scope

- [ ] Ingestion
- [ ] Retrieval
- [ ] Generation
- [ ] Storage
- [ ] Configuration / operations
- [ ] Documentation

### Constraints to preserve

Note any that apply: the ingestion/retrieval independence, offline operation
(nothing downloaded at run time), provenance set only by the pipeline, or the
least-privilege database boundary.
