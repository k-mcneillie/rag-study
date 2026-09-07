# Contributing / Git Workflow

This repository uses a two-tier branch strategy:

- **`main`** — protected. Reflects the last known-good state of the project.
  Never receives direct commits; only updated by merging `dev` in via a
  reviewed pull request.
- **`dev`** — protected. The primary working branch that all feature/issue
  branches are cut from and merged back into. Never receives direct commits
  either — all work happens on a branch off `dev`.

## Workflow

1. **Branch from `dev`** for every issue, ticket, or piece of implementation
   work:

   ```bash
   git checkout dev
   git pull
   git checkout -b <type>/<short-description>   # e.g. feature/pdf-extractor, fix/chunk-overlap, docs/phase-1-architecture
   ```

2. **Commit regularly** on that branch with clear, focused commit messages.
   Small, frequent commits are preferred over large, infrequent ones.

3. **Before pushing / opening a pull request into `dev`, complete full QA**:

   ```bash
   just check-all   # lint + format-check + type-check + test
   ```

   All checks must pass. Do not push work that fails `just check-all`.

4. **Update `CHANGELOG.md`** under the `[Unreleased]` section as part of the
   same branch/PR, describing what changed and why. A PR that changes
   behavior without a changelog entry is incomplete.

5. **Open a pull request into `dev`.** Once merged, delete the feature
   branch.

6. **Releases**: periodically, once `dev` is stable and QA'd as a whole, open
   a pull request from `dev` into `main`, move the relevant `[Unreleased]`
   entries in `CHANGELOG.md` under a new dated version heading, and tag the
   release after merging.

## Branch protection

`main` and `dev` should both be configured on GitHub (Settings → Branches)
to:

- require a pull request before merging (no direct pushes),
- require the CI status checks (`.github/workflows/ci.yml`) to pass before
  merging,
- disallow force pushes and branch deletion.

This can be applied with the GitHub CLI once it is installed and
authenticated (`gh auth login`), for example:

```bash
gh api -X PUT repos/k-mcneillie/rag-study/branches/main/protection \
  -F required_status_checks=null \
  -F enforce_admins=true \
  -F required_pull_request_reviews[required_approving_review_count]=0 \
  -F restrictions=null

gh api -X PUT repos/k-mcneillie/rag-study/branches/dev/protection \
  -F required_status_checks=null \
  -F enforce_admins=true \
  -F required_pull_request_reviews[required_approving_review_count]=0 \
  -F restrictions=null
```

Adjust `required_approving_review_count` and `required_status_checks` to match
the checks in `.github/workflows/ci.yml` and the review policy in force.
