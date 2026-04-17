# Contributing to policyengine-us-data

See the [shared PolicyEngine contribution guide](https://github.com/PolicyEngine/.github/blob/main/CONTRIBUTING.md) for cross-repo conventions (towncrier changelog fragments, `uv run`, PR description format, anti-patterns). This file covers policyengine-us-data specifics.

## Commands

```bash
make install                 # install deps (uv)
make format                  # format (required)
make test-unit               # unit tests (synthetic / mocked, seconds)
make test-integration        # integration tests (need built H5 datasets)
make test                    # both
make data                    # full dataset build (long)
make push-pr-branch          # push to upstream with correct tracking (use before opening PRs)
uv run pytest tests/unit/datasets/ -v
```

Python 3.12–3.14. Default branch: `main`.

## Test organisation

- `tests/unit/` — self-contained (synthetic data, mocks, checked-in fixtures). Run in seconds with no external deps.
  - `unit/datasets/` — dataset code
  - `unit/calibration/` — calibration code
- `tests/integration/` — requires built H5 datasets, HuggingFace downloads, `Microsimulation` objects, or DB ETL. Named after the dataset under test (e.g. `test_cps.py` tests `cps_2024.h5`).

**Placement rules:**

- **Never** put tests that need H5 files or `Microsimulation` in `unit/`.
- **Never** put synthetic-only tests in `integration/`.
- Sanity checks (value ranges, population counts) go in the per-dataset integration file, not a separate sanity file.
- When adding an integration test, extend the existing per-dataset file if one exists.

## Updating datasets

If your change is a non-bugfix update to a cloud-hosted dataset (CPS, enhanced CPS, PUF), bump both the filename and URL in the class definition and in `storage/upload_completed_datasets.py`. That lets us store historical dataset versions separately and reproducibly.

## Opening PRs

**Always create branches on the upstream repo, not a fork.** Fork PRs can't access workflow secrets and will fail on data-download steps. The convenience target:

```bash
make push-pr-branch
```

pushes the current branch to `upstream` with the correct tracking so `gh pr create` just works.

## Repo-specific anti-patterns

- **Never fabricate data or results.** This is a research codebase; reproducible aggregates only. Use `[TO BE CALCULATED]` placeholders if a number isn't computed yet.
- **Don't** open PRs from personal forks (CI will fail on secrets).
- **Don't** add `[codex]` or other agent-label prefixes to PR titles.
- **Don't** skip full-build CI when touching the imputation or calibration pipeline.
- **Don't** commit large binary artefacts — HuggingFace storage only.

## CI workflows

Five workflow files in `.github/workflows/`:

- `pr.yaml` — fork check, lint, uv.lock freshness, towncrier fragment check, unit tests, smoke test, docs build. Integration tests trigger when files in `policyengine_us_data/`, `modal_app/`, or `tests/integration/` change. ~2–3 min for the unit path.
- `push.yaml` — on push to main: either version-bump + PyPI publish (on `Update package version` commits), or a full Modal data build with integration tests (on everything else).
- `pipeline.yaml` — dispatch only, spawns the H5 generation pipeline on Modal with configurable GPU/epochs/workers.
- `local_area_publish.yaml` / `local_area_promote.yaml` — manual dispatch to build/stage and then promote local-area H5 files.
