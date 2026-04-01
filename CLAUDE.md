# CLAUDE.md - Guidelines for PolicyEngine US Data

## Build Commands
- `make install` - Install dependencies and dev environment
- `make build` - Build the package using Python build
- `make data` - Generate project datasets

## Testing

### Running Tests
- `make test-unit` - Run unit tests only (fast, no data dependencies)
- `make test-integration` - Run integration tests (requires built H5 datasets)
- `make test` - Run all tests
- `pytest tests/unit/ -v` - Unit tests directly
- `pytest tests/integration/test_cps.py -v` - Specific integration test

### Test Organization
Tests are in the top-level `tests/` directory, split into two sub-directories:

- **`tests/unit/`** — Self-contained tests that use synthetic data, mocks, patches, or checked-in fixtures. Run in seconds with no external dependencies.
  - `unit/datasets/` — unit tests for dataset code
  - `unit/calibration/` — unit tests for calibration code

- **`tests/integration/`** — Tests that require built H5 datasets, HuggingFace downloads, Microsimulation objects, or database ETL. Named after the dataset they test.

### Test Placement Rules
- **NEVER** put tests that require H5 files or Microsimulation in `unit/`
- **NEVER** put tests that use only synthetic data or mocks in `integration/`
- Integration test files are named after their dataset dependency: `test_cps.py` tests `cps_2024.h5`
- Sanity checks (value ranges, population counts) belong in the per-dataset integration test file, not in a separate sanity file
- When adding a new integration test, add it to the existing per-dataset file if one exists

## Formatting
- `make format` - Format all code using ruff
- `ruff format --check .` - Check formatting without changing files
- `ruff check .` - Run linter

## Code Style Guidelines
- **Imports**: Standard libraries first, then third-party, then internal
- **Type Hints**: Use for all function parameters and return values
- **Naming**: Classes: PascalCase, Functions/Variables: snake_case, Constants: UPPER_SNAKE_CASE
- **Documentation**: Google-style docstrings with Args and Returns sections
- **Error Handling**: Use validation checks with specific error messages
- **Line Length**: ruff default (see pyproject.toml for any override)
- **Python Version**: Targeting Python 3.12-3.14

## CI/CD Structure
Four workflow files in `.github/workflows/`:

- **`pr.yaml`** — Runs on every PR to main: fork check, lint, uv.lock freshness, changelog fragment, unit tests with Codecov, smoke test. Integration tests run only with `run-integration` label. ~2-3 minutes.
- **`push.yaml`** — Runs on push to main. Two paths:
  - Version bump commits (`Update package version`): build and publish to PyPI
  - All other commits: full Modal data build with integration tests → manual approval gate → pipeline dispatch
- **`pipeline.yaml`** — Dispatch only. Spawns the H5 generation pipeline on Modal with scope filtering (all/national/state/congressional/local/test).
- **`versioning.yaml`** — Auto-bumps version when changelog.d fragments are merged. Commits `Update package version` which triggers the publish path in push.yaml.

## Git and PR Guidelines
- **CRITICAL**: NEVER create PRs from personal forks - ALL PRs MUST be created from branches pushed to the upstream PolicyEngine repository
- CI requires access to secrets that are not available to fork PRs for security reasons
- Fork PRs will fail on data download steps and cannot be merged
- Before opening a PR, always run `make push-pr-branch` from the repo root. This pushes the current branch to the `upstream` remote and sets the upstream tracking branch correctly for PR creation.
- Do not prefix PR titles with `[codex]` or any other agent label. Use the plain descriptive title.
- Always create branches directly on the upstream repository:
  ```bash
  git checkout main
  git pull upstream main
  git checkout -b your-branch-name
  make push-pr-branch
  ```
- Use descriptive branch names like `fix-issue-123` or `add-feature-name`
- Always run `make format` before committing

## CRITICAL RULES FOR ACADEMIC INTEGRITY

### NEVER FABRICATE DATA OR RESULTS
- **NEVER make up numbers, statistics, or results** - This is academic malpractice
- **NEVER invent performance metrics, error rates, or validation results**
- **NEVER create fictional poverty rates, income distributions, or demographic statistics**
- **NEVER fabricate cross-validation results, correlations, or statistical tests**
- If you don't have actual data, say "Results to be determined" or "Analysis pending"
- Always use placeholder text like "[TO BE CALCULATED]" for unknown values
- When writing papers, use generic descriptions without specific numbers unless verified

### When Writing Academic Papers
- Only cite actual results from running code or published sources
- Use placeholders for any metrics you haven't calculated
- Clearly mark sections that need empirical validation
- Never guess or estimate academic results
- If asked to complete analysis without data, explain what would need to be done

### Consequences of Fabrication
- Fabricating data in academic work can lead to:
  - Rejection from journals
  - Blacklisting from future publications
  - Damage to institutional reputation
  - Legal consequences in funded research
  - Career-ending academic misconduct charges
