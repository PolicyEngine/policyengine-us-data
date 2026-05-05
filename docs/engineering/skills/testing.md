# Testing Skill

Use this skill whenever adding, moving, or reviewing tests.

## Canonical Layout

- Put unit tests under `tests/unit/`.
- Put data-dependent, runtime, deployed Modal, and staging integration tests
  under `tests/integration/`.
- Put post-build artifact validators under `validation/stage_1/`. These checks
  consume already-built Stage 1 artifacts and assert that their file structure,
  runtime behavior, calibration outputs, or handoff contracts meet expectations.
- Do not add pytest files under `policyengine_us_data/tests/`; CI does not
  collect that tree.

## Integration Test Roles

- Use `validation/stage_1/` for post-build validators only. They should not
  rebuild Stage 1 artifacts or exercise unrelated orchestration seams; they
  should load the artifacts produced by the build and fail when those artifacts
  do not satisfy the expected contract.
- Treat the old full dataset-build PR path as legacy linear integration testing.
  It proves the production build can run, but it is not the model for new
  fixture-scale integration tests.
- Use intra-stage integration tests for Stage 1-5 runtime contracts. These
  should run locally on tiny fixtures and verify each stage's input/output
  artifact shape, schema, cache behavior, and failure surface without invoking
  the full build.
- Use the tiny H5 pipeline tests for H5-builder integration only. They may use
  Modal, run-specific IDs, and seeded artifacts, but their job is to test seams
  inside the H5 builder and local-area staging path, not the full Stage 1-5
  dataset pipeline.
- Use the tiny Modal pipeline E2E tests for the full handoff from Stage 1-5
  shaped artifacts into the Modal H5/publish path. These tests should prove the
  stage-output contract is accepted by the deployed pipeline path.
- Use Modal runtime seam tests for deployment-specific contracts: imports,
  baked files, subprocess entrypoints, function lookup, volume paths, and clean
  credential/token skip behavior.
- Avoid adding a second test that proves the same seam. If two tests both seed
  static H5 fixtures and call the same builder path, either split the asserted
  contract explicitly or remove the duplicate.

## Fixtures And Helpers

- Keep root `tests/conftest.py` empty or very lightweight. It must not import
  cloud clients, Modal, Hugging Face, PolicyEngine runtime-heavy modules, or
  package modules that transitively import those dependencies.
- Put domain-specific fixtures in the narrowest `conftest.py` that covers the
  tests that use them.
- Put reusable helper functions in a local `support.py`, a local fixture module,
  or `tests/support/`.
- Do not import from `tests.conftest`; pytest discovers fixtures automatically.
- Do not import across test lanes, for example from `tests.integration` into
  `tests.unit` or from `tests.unit` into `tests.integration`. Move shared helpers
  to `tests/support/` or colocate them with the tests.

## Dependency Boundaries

- Unit tests should not require real network credentials, Modal, Hugging Face,
  or GCS. Mock those seams.
- Integration tests may require built data or heavier runtime setup, but should
  be explicit about those requirements and skip cleanly when local artifacts are
  unavailable.
- CI should run tests in an environment where project dependencies are installed
  with `uv sync --dev` or an equivalent full test dependency install. A full
  install is required, but it is not a substitute for fixture isolation.

## Quality Guards

Run this before opening or updating a PR:

```bash
uv run --no-sync --with pyyaml python scripts/run_quality_guards.py
```

The current guards enforce:

- No package-internal pytest files under `policyengine_us_data/tests/`.
- No pytest files outside `tests/unit/`, `tests/integration/`, and
  `validation/stage_1/`.
- No imports from `tests.conftest`.
- No imports across test lanes.
- Structured pipeline documentation metadata is valid.
- Stable pydoc-facing library nodes have basic docstring and typing coverage.

When adding a new guard, register it in `scripts/run_quality_guards.py` so CI
continues to expose a single `Quality guards` job.
