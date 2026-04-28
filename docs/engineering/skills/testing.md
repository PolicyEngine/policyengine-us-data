# Testing Skill

Use this skill whenever adding, moving, or reviewing tests.

## Canonical Layout

- Put unit tests under `tests/unit/`.
- Put data-dependent or runtime integration tests under `tests/integration/`.
- Put deployed Modal/staging tests under `tests/optimized/`.
- Do not add pytest files under `policyengine_us_data/tests/`; CI does not
  collect that tree.

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
python scripts/run_quality_guards.py
```

The current guard enforces:

- No package-internal pytest files under `policyengine_us_data/tests/`.
- No pytest files outside the approved top-level test lanes.
- No imports from `tests.conftest`.
- No imports across test lanes.

When adding a new guard, register it in `scripts/run_quality_guards.py` so CI
continues to expose a single `Quality guards` job.
