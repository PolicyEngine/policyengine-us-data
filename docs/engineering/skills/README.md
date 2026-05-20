# Engineering Skills

This directory is the canonical source for AI-facing engineering rules.

Tool-specific instruction files such as `AGENTS.md`, `CLAUDE.md`, and
`.github/copilot-instructions.md` should point here instead of duplicating
implementation-specific guidance. When a rule changes, update the skill here
first, then keep adapters thin.

Current skills:

- `documentation_review.md`: model-neutral review harness for checking pipeline
  docs, Pydoc coverage, generated artifacts, and PR-facing confidence/impact
  notes.
- `github-prs.md`: same-repository PR workflow, PR head verification, and title
  conventions.
- `pipeline_docs.md`: decorator-backed pipeline map maintenance and generated
  pydoc-style artifacts.
- `pipeline_operations.md`: model-neutral workflow for diagnosing deployed Modal
  pipeline status and durable error records.
- `testing.md`: test layout, fixture scope, helper placement, and quality guard
  expectations.

Stage-specific AI-facing engineering guides live under `docs/engineering/stages/`.
Use them alongside these cross-cutting skills when modifying a stage-specific
pipeline path.

Current stage guides:

- `build_outputs.md`: Stage 4 output-build library boundaries and test
  expectations.
- `release_promotion.md`: Stage 5 release candidate identity, validation-report
  schema, rerun comparison material, and side-effect boundaries.
