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
- `github-prs.md`: same-repository PR workflow, required changelog fragments,
  PR head verification, and title conventions.
- `pipeline_docs.md`: decorator-backed pipeline map maintenance and generated
  pydoc-style artifacts.
- `testing.md`: test layout, fixture scope, helper placement, and quality guard
  expectations.
