# Engineering Skills

This directory is the canonical source for AI-facing engineering rules.

Tool-specific instruction files such as `AGENTS.md`, `CLAUDE.md`, and
`.github/copilot-instructions.md` should point here instead of duplicating
implementation-specific guidance. When a rule changes, update the skill here
first, then keep adapters thin.

Current skills:

- `pipeline_docs.md`: decorator-backed pipeline map maintenance and generated
  pydoc-style artifacts.
- `testing.md`: test layout, fixture scope, helper placement, and quality guard
  expectations.
