# Codex Instructions

These instructions apply repository-wide.

## Skills system

Canonical AI-facing engineering skills live under `docs/engineering/skills/`.
Use those files as the source of truth across Codex, Claude, Copilot, and other
AI tools.

When adding, moving, or reviewing tests, read
`docs/engineering/skills/testing.md`. Do not put pytest files under
`policyengine_us_data/tests/`, do not import from `tests.conftest`, and do not
import helpers across test lanes.
