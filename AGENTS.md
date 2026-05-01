# Agent Instructions

These instructions apply repository-wide.

## Skills system

Canonical AI-facing engineering skills live under `docs/engineering/skills/`.
Use those files as the source of truth across Codex, Claude, Copilot, and other
AI tools.

When adding, moving, or reviewing tests, read
`docs/engineering/skills/testing.md`. Do not put pytest files under
`policyengine_us_data/tests/`, do not import from `tests.conftest`, and do not
import helpers across test lanes.

## GitHub PRs

Do not open `policyengine-us-data` PRs from forks. CI expects same-repository
branches on `PolicyEngine/policyengine-us-data`, so push PR branches to the
`upstream` remote (or another remote whose `gh repo view --json nameWithOwner`
is `PolicyEngine/policyengine-us-data`). If you cannot push a branch to the
upstream repository, stop and ask for access instead of creating a fork-based
PR. Before sharing a PR, verify that the PR head repository is
`PolicyEngine/policyengine-us-data`.
