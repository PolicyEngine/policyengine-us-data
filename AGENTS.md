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

## GitHub PRs

Read `docs/engineering/skills/github-prs.md` before opening, replacing, or
sharing any pull request.

Never open `policyengine-us-data` PRs from forks. CI rejects fork-based PRs
before running the real checks, which wastes the reviewer and agent loop.

Before creating or sharing any PR, all developers and agents must:

1. Confirm the target remote is the canonical repository:
   `gh repo view PolicyEngine/policyengine-us-data --json nameWithOwner`.
2. Push the branch to that repository, for example:
   `git push upstream HEAD:<branch-name>`.
3. Create the PR from the same repository, for example:
   `gh pr create --repo PolicyEngine/policyengine-us-data --head <branch-name> --base main`.
4. Verify the PR head repository before reporting it:
   `gh pr view <PR> --repo PolicyEngine/policyengine-us-data --json headRepositoryOwner,headRepository`.

The PR is valid only if the head repository is `PolicyEngine/policyengine-us-data`.
If you cannot push to the canonical repository, stop and ask for access. Do not
create a fork PR as a fallback. If you accidentally create one, immediately
close it and replace it with a same-repository PR.
