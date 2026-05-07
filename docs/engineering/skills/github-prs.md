# GitHub PRs

These rules apply to every developer and AI agent opening pull requests in this
repository.

## Same-repository PRs only

Open PRs from branches in `PolicyEngine/policyengine-us-data`, not from personal
forks. The PR workflow has a `check-fork` gate because fork PRs cannot access the
secrets needed by the data and Modal checks.

Before creating or sharing a PR:

1. Confirm the canonical repository is reachable:
   `gh repo view PolicyEngine/policyengine-us-data --json nameWithOwner`.
2. Add the required towncrier changelog fragment under `changelog.d/`.
3. Push the current branch to the canonical repository:
   `make push-pr-branch`.
4. Create the PR from that same repository:
   `gh pr create --repo PolicyEngine/policyengine-us-data --head "$(git branch --show-current)" --base main`.
5. Verify the PR head repository:
   `gh pr view <PR> --repo PolicyEngine/policyengine-us-data --json headRepositoryOwner,headRepository`.

The PR is valid only if the head repository is
`PolicyEngine/policyengine-us-data`. If you cannot push to the canonical
repository, stop and ask for access. Do not create a fork PR as a fallback. If
you accidentally create one, close it immediately and replace it with a
same-repository PR.

## Changelog fragment

A changelog entry is required before opening, replacing, or sharing a PR. When a
user asks an AI agent to open a PR, the agent must check for an appropriate
fragment and add one if it is missing before running `gh pr create`.

Use towncrier fragments in this format:

```text
changelog.d/<short-slug>.<type>.md
```

Allowed `<type>` values are configured in `pyproject.toml`:

- `breaking`
- `added`
- `changed`
- `fixed`
- `removed`

Examples:

```text
changelog.d/fix-stage-validation.fixed.md
changelog.d/add-agent-pr-guidance.changed.md
```

Write one concise Markdown sentence in the fragment. Use `fixed` for bug fixes,
`added` for new user-facing capabilities, `changed` for behavior, documentation,
tooling, or refactors, `removed` for removals, and `breaking` only for changes
that intentionally break compatibility. Prefer updating an existing branch
fragment over adding duplicate fragments for the same PR.

## PR title

Do not add `[codex]`, `[claude]`, `[copilot]`, or other agent labels to PR
titles. Use a plain descriptive title.
