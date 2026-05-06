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
2. Push the current branch to the canonical repository:
   `make push-pr-branch`.
3. Create the PR from that same repository:
   `gh pr create --repo PolicyEngine/policyengine-us-data --head "$(git branch --show-current)" --base main`.
4. Verify the PR head repository:
   `gh pr view <PR> --repo PolicyEngine/policyengine-us-data --json headRepositoryOwner,headRepository`.

The PR is valid only if the head repository is
`PolicyEngine/policyengine-us-data`. If you cannot push to the canonical
repository, stop and ask for access. Do not create a fork PR as a fallback. If
you accidentally create one, close it immediately and replace it with a
same-repository PR.

## PR title

Do not add `[codex]`, `[claude]`, `[copilot]`, or other agent labels to PR
titles. Use a plain descriptive title.
