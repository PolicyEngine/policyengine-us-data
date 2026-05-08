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
2. Open a GitHub issue for the work, or verify that an appropriate issue
   already exists.
3. Put `Fixes #ISSUE_NUMBER` as the first line of the PR description, using the
   issue number from the issue created or found in the previous step.
4. Add a Towncrier changelog fragment under `changelog.d/` using the issue
   number and the appropriate configured type, for example
   `changelog.d/ISSUE_NUMBER.added`.
5. Run the repository lint target:
   `make lint`.
6. Push the current branch to the canonical repository:
   `make push-pr-branch`.
7. Create the PR as a draft from that same repository:
   `gh pr create --draft --repo PolicyEngine/policyengine-us-data --head "$(git branch --show-current)" --base main`.
8. Verify the PR is draft and the head repository is canonical:
   `gh pr view <PR> --repo PolicyEngine/policyengine-us-data --json isDraft,headRepositoryOwner,headRepository`.
9. Leave the PR as draft unless a maintainer explicitly asks for it to be
   marked ready for review.

The PR is valid only if `isDraft` is `true` and the head repository is
`PolicyEngine/policyengine-us-data`. If you cannot push to the canonical
repository, stop and ask for access. Do not create a fork PR as a fallback. If
you accidentally create one, close it immediately and replace it with a
same-repository draft PR.

## PR title

Do not add `[codex]`, `[claude]`, `[copilot]`, or other agent labels to PR
titles. Use a plain descriptive title.
