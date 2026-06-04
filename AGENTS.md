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

When reviewing PRs that change pipeline behavior, stage boundaries, generated
artifacts, or public library functions, read
`docs/engineering/skills/documentation_review.md`.

When diagnosing a deployed Modal pipeline run or a failed publication pipeline,
read `docs/engineering/skills/pipeline_operations.md`.

When adding, changing, or reviewing calibration target definitions, read
`docs/engineering/skills/calibration_targets.md`.

When adding, changing, or reviewing donor-survey imputations, read
`docs/engineering/skills/imputation.md`.

## Weighting / population aggregates (CRITICAL)

Never read or sum a weight array directly, and never report unweighted record
counts or raw HDF5 column sums as population figures — both are wrong. To get any
population aggregate from a dataset, load it as a `Microsimulation` and aggregate
the result; microdf auto-weights with the household weight, so you never touch a
weight:

```python
from policyengine_us import Microsimulation
sim = Microsimulation(dataset=path)
total      = sim.calculate("taxable_private_pension_income", 2024).sum()        # weighted $
recipients = (sim.calculate("taxable_private_pension_income", 2024) > 0).sum()  # weighted count
```

If you ever must reference a weight at all, it is `household_weight` ONLY; the
person/tax_unit/family/marital weights are derived and must never be used directly.

## Calibration targets

Manually sourced national or local-file calibration targets must be registered
in every active target path before merging:

1. `policyengine_us_data/utils/loss.py` for the ECPS loss matrix.
2. The appropriate `policyengine_us_data/db/etl_*.py` loader for
   `policy_data.db` and local H5 validation inputs. National targets usually
   belong in `etl_national_targets.py`; state or local targets should use a
   state/local ETL module and must still be added to this DB path.
3. `policyengine_us_data/calibration/target_config.yaml` when the default
   calibration uses an `include:` list; otherwise the target can exist in
   `policy_data.db` but still be omitted from calibration.

Do not treat a target appearing in `policy_data.db` as proof that published
datasets were calibrated to it. Add or update tests that fail if a new target is
present in one path but missing from another.

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
3. Create the PR as a draft from the same repository, for example:
   `gh pr create --draft --repo PolicyEngine/policyengine-us-data --head <branch-name> --base main`.
4. Verify the PR is draft and the head repository is canonical before reporting
   it:
   `gh pr view <PR> --repo PolicyEngine/policyengine-us-data --json isDraft,headRepositoryOwner,headRepository`.

The PR is valid only if `isDraft` is `true` and the head repository is
`PolicyEngine/policyengine-us-data`.
If you cannot push to the canonical repository, stop and ask for access. Do not
create a fork PR as a fallback. If you accidentally create one, immediately
close it and replace it with a same-repository draft PR.
