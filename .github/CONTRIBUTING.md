## Updating data

If your changes present a non-bugfix change to one or more datasets which are cloud-hosted (CPS, ECPS and PUF), then please change both the filename and URL (in both the class definition file and in `storage/upload_completed_datasets.py`. This enables us to store historical versions of datasets separately and reproducibly.

## Opening PRs

Push PR branches to the upstream `PolicyEngine/policyengine-us-data` repository, not to a personal fork. From the repo root, run:

`make push-pr-branch`

This avoids the fork-only CI failure path and sets the upstream tracking branch correctly before opening the PR.
