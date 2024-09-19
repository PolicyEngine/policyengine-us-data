## Updating data

If your changes present a non-bugfix change to one or more datasets which are cloud-hosted (CPS, ECPS and PUF), then please change both the filename and URL (in both the class definition file and in `storage/upload_completed_datasets.py`. This enables us to store historical versions of datasets separately and reproducibly.

## Updating versioning

Please add a versioning entry to `changelog_entry.yaml` (see previous PRs for examples), then run `make changelog` and commit the results ONCE in this PR.
