"""
One-time migration script to copy files from v1.56.0/ to production paths.

Usage:
    python scripts/migrate_versioned_to_production.py --dry-run
    python scripts/migrate_versioned_to_production.py --execute
"""

import argparse
from google.cloud import storage
import google.auth
from huggingface_hub import HfApi, CommitOperationCopy
import os


def migrate_gcs(dry_run: bool = True):
    """Copy files from v1.56.0/ to production paths in GCS."""
    credentials, project_id = google.auth.default()
    client = storage.Client(credentials=credentials, project=project_id)
    bucket = client.bucket("policyengine-us-data")

    blobs = list(bucket.list_blobs(prefix="v1.56.0/"))
    print(f"Found {len(blobs)} files in v1.56.0/")

    copied = 0
    for blob in blobs:
        # v1.56.0/states/AL.h5 -> states/AL.h5
        new_name = blob.name.replace("v1.56.0/", "")
        if not new_name:
            continue

        if dry_run:
            print(f"  Would copy: {blob.name} -> {new_name}")
        else:
            bucket.copy_blob(blob, bucket, new_name)
            print(f"  Copied: {blob.name} -> {new_name}")
        copied += 1

    print(f"{'Would copy' if dry_run else 'Copied'} {copied} files in GCS")
    return copied


def migrate_hf(dry_run: bool = True):
    """Copy files from v1.56.0/ to production paths in HuggingFace."""
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()
    repo_id = "policyengine/policyengine-us-data"

    files = api.list_repo_files(repo_id)
    versioned_files = [f for f in files if f.startswith("v1.56.0/")]
    print(f"Found {len(versioned_files)} files in v1.56.0/")

    if dry_run:
        for f in versioned_files[:10]:
            new_path = f.replace("v1.56.0/", "")
            print(f"  Would copy: {f} -> {new_path}")
        if len(versioned_files) > 10:
            print(f"  ... and {len(versioned_files) - 10} more")
        return len(versioned_files)

    operations = []
    for f in versioned_files:
        new_path = f.replace("v1.56.0/", "")
        if not new_path:
            continue
        operations.append(
            CommitOperationCopy(
                src_path_in_repo=f,
                path_in_repo=new_path,
            )
        )

    if operations:
        api.create_commit(
            token=token,
            repo_id=repo_id,
            operations=operations,
            repo_type="model",
            commit_message="Promote v1.56.0 files to production paths",
        )
        print(f"Copied {len(operations)} files in one HuggingFace commit")

    return len(operations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without doing it",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Actually perform the migration"
    )
    parser.add_argument(
        "--gcs-only", action="store_true", help="Only migrate GCS"
    )
    parser.add_argument(
        "--hf-only", action="store_true", help="Only migrate HuggingFace"
    )
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("Must specify --dry-run or --execute")
        return

    dry_run = args.dry_run

    if not args.hf_only:
        print("\n=== GCS Migration ===")
        migrate_gcs(dry_run)

    if not args.gcs_only:
        print("\n=== HuggingFace Migration ===")
        migrate_hf(dry_run)

    if dry_run:
        print("\n(Dry run - no changes made. Use --execute to apply.)")


if __name__ == "__main__":
    main()
