"""Rewrite calibration_package.pkl metadata.dataset_path / metadata.db_path.

The Modal-built calibration package records absolute paths from the build
container (`/pipeline/artifacts/<run_id>/...`). When we copy the package down
to a local checkout, those paths no longer resolve, so the IPF conversion
step in `benchmark_export.build_ipf_inputs` fails its existence check before
it can read the SQLite DB or the dataset H5.

This helper repoints the metadata to the local copies so the rest of the
pipeline (export, IPF conversion, runners) can run unchanged. It does *not*
re-fit anything — it only mutates the metadata dict and re-pickles the
package. Idempotent: re-running with the same paths is a no-op.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def patch(
    package_path: Path,
    *,
    dataset_path: Path | None,
    db_path: Path | None,
) -> dict:
    with open(package_path, "rb") as f:
        package = pickle.load(f)
    metadata = package.get("metadata", {}) or {}
    before = {
        "dataset_path": metadata.get("dataset_path"),
        "db_path": metadata.get("db_path"),
    }
    if dataset_path is not None:
        metadata["dataset_path"] = str(dataset_path.resolve())
    if db_path is not None:
        metadata["db_path"] = str(db_path.resolve())
    package["metadata"] = metadata
    with open(package_path, "wb") as f:
        pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "before": before,
        "after": {
            "dataset_path": metadata.get("dataset_path"),
            "db_path": metadata.get("db_path"),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--package",
        type=Path,
        default=Path(
            "policyengine_us_data/storage/calibration/calibration_package.pkl"
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Local path to the source dataset H5 referenced by IPF.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("policyengine_us_data/storage/calibration/policy_data.db"),
        help="Local path to policy_data.db.",
    )
    args = parser.parse_args(argv)
    info = patch(
        args.package,
        dataset_path=args.dataset_path,
        db_path=args.db_path,
    )
    import json

    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
