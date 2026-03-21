"""Skip dataset tests that need full data build artifacts.

In basic CI (full_suite=false), H5 files are not built locally
and Microsimulation requires ~16GB RAM. These tests run inside
Modal containers (32GB) during full_suite=true builds.
"""

import pytest
from policyengine_us_data.storage import STORAGE_FOLDER

NEEDS_ECPS = not (STORAGE_FOLDER / "enhanced_cps_2024.h5").exists()
NEEDS_CPS = not (STORAGE_FOLDER / "cps_2024.h5").exists()

collect_ignore_glob = []
if NEEDS_ECPS:
    collect_ignore_glob.extend(
        [
            "test_enhanced_cps.py",
            "test_dataset_sanity.py",
            "test_small_enhanced_cps.py",
            "test_sparse_enhanced_cps.py",
            "test_sipp_assets.py",
        ]
    )
if NEEDS_CPS:
    collect_ignore_glob.append("test_cps.py")
