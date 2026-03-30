"""Integration test configuration.

Skips tests when prerequisite data files are not available.
Provides shared fixtures for calibration database and dataset paths.
"""

import pytest
from sqlalchemy import create_engine

from policyengine_us_data.db.create_database_tables import (
    create_or_replace_views,
)
from policyengine_us_data.storage import STORAGE_FOLDER

# ── Skip logic for missing datasets ───────────────────────────

NEEDS_ECPS = not (STORAGE_FOLDER / "enhanced_cps_2024.h5").exists()
NEEDS_CPS = not (STORAGE_FOLDER / "cps_2024.h5").exists()

collect_ignore_glob = []
if NEEDS_ECPS:
    collect_ignore_glob.extend(
        [
            "test_enhanced_cps.py",
            "test_small_enhanced_cps.py",
            "test_sparse_enhanced_cps.py",
            "test_sipp_assets.py",
        ]
    )
if NEEDS_CPS:
    collect_ignore_glob.append("test_cps.py")


# ── Shared fixtures for calibration tests ─────────────────────


@pytest.fixture(scope="session", autouse=True)
def refresh_policy_db_views():
    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    if db_path.exists():
        engine = create_engine(f"sqlite:///{db_path}")
        try:
            create_or_replace_views(engine)
        finally:
            engine.dispose()


@pytest.fixture(scope="module")
def db_uri():
    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    return f"sqlite:///{db_path}"


@pytest.fixture(scope="module")
def dataset_path():
    return str(
        STORAGE_FOLDER / "source_imputed_stratified_extended_cps_2024.h5"
    )
