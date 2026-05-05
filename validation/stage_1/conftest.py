"""Stage 1 post-build validation configuration."""

import pytest
from sqlalchemy import create_engine

from policyengine_us_data.db.create_database_tables import (
    create_or_replace_views,
)
from policyengine_us_data.storage import STORAGE_FOLDER


NEEDS_ECPS = not (STORAGE_FOLDER / "enhanced_cps_2024.h5").exists()
NEEDS_SMALL_ECPS = not (STORAGE_FOLDER / "small_enhanced_cps_2024.h5").exists()
NEEDS_EXTENDED_CPS = not (STORAGE_FOLDER / "extended_cps_2024.h5").exists()

collect_ignore_glob = []
if NEEDS_ECPS:
    collect_ignore_glob.extend(
        [
            "test_enhanced_cps.py",
            "test_sparse_enhanced_cps.py",
            "test_sipp_assets.py",
        ]
    )
if NEEDS_SMALL_ECPS:
    collect_ignore_glob.append("test_small_enhanced_cps.py")
if NEEDS_EXTENDED_CPS:
    collect_ignore_glob.append("test_no_formula_variables_stored.py")


@pytest.fixture(scope="session", autouse=True)
def refresh_policy_db_views():
    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    if db_path.exists():
        engine = create_engine(f"sqlite:///{db_path}")
        try:
            create_or_replace_views(engine)
        finally:
            engine.dispose()
