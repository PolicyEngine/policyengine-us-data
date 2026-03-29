"""Shared fixtures for local area calibration tests."""

import pytest
from sqlalchemy import create_engine

from policyengine_us_data.db.create_database_tables import (
    create_or_replace_views,
)
from policyengine_us_data.storage import STORAGE_FOLDER


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
    return str(STORAGE_FOLDER / "source_imputed_stratified_extended_cps_2024.h5")
