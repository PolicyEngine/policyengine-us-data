"""Shared fixtures for local area calibration tests."""

import pytest

from policyengine_us_data.storage import STORAGE_FOLDER


@pytest.fixture(scope="module")
def db_uri():
    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    return f"sqlite:///{db_path}"


@pytest.fixture(scope="module")
def dataset_path():
    return str(STORAGE_FOLDER / "stratified_extended_cps_2024.h5")
