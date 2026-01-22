"""Shared fixtures for local area calibration tests.

Importantly, this file determines which variables will be included in the sparse matrix and calibrating routine.
"""

import pytest
import numpy as np
from sqlalchemy import create_engine, text

from policyengine_us import Microsimulation
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.local_area_calibration.sparse_matrix_builder import (
    SparseMatrixBuilder,
)
from policyengine_us_data.datasets.cps.local_area_calibration.matrix_tracer import (
    MatrixTracer,
)
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_calculated_variables,
)

# Variables to test for state-level value matching
# Format: (variable_name, rtol)
#     variable_name as per the targets in policy_data.db
#     rtol is relative tolerance for comparison
VARIABLES_TO_TEST = [
    ("snap", 1e-2),
    ("health_insurance_premiums_without_medicare_part_b", 1e-2),
    ("medicaid", 1e-2),
    ("medicare_part_b_premiums", 1e-2),
    ("other_medical_expenses", 1e-2),
    ("over_the_counter_health_expenses", 1e-2),
    ("salt_deduction", 1e-2),
    ("spm_unit_capped_work_childcare_expenses", 1e-2),
    ("spm_unit_capped_housing_subsidy", 1e-2),
    ("ssi", 1e-2),
    ("tanf", 1e-2),
    ("tip_income", 1e-2),
    ("unemployment_compensation", 1e-2),
]

# Combined filter config to build matrix with all variables at once
COMBINED_FILTER_CONFIG = {
    "stratum_group_ids": [
        4,  # SNAP targets
        5,  # Medicaid targets
        112,  # Unemployment compensation targets
    ],
    "variables": [
        "snap",
        "health_insurance_premiums_without_medicare_part_b",
        "medicaid",
        "medicare_part_b_premiums",
        "other_medical_expenses",
        "over_the_counter_health_expenses",
        "salt_deduction",
        "spm_unit_capped_work_childcare_expenses",
        "spm_unit_capped_housing_subsidy",
        "ssi",
        "tanf",
        "tip_income",
        "unemployment_compensation",
    ],
}

# Maximum allowed mismatch rate for state-level value comparison
MAX_MISMATCH_RATE = 0.02

# Number of samples for cell-level verification tests
N_VERIFICATION_SAMPLES = 200


@pytest.fixture(scope="module")
def db_uri():
    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    return f"sqlite:///{db_path}"


@pytest.fixture(scope="module")
def dataset_path():
    return str(STORAGE_FOLDER / "stratified_extended_cps_2023.h5")


@pytest.fixture(scope="module")
def test_cds(db_uri):
    """CDs from multiple states for comprehensive testing."""
    engine = create_engine(db_uri)
    query = """
    SELECT DISTINCT sc.value as cd_geoid
    FROM strata s
    JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
    WHERE s.stratum_group_id = 1
      AND sc.constraint_variable = 'congressional_district_geoid'
      AND (
        sc.value LIKE '37__'
        OR sc.value LIKE '150_'
        OR sc.value LIKE '300_'
        OR sc.value = '200' OR sc.value = '201'
        OR sc.value IN ('101', '102')
        OR sc.value IN ('601', '602')
        OR sc.value IN ('3601', '3602')
        OR sc.value IN ('4801', '4802')
      )
    ORDER BY sc.value
    """
    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchall()
        return [row[0] for row in result]


@pytest.fixture(scope="module")
def sim(dataset_path):
    return Microsimulation(dataset=dataset_path)


@pytest.fixture(scope="module")
def matrix_data(db_uri, dataset_path, test_cds, sim):
    """Build sparse matrix with all configured variables."""
    builder = SparseMatrixBuilder(
        db_uri,
        time_period=2023,
        cds_to_calibrate=test_cds,
        dataset_path=dataset_path,
    )
    targets_df, X_sparse, household_id_mapping = builder.build_matrix(
        sim, target_filter=COMBINED_FILTER_CONFIG
    )
    return targets_df, X_sparse, household_id_mapping


@pytest.fixture(scope="module")
def targets_df(matrix_data):
    return matrix_data[0]


@pytest.fixture(scope="module")
def X_sparse(matrix_data):
    return matrix_data[1]


@pytest.fixture(scope="module")
def household_id_mapping(matrix_data):
    return matrix_data[2]


@pytest.fixture(scope="module")
def tracer(targets_df, X_sparse, household_id_mapping, test_cds, sim):
    return MatrixTracer(
        targets_df, X_sparse, household_id_mapping, test_cds, sim
    )


@pytest.fixture(scope="module")
def n_households(tracer):
    return tracer.n_households


@pytest.fixture(scope="module")
def household_ids(tracer):
    return tracer.original_household_ids


@pytest.fixture(scope="module")
def household_states(sim):
    return sim.calculate("state_fips", map_to="household").values


def create_state_simulation(dataset_path, n_households, state):
    """Create simulation with all households assigned to a specific state."""
    s = Microsimulation(dataset=dataset_path)
    s.set_input(
        "state_fips", 2023, np.full(n_households, state, dtype=np.int32)
    )
    for var in get_calculated_variables(s):
        s.delete_arrays(var)
    return s
