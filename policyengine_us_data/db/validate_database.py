"""
This is the start of a data validation pipeline. It is meant to be a separate
validation track from the unit tests in policyengine_us_data/tests in that it tests
the overall correctness of data after a full pipeline run with production data.
"""

import sqlite3

import pandas as pd
from policyengine_us.system import system

conn = sqlite3.connect("policyengine_us_data/storage/calibration/policy_data.db")

stratum_constraints_df = pd.read_sql("SELECT * FROM stratum_constraints", conn)
targets_df = pd.read_sql("SELECT * FROM targets", conn)

for var_name in set(targets_df["variable"]):
    if var_name not in system.variables.keys():
        raise ValueError(f"{var_name} not a policyengine-us variable")

for var_name in set(stratum_constraints_df["constraint_variable"]):
    if var_name not in system.variables.keys():
        raise ValueError(f"{var_name} not a policyengine-us variable")

TAX_EXPENDITURE_VARS = [
    "salt_deduction",
    "charitable_deduction",
    "deductible_mortgage_interest",
    "medical_expense_deduction",
    "qualified_business_income_deduction",
]

root_stratum_ids = pd.read_sql(
    "SELECT stratum_id FROM strata WHERE parent_stratum_id IS NULL", conn
)["stratum_id"].tolist()

for var in TAX_EXPENDITURE_VARS:
    matches = targets_df[
        (targets_df["variable"] == var)
        & (targets_df["active"] == 1)
        & (targets_df["stratum_id"].isin(root_stratum_ids))
        & (targets_df["reform_id"] > 0)
    ]
    if matches.empty:
        raise ValueError(
            f"Validation failed: {var} has no active target with "
            f"reform_id > 0 in the root stratum. Tax expenditure targets "
            f"must have a non-zero reform_id for correct calibration."
        )
