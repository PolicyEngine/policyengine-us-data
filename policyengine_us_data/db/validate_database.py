"""
This is the start of a data validation pipeline. It is meant to be a separate
validation track from the unit tests in policyengine_us_data/tests in that it tests
the overall correctness of data after a full pipeline run with production data.
"""

import sqlite3

import pandas as pd
from policyengine_us.system import system

conn = sqlite3.connect(
    "policyengine_us_data/storage/calibration/policy_data.db"
)

stratum_constraints_df = pd.read_sql("SELECT * FROM stratum_constraints", conn)
targets_df = pd.read_sql("SELECT * FROM targets", conn)

for var_name in set(targets_df["variable"]):
    if not var_name in system.variables.keys():
        raise ValueError(f"{var_name} not a policyengine-us variable")

for var_name in set(stratum_constraints_df["constraint_variable"]):
    if not var_name in system.variables.keys():
        raise ValueError(f"{var_name} not a policyengine-us variable")
