"""
This is the start of a data validation pipeline. It is meant to be a separate
validation track from the unit tests in policyengine_us_data/tests in that it tests
the overall correctness of data after a full pipeline run with production data.
"""

import logging
import sqlite3

import numpy as np
import pandas as pd
from policyengine_us.system import system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

conn = sqlite3.connect(
    "policyengine_us_data/storage/calibration/policy_data.db"
)

stratum_constraints_df = pd.read_sql("SELECT * FROM stratum_constraints", conn)
targets_df = pd.read_sql("SELECT * FROM targets", conn)

for var_name in set(targets_df["variable"]):
    if var_name not in system.variables.keys():
        raise ValueError(f"{var_name} not a policyengine-us variable")

for var_name in set(stratum_constraints_df["constraint_variable"]):
    if var_name not in system.variables.keys():
        raise ValueError(f"{var_name} not a policyengine-us variable")


# ------------------------------------------------------------------
# Validate geographic target reconciliation
# ------------------------------------------------------------------

strata_df = pd.read_sql("SELECT * FROM strata", conn)

# Build parent chain for each stratum to find geo ancestor
geo_strata = strata_df[strata_df["stratum_group_id"] == 1].copy()

# Classify geo strata
sc_df = stratum_constraints_df.copy()
geo_constraints = sc_df[
    sc_df["stratum_id"].isin(geo_strata["stratum_id"])
].copy()

national_ids = set(
    geo_strata[geo_strata["parent_stratum_id"].isna()]["stratum_id"]
)

state_constraints = geo_constraints[
    geo_constraints["constraint_variable"] == "state_fips"
]
state_stratum_to_fips = dict(
    zip(
        state_constraints["stratum_id"],
        state_constraints["value"].astype(int),
    )
)

cd_constraints = geo_constraints[
    geo_constraints["constraint_variable"] == "congressional_district_geoid"
]
cd_stratum_to_geoid = dict(
    zip(
        cd_constraints["stratum_id"],
        cd_constraints["value"].astype(int),
    )
)

# Build a lookup: stratum_id -> geo ancestor stratum_id
parent_map = dict(zip(strata_df["stratum_id"], strata_df["parent_stratum_id"]))
group_map = dict(zip(strata_df["stratum_id"], strata_df["stratum_group_id"]))


def find_geo_ancestor(sid):
    """Walk up parent chain to find geo stratum (group_id==1)."""
    if group_map.get(sid) == 1:
        return sid
    current = sid
    visited = set()
    while current is not None and current not in visited:
        visited.add(current)
        p = parent_map.get(current)
        if p is None or np.isnan(p) if isinstance(p, float) else False:
            return None
        p = int(p)
        if group_map.get(p) == 1:
            return p
        current = p
    return None


active_targets = targets_df[targets_df["active"] == 1].copy()
active_targets["geo_ancestor"] = active_targets["stratum_id"].apply(
    find_geo_ancestor
)

# Drop targets with no geo ancestor
geo_targets = active_targets.dropna(subset=["geo_ancestor"]).copy()
geo_targets["geo_ancestor"] = geo_targets["geo_ancestor"].astype(int)


# Classify each target's geo level
def classify_geo(geo_sid):
    if geo_sid in national_ids:
        return "national"
    if geo_sid in state_stratum_to_fips:
        return "state"
    if geo_sid in cd_stratum_to_geoid:
        return "district"
    return "unknown"


geo_targets["geo_level"] = geo_targets["geo_ancestor"].apply(classify_geo)

# For state targets, get state_fips
geo_targets["state_fips"] = geo_targets["geo_ancestor"].map(
    state_stratum_to_fips
)

# For district targets, derive state_fips from geoid
geo_targets["cd_geoid"] = geo_targets["geo_ancestor"].map(cd_stratum_to_geoid)
geo_targets.loc[geo_targets["geo_level"] == "district", "state_fips"] = (
    geo_targets.loc[geo_targets["geo_level"] == "district", "cd_geoid"] // 100
)

# Check: for each (variable, period, reform_id), sum(state) ≈ national
RTOL = 1e-6
reconciliation_failures = []

for (var, period, reform), group in geo_targets.groupby(
    ["variable", "period", "reform_id"]
):
    nat = group[group["geo_level"] == "national"]
    states = group[group["geo_level"] == "state"]

    if nat.empty or states.empty:
        continue

    nat_val = nat["value"].sum()
    state_sum = states["value"].sum()

    if nat_val != 0 and state_sum != 0:
        ratio = abs(state_sum - nat_val) / abs(nat_val)
        if ratio > RTOL:
            reconciliation_failures.append(
                {
                    "variable": var,
                    "period": period,
                    "reform_id": reform,
                    "level": "state->national",
                    "parent_value": nat_val,
                    "child_sum": state_sum,
                    "ratio": ratio,
                }
            )

    # Check CDs sum to state
    districts = group[group["geo_level"] == "district"]
    if districts.empty:
        continue

    for fips, state_group in states.groupby("state_fips"):
        state_val = state_group["value"].sum()
        cd_group = districts[districts["state_fips"] == fips]
        if cd_group.empty:
            continue
        cd_sum = cd_group["value"].sum()

        if state_val != 0 and cd_sum != 0:
            ratio = abs(cd_sum - state_val) / abs(state_val)
            if ratio > RTOL:
                reconciliation_failures.append(
                    {
                        "variable": var,
                        "period": period,
                        "reform_id": reform,
                        "level": f"cd->state(fips={fips})",
                        "parent_value": state_val,
                        "child_sum": cd_sum,
                        "ratio": ratio,
                    }
                )

if reconciliation_failures:
    logger.warning(
        "Found %d geographic reconciliation mismatches:",
        len(reconciliation_failures),
    )
    for f in reconciliation_failures[:10]:
        logger.warning(
            "  %s %s/%s: %s parent=%.2f child_sum=%.2f " "ratio=%.6f",
            f["variable"],
            f["period"],
            f["reform_id"],
            f["level"],
            f["parent_value"],
            f["child_sum"],
            f["ratio"],
        )
    raise ValueError(
        f"{len(reconciliation_failures)} geographic target groups "
        f"are not reconciled (rtol={RTOL})"
    )

logger.info("All geographic target reconciliation checks passed.")

conn.close()
