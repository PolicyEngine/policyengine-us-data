"""Demo: ACA PTC hierarchical uprating pipeline."""

import sys
import numpy as np
import pandas as pd

sys.path.insert(
    0,
    str(__import__("pathlib").Path(__file__).resolve().parent),
)

from policyengine_us_data.storage import STORAGE_FOLDER
from sparse_matrix_builder import SparseMatrixBuilder
from calibration_utils import get_all_cds_from_database, STATE_CODES

db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
db_uri = f"sqlite:///{db_path}"
cds = get_all_cds_from_database(db_uri)
builder = SparseMatrixBuilder(db_uri, time_period=2024, cds_to_calibrate=cds)

# --- 1. Query all aca_ptc targets ---
print("=" * 70)
print("1. RAW TARGETS from target_overview (domain_variable = 'aca_ptc')")
print("=" * 70)
raw = builder._query_targets({"domain_variables": ["aca_ptc"]})
summary = (
    raw.groupby(["geo_level", "variable", "period"])
    .agg(count=("value", "size"), total_value=("value", "sum"))
    .reset_index()
)
print(summary.to_string(index=False))
print(f"\nTotal rows: {len(raw)}")

# --- 2. Apply generic (CPI/pop) uprating ---
print("\n" + "=" * 70)
print("2. GENERIC UPRATING (CPI/pop factors)")
print("=" * 70)

from policyengine_us import Microsimulation

sim = Microsimulation()
params = sim.tax_benefit_system.parameters
uprating_factors = builder._calculate_uprating_factors(params)

for (yr, kind), f in sorted(uprating_factors.items()):
    if f != 1.0:
        print(f"  {yr} -> 2024 ({kind}): {f:.6f}")

raw["original_value"] = raw["value"].copy()
raw["uprating_factor"] = raw.apply(
    lambda r: builder._get_uprating_info(
        r["variable"], r["period"], uprating_factors
    )[0],
    axis=1,
)
raw["value"] = raw["original_value"] * raw["uprating_factor"]

# Show before/after for sample states
sample_states = {6: "CA", 48: "TX", 36: "NY"}
print("\nBefore / After for sample states (state-level rows):")
for fips, abbr in sample_states.items():
    rows = raw[
        (raw["geo_level"] == "state") & (raw["geographic_id"] == str(fips))
    ]
    for _, r in rows.iterrows():
        print(
            f"  {abbr} {r['variable']:20s}  "
            f"orig={r['original_value']:>14,.0f}  "
            f"factor={r['uprating_factor']:.4f}  "
            f"uprated={r['value']:>14,.0f}"
        )

# --- 3. Hierarchical reconciliation ---
print("\n" + "=" * 70)
print("3. HIERARCHICAL RECONCILIATION (CD values rescaled to state)")
print("=" * 70)

result = builder._apply_hierarchical_uprating(
    raw, ["aca_ptc"], uprating_factors
)

for fips, abbr in sample_states.items():
    cd_rows = result[
        (result["geo_level"] == "district")
        & (
            result["geographic_id"].apply(
                lambda g: int(g) // 100 == fips if g not in ("US",) else False
            )
        )
    ]
    for var in cd_rows["variable"].unique():
        var_rows = cd_rows[cd_rows["variable"] == var]
        recon = var_rows["reconciliation_factor"].iloc[0]
        cd_sum = var_rows["value"].sum()
        # Look up the uprated state value from raw (pre-drop)
        st_row = raw[
            (raw["geo_level"] == "state")
            & (raw["geographic_id"] == str(fips))
            & (raw["variable"] == var)
        ]
        uprated_state = st_row["value"].iloc[0] if len(st_row) else np.nan
        print(
            f"  {abbr} {var:20s}  "
            f"recon_factor={recon:.6f}  "
            f"sum(CDs)={cd_sum:>14,.0f}  "
            f"uprated_state={uprated_state:>14,.0f}"
        )

# --- 4. Row filtering ---
print("\n" + "=" * 70)
print("4. ROW FILTERING (geo_level counts after hierarchical uprating)")
print("=" * 70)
level_counts = result["geo_level"].value_counts()
print(level_counts.to_string())
nat_rows = result[result["geo_level"] == "national"]
if len(nat_rows):
    print(f"\nKept national rows:")
    for _, r in nat_rows.iterrows():
        print(
            f"  {r['variable']}  period={r['period']}  value={r['value']:,.0f}"
        )

# --- 5. Verification: sum(CDs) == uprated state for all 51 ---
print("\n" + "=" * 70)
print("5. VERIFICATION: sum(CDs) == uprated state for all states")
print("=" * 70)

all_ok = True
for fips, abbr in sorted(STATE_CODES.items()):
    cd_rows = result[
        (result["geo_level"] == "district")
        & (
            result["geographic_id"].apply(
                lambda g, s=fips: (
                    int(g) // 100 == s if g not in ("US",) else False
                )
            )
        )
    ]
    if cd_rows.empty:
        continue
    for var in cd_rows["variable"].unique():
        cd_sum = cd_rows[cd_rows["variable"] == var]["value"].sum()
        st = raw[
            (raw["geo_level"] == "state")
            & (raw["geographic_id"] == str(fips))
            & (raw["variable"] == var)
        ]
        if st.empty:
            continue
        uprated = st["value"].iloc[0]
        ok = np.isclose(cd_sum, uprated, rtol=1e-6)
        if not ok:
            print(
                f"  FAIL {abbr} {var}: sum(CDs)={cd_sum:.2f} != state={uprated:.2f}"
            )
            all_ok = False

print("  ALL PASSED" if all_ok else "  SOME FAILED")
