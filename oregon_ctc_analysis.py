"""
Oregon Child Tax Credit Analysis by State Senate District

Calculates the impact of doubling Oregon's Young Child Tax Credit (or_ctc)
by State Legislative District Upper (SLDU) - i.e., State Senate districts.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from policyengine_us import Microsimulation
from policyengine_core.reforms import Reform

# Local imports
from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
    assign_geography_for_cd,
    load_block_crosswalk,
)
from policyengine_us_data.storage import STORAGE_FOLDER

# Oregon congressional districts (119th Congress)
# Oregon has 6 CDs, geoid format: state_fips * 100 + district
# Oregon FIPS = 41, so: 4101, 4102, 4103, 4104, 4105, 4106
OREGON_CD_GEOIDS = [4101, 4102, 4103, 4104, 4105, 4106]


def load_district_data(cd_geoid: int) -> dict:
    """Load household data from a district H5 file."""
    h5_path = STORAGE_FOLDER / "districts" / f"OR-{cd_geoid % 100:02d}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"District file not found: {h5_path}")

    import h5py

    data = {}
    with h5py.File(h5_path, "r") as f:
        # Get key variables we need
        for var in [
            "household_weight",
            "household_id",
            "person_id",
            "age",
            "is_tax_unit_head",
            "tax_unit_id",
        ]:
            if var in f:
                # Handle year dimension if present
                arr = f[var][:]
                if len(arr.shape) > 1:
                    arr = arr[:, 0]  # Take first year
                data[var] = arr
    return data


def run_oregon_ctc_analysis():
    """Run the Oregon CTC analysis by state senate district."""
    print("=" * 60)
    print("Oregon Child Tax Credit Analysis by State Senate District")
    print("=" * 60)

    # Load block crosswalk for SLDU lookups
    print("\nLoading block crosswalk...")
    crosswalk = load_block_crosswalk()
    oregon_blocks = crosswalk[crosswalk["block_geoid"].str[:2] == "41"]
    print(f"  Oregon blocks: {len(oregon_blocks):,}")
    print(f"  Unique SLDUs: {oregon_blocks['sldu'].nunique()}")

    # Results accumulator
    results_by_sldu = {}

    print("\nProcessing Oregon congressional districts...")

    for cd_geoid in OREGON_CD_GEOIDS:
        cd_name = f"OR-{cd_geoid % 100:02d}"
        print(f"\n  Processing {cd_name}...")

        # Load district data
        h5_path = STORAGE_FOLDER / "districts" / f"{cd_name}.h5"
        if not h5_path.exists():
            print(f"    Skipping - file not found")
            continue

        # Run microsimulation for this district
        # Baseline
        baseline = Microsimulation(dataset=str(h5_path))
        baseline_ctc = baseline.calculate("or_ctc", 2024)
        baseline_weights = baseline.calculate("household_weight", 2024)

        # Reform: double the OR CTC max amounts
        # or_young_child_tax_credit_max is the parameter
        def double_or_ctc(parameters):
            # Double the max credit amount
            or_ctc = parameters.gov.states.or_.tax.income.credits.ctc
            or_ctc.amount.update(
                start=pd.Timestamp("2024-01-01"),
                stop=pd.Timestamp("2100-12-31"),
                value=or_ctc.amount("2024-01-01") * 2,
            )
            return parameters

        class DoubleORCTC(Reform):
            def apply(self):
                self.modify_parameters(double_or_ctc)

        reform = Microsimulation(dataset=str(h5_path), reform=DoubleORCTC)
        reform_ctc = reform.calculate("or_ctc", 2024)

        # Get number of households for block assignment
        n_households = len(baseline_weights)
        print(f"    Households: {n_households:,}")

        # Assign blocks and get SLDU for each household
        geo = assign_geography_for_cd(
            cd_geoid=str(cd_geoid),
            n_households=n_households,
            seed=cd_geoid,  # Reproducible
        )

        sldu_assignments = geo["sldu"]

        # Calculate impact per household
        impact = reform_ctc - baseline_ctc

        # Aggregate by SLDU
        unique_sldus = np.unique(sldu_assignments[sldu_assignments != ""])

        for sldu in unique_sldus:
            mask = sldu_assignments == sldu
            sldu_impact = np.sum(impact[mask] * baseline_weights[mask])
            sldu_baseline = np.sum(baseline_ctc[mask] * baseline_weights[mask])
            sldu_reform = np.sum(reform_ctc[mask] * baseline_weights[mask])
            sldu_hh = np.sum(mask)
            sldu_weighted_hh = np.sum(baseline_weights[mask])

            if sldu not in results_by_sldu:
                results_by_sldu[sldu] = {
                    "baseline_ctc": 0,
                    "reform_ctc": 0,
                    "impact": 0,
                    "households": 0,
                    "weighted_households": 0,
                }

            results_by_sldu[sldu]["baseline_ctc"] += sldu_baseline
            results_by_sldu[sldu]["reform_ctc"] += sldu_reform
            results_by_sldu[sldu]["impact"] += sldu_impact
            results_by_sldu[sldu]["households"] += sldu_hh
            results_by_sldu[sldu]["weighted_households"] += sldu_weighted_hh

    # Create results DataFrame
    print("\n" + "=" * 60)
    print("RESULTS: Impact of Doubling Oregon CTC by State Senate District")
    print("=" * 60)

    df = pd.DataFrame.from_dict(results_by_sldu, orient="index")
    df.index.name = "sldu"
    df = df.reset_index()

    # Convert to millions
    df["baseline_ctc_millions"] = df["baseline_ctc"] / 1e6
    df["reform_ctc_millions"] = df["reform_ctc"] / 1e6
    df["impact_millions"] = df["impact"] / 1e6

    # Sort by impact
    df = df.sort_values("impact_millions", ascending=False)

    # Display results
    print(
        f"\n{'SLDU':<8} {'Baseline':>12} {'Reform':>12} {'Impact':>12} {'Households':>12}"
    )
    print(f"{'':8} {'($M)':>12} {'($M)':>12} {'($M)':>12} {'(weighted)':>12}")
    print("-" * 60)

    for _, row in df.iterrows():
        print(
            f"{row['sldu']:<8} "
            f"{row['baseline_ctc_millions']:>12.2f} "
            f"{row['reform_ctc_millions']:>12.2f} "
            f"{row['impact_millions']:>12.2f} "
            f"{row['weighted_households']:>12,.0f}"
        )

    print("-" * 60)
    total_baseline = df["baseline_ctc_millions"].sum()
    total_reform = df["reform_ctc_millions"].sum()
    total_impact = df["impact_millions"].sum()
    total_hh = df["weighted_households"].sum()
    print(
        f"{'TOTAL':<8} {total_baseline:>12.2f} {total_reform:>12.2f} "
        f"{total_impact:>12.2f} {total_hh:>12,.0f}"
    )

    # Save to CSV
    output_path = Path("oregon_ctc_by_sldu.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return df


if __name__ == "__main__":
    run_oregon_ctc_analysis()
