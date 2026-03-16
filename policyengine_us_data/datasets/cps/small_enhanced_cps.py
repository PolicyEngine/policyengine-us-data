import os
import pandas as pd
import numpy as np
import h5py

from policyengine_us import Microsimulation
from policyengine_us_data.datasets import EnhancedCPS_2024
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_core.enums import Enum
from policyengine_core.data.dataset import Dataset
import logging


def create_small_ecps():
    """Create a small version of the ECPS by random sampling"""
    simulation = Microsimulation(
        dataset=EnhancedCPS_2024,
    )
    simulation.subsample(1_000)

    # Basic validation that subsample has reasonable data
    weights = simulation.calculate("household_weight").values
    if np.all(weights == 0):
        raise ValueError(
            "create_small_ecps: all household weights are zero after subsample"
        )
    logging.info(
        f"create_small_ecps: subsample has "
        f"{len(weights)} households, "
        f"{int(np.sum(weights > 0))} with non-zero weight"
    )

    data = {}
    for variable in simulation.tax_benefit_system.variables:
        data[variable] = {}
        for time_period in simulation.get_holder(variable).get_known_periods():
            values = simulation.get_holder(variable).get_array(time_period)
            if simulation.tax_benefit_system.variables.get(
                variable
            ).value_type in (
                Enum,
                str,
            ):
                if hasattr(values, "decode_to_str"):
                    values = values.decode_to_str().astype("S")
                else:
                    values = values.astype("S")
            else:
                values = np.array(values)
            if values is not None:
                data[variable][time_period] = values

        if len(data[variable]) == 0:
            del data[variable]

    with h5py.File(STORAGE_FOLDER / "small_enhanced_cps_2024.h5", "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)


def create_sparse_ecps():
    """create a small version of the ECPS by L0 regularization"""
    time_period = 2024

    ecps = EnhancedCPS_2024()
    h5 = ecps.load()
    sparse_weights = h5["household_weight"][str(time_period)][:]
    hh_ids = h5["household_id"][str(time_period)][:]
    h5.close()

    template_sim = Microsimulation(
        dataset=EnhancedCPS_2024,
    )
    template_sim.set_input("household_weight", time_period, sparse_weights)

    df = template_sim.to_input_dataframe()
    del template_sim

    household_weight_column = f"household_weight__{time_period}"
    df_household_id_column = f"household_id__{time_period}"
    df_person_id_column = f"person_id__{time_period}"

    # Group by household ID and get the first entry for each group
    h_df = df.groupby(df_household_id_column).first()
    h_ids = pd.Series(h_df.index)
    h_weights = pd.Series(h_df[household_weight_column].values)

    # Seed the random number generators for reproducibility
    h_ids = h_ids[h_weights > 0]
    h_weights = h_weights[h_weights > 0]

    if len(h_ids) < 1000:
        raise ValueError(
            f"create_sparse_ecps: only {len(h_ids)} households with "
            f"non-zero weight (expected > 1000)"
        )
    logging.info(
        f"create_sparse_ecps: {len(h_ids)} households after zero-weight filtering"
    )

    subset_df = df[df[df_household_id_column].isin(h_ids)].copy()

    # Update the dataset and rebuild the simulation
    sim = Microsimulation()
    sim.dataset = Dataset.from_dataframe(subset_df, time_period)
    sim.build_from_dataset()

    # Write the data to an h5
    data = {}
    for variable in sim.tax_benefit_system.variables:
        data[variable] = {}
        for time_period in sim.get_holder(variable).get_known_periods():
            values = sim.get_holder(variable).get_array(time_period)
            if (
                sim.tax_benefit_system.variables.get(variable).value_type
                in (Enum, str)
                and variable != "county_fips"
            ):
                values = values.decode_to_str().astype("S")
            elif variable == "county_fips":
                values = values.astype("int32")
            else:
                values = np.array(values)
            if values is not None:
                data[variable][time_period] = values

        if len(data[variable]) == 0:
            del data[variable]

    # Validate critical variables exist before writing
    critical_vars = [
        "household_weight",
        "employment_income_before_lsr",
        "household_id",
        "person_id",
    ]
    missing = [v for v in critical_vars if v not in data]
    if missing:
        raise ValueError(
            f"create_sparse_ecps: missing critical variables: {missing}"
        )
    logging.info(f"create_sparse_ecps: data dict has {len(data)} variables")

    output_path = STORAGE_FOLDER / "sparse_enhanced_cps_2024.h5"
    with h5py.File(output_path, "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)

    file_size = os.path.getsize(output_path)
    if file_size < 1_000_000:
        raise ValueError(
            f"create_sparse_ecps: output file only {file_size:,} bytes (expected > 1MB)"
        )
    logging.info(
        f"create_sparse_ecps: wrote {file_size / 1e6:.1f}MB to {output_path}"
    )


if __name__ == "__main__":
    create_small_ecps()
    logging.info("Small CPS dataset created successfully.")
    create_sparse_ecps()
    logging.info("Sparse CPS dataset created successfully.")
