import pandas as pd
import numpy as np

from policyengine_core.data.dataset import Dataset


def create_small_ecps():
    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets import EnhancedCPS_2024
    from policyengine_us_data.storage import STORAGE_FOLDER
    from policyengine_core.enums import Enum

    simulation = Microsimulation(
        dataset=EnhancedCPS_2024,
    )
    simulation.subsample(1_000)

    data = {}
    for variable in simulation.tax_benefit_system.variables:
        data[variable] = {}
        for time_period in simulation.get_holder(variable).get_known_periods():
            values = simulation.get_holder(variable).get_array(time_period)
            values = np.array(values)
            if simulation.tax_benefit_system.variables.get(
                variable
            ).value_type in (Enum, str):
                values = values.astype("S")
            if values is not None:
                data[variable][time_period] = values

        if len(data[variable]) == 0:
            del data[variable]

    import h5py

    with h5py.File(STORAGE_FOLDER / "small_enhanced_cps_2024.h5", "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)


def create_sparse_ecps():
    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets import SparseEnhancedCPS_2024
    from policyengine_us_data.storage import STORAGE_FOLDER
    from policyengine_core.enums import Enum

    time_period = 2024

    ecps = SparseEnhancedCPS_2024()

    # Check if sparse weights exist, if not generate them
    try:
        h5 = ecps.load()
        sparse_weights = h5["household_sparse_weight"]["2024"][:]
        hh_ids = h5["household_id"]["2024"][:]
    except KeyError:
        print(
            "Sparse weights not found. Generating SparseEnhancedCPS_2024 dataset..."
        )
        ecps.generate()
        h5 = ecps.load()
        sparse_weights = h5["household_sparse_weight"]["2024"][:]
        hh_ids = h5["household_id"]["2024"][:]

    template_sim = Microsimulation(
        dataset=EnhancedCPS_2024,
    )
    template_sim.set_input("household_weight", 2024, sparse_weights)

    template_df = template_sim.to_input_dataframe()

    household_weight_column = f"household_weight__{time_period}"
    df_household_id_column = f"household_id__{time_period}"
    df_person_id_column = f"person_id__{time_period}"

    # Group by household ID and get the first entry for each group
    df = template_df
    h_df = df.groupby(df_household_id_column).first()
    h_ids = pd.Series(h_df.index)
    h_weights = pd.Series(h_df[household_weight_column].values)

    # Seed the random number generators for reproducibility
    h_ids = h_ids[h_weights > 0]
    h_weights = h_weights[h_weights > 0]

    subset_df = df[df[df_household_id_column].isin(h_ids)].copy()

    household_id_to_count = {}
    for household_id in h_ids:
        if household_id not in household_id_to_count:
            household_id_to_count[household_id] = 0
        household_id_to_count[household_id] += 1

    household_counts = subset_df[df_household_id_column].map(
        lambda x: household_id_to_count.get(x, 0)
    )

    # NOTE: from subsample. I don't think I want to do this!
    ## Adjust household weights to maintain the total weight
    # for col in subset_df.columns:
    #    if "weight__" in col:
    #        target_total_weight = df[col].values.sum()
    #        if not quantize_weights:
    #            subset_df[col] *= household_counts.values
    #        else:
    #            subset_df[col] = household_counts.values
    #        subset_df[col] *= (
    #            target_total_weight / subset_df[col].values.sum()
    #        )

    df = subset_df

    # Update the dataset and rebuild the simulation
    sim = Microsimulation()
    sim.dataset = Dataset.from_dataframe(df, sim.dataset.time_period)
    sim.build_from_dataset()

    # Ensure the baseline branch has the new data.
    if "baseline" in sim.branches:
        baseline_tax_benefit_system = sim.branches[
            "baseline"
        ].tax_benefit_system
        sim.branches["baseline"] = sim.clone()
        sim.branches["tax_benefit_system"] = baseline_tax_benefit_system

    sim.default_calculation_period = time_period

    # Get ready to write it out
    simulation = sim
    data = {}
    for variable in simulation.tax_benefit_system.variables:
        data[variable] = {}
        for time_period in simulation.get_holder(variable).get_known_periods():
            values = simulation.get_holder(variable).get_array(time_period)
            values = np.array(values)
            if simulation.tax_benefit_system.variables.get(
                variable
            ).value_type in (Enum, str):
                values = values.astype("S")
            if values is not None:
                data[variable][time_period] = values

        if len(data[variable]) == 0:
            del data[variable]

    import h5py

    with h5py.File(STORAGE_FOLDER / "sparse_enhanced_cps_2024.h5", "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)


if __name__ == "__main__":
    create_small_ecps()
    print("Small CPS dataset created successfully.")
    create_sparse_ecps()
    print("Sparse CPS dataset created successfully.")
