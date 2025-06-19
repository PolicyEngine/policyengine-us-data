import numpy as np


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


if __name__ == "__main__":
    create_small_ecps()
    print("Small CPS dataset created successfully.")
