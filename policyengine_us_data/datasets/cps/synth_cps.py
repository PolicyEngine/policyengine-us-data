import numpy as np


def create_synth_cps():
    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets import EnhancedCPS_2024
    from policyengine_us_data.storage import STORAGE_FOLDER

    simulation = Microsimulation(
        dataset=EnhancedCPS_2024,
    )

    data = {}
    for variable in simulation.tax_benefit_system.variables:
        data[variable] = {}
        for time_period in simulation.get_holder(variable).get_known_periods():
            values = simulation.get_holder(variable).get_array(time_period)
            values = np.array(values)
            if values is not None:
                data[variable][time_period] = values
        if len(data[variable]) == 0:
            del data[variable]

    import h5py

    with h5py.File(STORAGE_FOLDER / "synthetic_cps_2024.h5", "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)
