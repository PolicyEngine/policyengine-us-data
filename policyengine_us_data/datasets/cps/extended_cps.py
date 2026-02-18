import logging
from typing import Type

import numpy as np
from policyengine_core.data import Dataset

from policyengine_us_data.datasets.cps.cps import *  # noqa: F403
from policyengine_us_data.datasets.puf import *  # noqa: F403
from policyengine_us_data.storage import STORAGE_FOLDER

logger = logging.getLogger(__name__)


class ExtendedCPS(Dataset):
    cps: Type[CPS]
    puf: Type[PUF]
    data_format = Dataset.TIME_PERIOD_ARRAYS

    def generate(self):
        from policyengine_us import Microsimulation

        from policyengine_us_data.calibration.clone_and_assign import (
            load_global_block_distribution,
        )
        from policyengine_us_data.calibration.puf_impute import (
            puf_clone_dataset,
        )

        logger.info("Loading CPS dataset: %s", self.cps)
        cps_sim = Microsimulation(dataset=self.cps)
        data = cps_sim.dataset.load_dataset()
        del cps_sim

        data_dict = {}
        for var in data:
            data_dict[var] = {self.time_period: data[var][...]}

        n_hh = len(data_dict["household_id"][self.time_period])
        _, _, block_states, block_probs = load_global_block_distribution()
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(block_states), size=n_hh, p=block_probs)
        state_fips = block_states[indices]

        logger.info("PUF clone with dataset: %s", self.puf)
        new_data = puf_clone_dataset(
            data=data_dict,
            state_fips=state_fips,
            time_period=self.time_period,
            puf_dataset=self.puf,
            dataset_path=str(self.cps.file_path),
        )

        self.save_dataset(new_data)


class ExtendedCPS_2024(ExtendedCPS):
    cps = CPS_2024_Full
    puf = PUF_2024
    name = "extended_cps_2024"
    label = "Extended CPS (2024)"
    file_path = STORAGE_FOLDER / "extended_cps_2024.h5"
    time_period = 2024


if __name__ == "__main__":
    ExtendedCPS_2024().generate()
