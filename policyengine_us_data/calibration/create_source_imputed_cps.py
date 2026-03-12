"""Create source-imputed stratified extended CPS.

Standalone step that runs ACS/SIPP/SCF source imputations on the
stratified extended CPS, producing the dataset used by calibration
and H5 generation.

Usage:
    python policyengine_us_data/calibration/create_source_imputed_cps.py
"""

import logging
import sys
from pathlib import Path

import h5py

from policyengine_us_data.storage import STORAGE_FOLDER

logger = logging.getLogger(__name__)

INPUT_PATH = str(STORAGE_FOLDER / "stratified_extended_cps_2024.h5")
OUTPUT_PATH = str(STORAGE_FOLDER / "source_imputed_stratified_extended_cps_2024.h5")


def create_source_imputed_cps(
    input_path: str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
    seed: int = 42,
):
    from policyengine_us import Microsimulation
    from policyengine_us_data.calibration.clone_and_assign import (
        assign_random_geography,
    )
    from policyengine_us_data.calibration.source_impute import (
        impute_source_variables,
    )

    logger.info("Loading dataset from %s", input_path)
    sim = Microsimulation(dataset=input_path)
    n_records = len(sim.calculate("household_id", map_to="household").values)

    raw_keys = sim.dataset.load_dataset()["household_id"]
    if isinstance(raw_keys, dict):
        time_period = int(next(iter(raw_keys)))
    else:
        time_period = 2024

    logger.info("Loaded %d households, time_period=%d", n_records, time_period)

    geography = assign_random_geography(n_records=n_records, n_clones=1, seed=seed)
    base_states = geography.state_fips[:n_records]

    raw_data = sim.dataset.load_dataset()
    data_dict = {}
    for var in raw_data:
        val = raw_data[var]
        if isinstance(val, dict):
            data_dict[var] = {int(k) if k.isdigit() else k: v for k, v in val.items()}
        else:
            data_dict[var] = {time_period: val[...]}

    logger.info("Running source imputations...")
    data_dict = impute_source_variables(
        data=data_dict,
        state_fips=base_states,
        time_period=time_period,
        dataset_path=input_path,
    )

    logger.info("Saving to %s", output_path)
    with h5py.File(output_path, "w") as f:
        for var, time_dict in data_dict.items():
            for tp, values in time_dict.items():
                f.create_dataset(f"{var}/{tp}", data=values)

    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    create_source_imputed_cps()
