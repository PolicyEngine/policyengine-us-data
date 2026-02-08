from policyengine_core.data import Dataset
import numpy as np
from typing import Type
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.extended_cps import (
    ExtendedCPS_2024,
)
import logging


class EnhancedCPS(Dataset):
    data_format = Dataset.TIME_PERIOD_ARRAYS
    input_dataset: Type[Dataset]
    start_year: int
    end_year: int

    def generate(self):
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=self.input_dataset)
        data = sim.dataset.load_dataset()
        data["household_weight"] = {}
        original_weights = sim.calculate("household_weight")
        original_weights = original_weights.values + np.random.normal(
            1, 0.1, len(original_weights)
        )

        for year in range(self.start_year, self.end_year + 1):
            optimised_weights = self._calibrate(sim, original_weights, year)
            data["household_weight"][year] = optimised_weights

        self.save_dataset(data)

    def _calibrate(self, sim, original_weights, year):
        """Run national calibration for one year.

        Reads active targets from policy_data.db via
        NationalMatrixBuilder, then fits weights using
        l0-python's SparseCalibrationWeights.

        Args:
            sim: Microsimulation instance.
            original_weights: Jittered household weights.
            year: Tax year to calibrate.

        Returns:
            Optimised weight array (n_households,).
        """
        from policyengine_us_data.calibration.fit_national_weights import (
            build_calibration_inputs,
            fit_national_weights,
            initialize_weights,
        )

        db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
        matrix, targets, names = build_calibration_inputs(
            dataset_class=self.input_dataset,
            time_period=year,
            db_path=str(db_path),
            sim=sim,
        )

        init_weights = initialize_weights(original_weights)
        optimised_weights = fit_national_weights(
            matrix=matrix,
            targets=targets,
            initial_weights=init_weights,
            epochs=500,
        )

        logging.info(
            f"Calibration for {year}: "
            f"{len(targets)} targets, "
            f"{(optimised_weights > 0).sum():,} non-zero weights"
        )
        return optimised_weights


class EnhancedCPS_2024(EnhancedCPS):
    input_dataset = ExtendedCPS_2024
    start_year = 2024
    end_year = 2024
    name = "enhanced_cps_2024"
    label = "Enhanced CPS 2024"
    file_path = STORAGE_FOLDER / "enhanced_cps_2024.h5"
    url = "hf://policyengine/policyengine-us-data/" "enhanced_cps_2024.h5"


if __name__ == "__main__":
    EnhancedCPS_2024().generate()
