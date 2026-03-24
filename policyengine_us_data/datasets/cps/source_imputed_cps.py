from typing import Type

from policyengine_core.data import Dataset

from policyengine_us_data.calibration.create_source_imputed_cps import (
    create_source_imputed_cps,
)
from policyengine_us_data.calibration.create_stratified_cps import (
    create_stratified_cps_dataset,
)
from policyengine_us_data.datasets.cps.cps import CPS_2024
from policyengine_us_data.datasets.cps.extended_cps import ExtendedCPS_2024
from policyengine_us_data.storage import STORAGE_FOLDER


class StratifiedExtendedCPS(Dataset):
    data_format = Dataset.TIME_PERIOD_ARRAYS
    base_dataset: Type[Dataset]
    target_households = 30_000
    high_income_percentile = 99
    oversample_poor = False
    seed = None

    def generate(self):
        self.base_dataset(require=True)
        create_stratified_cps_dataset(
            target_households=self.target_households,
            high_income_percentile=self.high_income_percentile,
            oversample_poor=self.oversample_poor,
            seed=self.seed,
            base_dataset=str(self.base_dataset.file_path),
            output_path=str(self.file_path),
        )


class StratifiedExtendedCPS_2024(StratifiedExtendedCPS):
    base_dataset = ExtendedCPS_2024
    name = "stratified_extended_cps_2024"
    label = "Stratified Extended CPS (2024)"
    file_path = STORAGE_FOLDER / "stratified_extended_cps_2024.h5"
    time_period = 2024


class SourceImputedDataset(Dataset):
    data_format = Dataset.TIME_PERIOD_ARRAYS
    input_dataset: Type[Dataset]
    seed = 42
    use_existing_state_fips = False

    def generate(self):
        self.input_dataset(require=True)
        create_source_imputed_cps(
            input_path=str(self.input_dataset.file_path),
            output_path=str(self.file_path),
            seed=self.seed,
            use_existing_state_fips=self.use_existing_state_fips,
            time_period=self.time_period,
        )


class SourceImputedCPS(SourceImputedDataset):
    use_existing_state_fips = True


class SourceImputedCPS_2024(SourceImputedCPS):
    input_dataset = CPS_2024
    name = "source_imputed_cps_2024"
    label = "Source-Imputed CPS (2024)"
    file_path = STORAGE_FOLDER / "source_imputed_cps_2024.h5"
    time_period = 2024


class SourceImputedStratifiedExtendedCPS(SourceImputedDataset):
    pass


class SourceImputedStratifiedExtendedCPS_2024(SourceImputedStratifiedExtendedCPS):
    input_dataset = StratifiedExtendedCPS_2024
    name = "source_imputed_stratified_extended_cps_2024"
    label = "Source-Imputed Stratified Extended CPS (2024)"
    file_path = STORAGE_FOLDER / "source_imputed_stratified_extended_cps_2024.h5"
    url = "hf://policyengine/policyengine-us-data/calibration/source_imputed_stratified_extended_cps.h5"
    time_period = 2024
