from policyengine_core.data import Dataset
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.scf.fed_scf import (
    FedSCF,
    FedSCF_2016,
    FedSCF_2019,
    FedSCF_2022,
)
import pandas as pd
import numpy as np
import os
from typing import List, Optional, Union, Type


class SCF(Dataset):
    """Dataset containing processed Survey of Consumer Finances data."""

    name = "scf"
    label = "SCF"
    raw_scf: Type[FedSCF] = None
    data_format = Dataset.TABLES

    def generate(self):
        """Generates the SCF dataset for PolicyEngine US microsimulations.

        Downloads the raw SCF data and processes it for use in PolicyEngine.
        """
        if self.raw_scf is None:
            raise ValueError("No raw SCF data class specified.")

        # Initialize raw data instance
        raw_scf_instance = self.raw_scf()

        # Check if raw data exists, if not, generate it
        if not os.path.exists(raw_scf_instance.file_path):
            print(f"Raw SCF data not found. Generating from source.")
            raw_scf_instance.generate()

        # Open the raw data file
        with pd.HDFStore(raw_scf_instance.file_path, mode="r") as raw_storage:
            # Read the raw data
            raw_data = raw_storage["data"]

            # Create a new HDF storage for this dataset
            with pd.HDFStore(self.file_path, mode="w") as storage:
                # Store the raw data as is for now
                # In the future, preprocessing steps will be added here
                storage["data"] = raw_data

        print(f"SCF dataset for {self.time_period} has been generated.")

    def load(self):
        """Loads the SCF dataset.

        Returns:
            pandas.DataFrame: The SCF data.
        """
        # Check if file exists
        if not os.path.exists(self.file_path):
            print(f"SCF dataset file not found. Generating it.")
            self.generate()

        with pd.HDFStore(self.file_path, mode="r") as storage:
            return storage["data"]


class SCF_2022(SCF):
    """SCF dataset for 2022."""

    name = "scf_2022"
    label = "SCF 2022"
    raw_scf = FedSCF_2022
    file_path = STORAGE_FOLDER / "scf_2022.h5"
    time_period = 2022


class SCF_2019(SCF):
    """SCF dataset for 2019."""

    name = "scf_2019"
    label = "SCF 2019"
    raw_scf = FedSCF_2019
    file_path = STORAGE_FOLDER / "scf_2019.h5"
    time_period = 2019


class SCF_2016(SCF):
    """SCF dataset for 2016."""

    name = "scf_2016"
    label = "SCF 2016"
    raw_scf = FedSCF_2016
    file_path = STORAGE_FOLDER / "scf_2016.h5"
    time_period = 2016


if __name__ == "__main__":
    # Generate all SCF datasets
    SCF_2016().generate()
    SCF_2019().generate()
    SCF_2022().generate()
