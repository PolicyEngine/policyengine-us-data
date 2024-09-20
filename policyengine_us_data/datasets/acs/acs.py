import logging
from policyengine_core.data import Dataset
import h5py
from policyengine_us_data.datasets.acs.raw_acs import RawACS
from policyengine_us_data.storage import STORAGE_FOLDER
from pandas import DataFrame
import numpy as np
import os


class ACS(Dataset):
    name = "acs"
    label = "ACS"
    data_format = Dataset.ARRAYS
    time_period = None

    def __init__(self):
        super().__init__()
        self.raw_acs = RawACS()

    def generate(self) -> None:
        """Generates the ACS dataset."""
        if self.time_period is None:
            raise ValueError("time_period must be set in child classes")

        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        if self.time_period not in self.raw_acs.years:
            self.raw_acs.generate(self.time_period)

        raw_data = self.raw_acs.load(self.time_period)
        acs = h5py.File(self.file_path, mode="w")
        person, spm_unit, household = [
            raw_data[entity] for entity in ("person", "spm_unit", "household")
        ]

        self.add_id_variables(acs, person, spm_unit, household)
        self.add_person_variables(acs, person)
        self.add_spm_variables(acs, spm_unit)
        self.add_household_variables(acs, household)

        acs.close()

    @staticmethod
    def add_id_variables(
        acs: h5py.File,
        person: DataFrame,
        spm_unit: DataFrame,
        household: DataFrame,
    ) -> None:
        # Create numeric IDs based on SERIALNO
        person["numeric_id"] = person["SERIALNO"].astype("category").cat.codes
        household["numeric_id"] = (
            household["SERIALNO"].astype("category").cat.codes
        )

        acs["person_id"] = person["numeric_id"] * 100 + person.SPORDER.astype(
            int
        )
        acs["person_spm_unit_id"] = person.SPM_ID
        acs["spm_unit_id"] = spm_unit.index
        acs["tax_unit_id"] = (
            spm_unit.index
        )  # Using SPM unit as proxy for tax unit
        acs["family_id"] = spm_unit.index  # Using SPM unit as proxy for family
        acs["person_household_id"] = person["numeric_id"]
        acs["person_tax_unit_id"] = person.SPM_ID
        acs["person_family_id"] = person.SPM_ID
        acs["household_id"] = household["numeric_id"]
        acs["person_marital_unit_id"] = person["numeric_id"]
        acs["marital_unit_id"] = np.unique(person["numeric_id"])
        acs["person_weight"] = person.PWGTP
        acs["household_weight"] = household.WGTP

    @staticmethod
    def add_person_variables(acs: h5py.File, person: DataFrame) -> None:
        acs["age"] = person.AGEP
        acs["employment_income"] = person.WAGP
        acs["self_employment_income"] = person.SEMP
        acs["total_income"] = person.PINCP

    @staticmethod
    def add_spm_variables(acs: h5py.File, spm_unit: DataFrame) -> None:
        acs["spm_unit_net_income_reported"] = spm_unit.SPM_RESOURCES
        acs["spm_unit_spm_threshold"] = spm_unit.SPM_POVTHRESHOLD

    @staticmethod
    def add_household_variables(acs: h5py.File, household: DataFrame) -> None:
        acs["household_vehicles_owned"] = household.VEH
        acs["state_fips"] = acs["household_state_fips"] = household.ST.astype(
            int
        )


class ACS_2022(ACS):
    name = "acs_2022"
    label = "ACS 2022"
    time_period = 2022
    file_path = STORAGE_FOLDER / "acs_2022.h5"
    url = "release://PolicyEngine/policyengine-us-data/release/acs_2022.h5"
