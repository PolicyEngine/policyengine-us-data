from policyengine_core.data import Dataset
import h5py
from policyengine_us_data.datasets.acs.census_acs import CensusACS_2022
from policyengine_us_data.storage import STORAGE_FOLDER
from pandas import DataFrame
import numpy as np
import pandas as pd


class ACS(Dataset):
    data_format = Dataset.ARRAYS
    time_period = None
    census_acs = None

    def generate(self) -> None:
        """Generates the ACS dataset."""

        raw_data = self.census_acs(require=True).load()
        acs = h5py.File(self.file_path, mode="w")
        person, household = [raw_data[entity] for entity in ("person", "household")]

        self.add_id_variables(acs, person, household)
        self.add_person_variables(acs, person, household)
        self.add_household_variables(acs, person, household)

        acs.close()
        raw_data.close()

    @staticmethod
    def add_id_variables(
        acs: h5py.File,
        person: DataFrame,
        household: DataFrame,
    ) -> None:
        # Create numeric IDs based on SERIALNO
        h_id_to_number = pd.Series(
            np.arange(len(household)), index=household["SERIALNO"]
        )
        household["household_id"] = h_id_to_number[household["SERIALNO"]].values
        person["household_id"] = h_id_to_number[person["SERIALNO"]].values
        person["person_id"] = person.index + 1

        acs["person_id"] = person["person_id"]
        acs["household_id"] = household["household_id"]
        acs["spm_unit_id"] = acs["household_id"]
        acs["tax_unit_id"] = acs["household_id"]
        acs["family_id"] = acs["household_id"]
        acs["marital_unit_id"] = acs["household_id"]
        acs["person_household_id"] = person["household_id"]
        acs["person_spm_unit_id"] = person["household_id"]
        acs["person_tax_unit_id"] = person["household_id"]
        acs["person_family_id"] = person["household_id"]
        acs["person_marital_unit_id"] = person["household_id"]
        acs["household_weight"] = household.WGTP

    @staticmethod
    def add_person_variables(
        acs: h5py.File, person: DataFrame, household: DataFrame
    ) -> None:
        acs["age"] = person.AGEP
        acs["is_male"] = person.SEX == 1
        acs["employment_income"] = person.WAGP
        acs["self_employment_income"] = person.SEMP
        acs["social_security"] = person.SSP
        acs["taxable_private_pension_income"] = person.RETP
        person[["rent", "real_estate_taxes"]] = (
            household.set_index("household_id")
            .loc[person["household_id"]][["RNTP", "TAXAMT"]]
            .values
        )
        acs["is_household_head"] = person.SPORDER == 1
        factor = person.SPORDER == 1
        person.rent *= factor * 12
        person.real_estate_taxes *= factor
        acs["rent"] = person.rent
        acs["real_estate_taxes"] = person.real_estate_taxes
        acs["tenure_type"] = (
            household.TEN.astype(int)
            .map(
                {
                    1: "OWNED_WITH_MORTGAGE",
                    2: "OWNED_OUTRIGHT",
                    3: "RENTED",
                }
            )
            .fillna("NONE")
            .astype("S")
        )

    @staticmethod
    def add_spm_variables(acs: h5py.File, spm_unit: DataFrame) -> None:
        acs["spm_unit_net_income_reported"] = spm_unit.SPM_RESOURCES
        acs["spm_unit_spm_threshold"] = spm_unit.SPM_POVTHRESHOLD

    @staticmethod
    def add_household_variables(
        acs: h5py.File, person: DataFrame, household: DataFrame
    ) -> None:
        acs["household_vehicles_owned"] = household.VEH
        # ``state_fips`` is a person-level variable in policyengine-us;
        # broadcast the household-level ST assignment through the
        # person -> household mapping so lengths line up.
        household_state_fips = household.ST.astype(int)
        acs["household_state_fips"] = household_state_fips.values
        state_fips_by_household_id = pd.Series(
            household_state_fips.values, index=household["household_id"].values
        )
        acs["state_fips"] = state_fips_by_household_id.loc[
            person["household_id"].values
        ].values


class ACS_2022(ACS):
    name = "acs_2022"
    label = "ACS 2022"
    time_period = 2022
    file_path = STORAGE_FOLDER / "acs_2022.h5"
    census_acs = CensusACS_2022
    url = "release://PolicyEngine/policyengine-us-data/1.13.0/acs_2022.h5"


if __name__ == "__main__":
    ACS_2022().generate()
