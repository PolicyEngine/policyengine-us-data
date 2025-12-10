import logging
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
        person, household = [
            raw_data[entity] for entity in ("person", "household")
        ]

        self.add_id_variables(acs, person, household)
        self.add_person_variables(acs, person, household)
        self.add_household_variables(acs, household)
        self.add_spm_variables(acs, person, household, self.time_period)

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
        household["household_id"] = h_id_to_number[
            household["SERIALNO"]
        ].values
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
    def add_spm_variables(
        acs: h5py.File,
        person: DataFrame,
        household: DataFrame,
        time_period: int,
    ) -> None:
        from policyengine_us_data.utils.spm import (
            calculate_spm_thresholds_national,
            map_tenure_acs_to_spm,
        )

        # In ACS, SPM unit = household
        # Calculate number of adults (18+) and children (<18) per household
        person_with_hh = person.copy()
        person_with_hh["is_adult"] = person_with_hh["AGEP"] >= 18
        person_with_hh["is_child"] = person_with_hh["AGEP"] < 18

        hh_counts = (
            person_with_hh.groupby("household_id")
            .agg({"is_adult": "sum", "is_child": "sum"})
            .rename(
                columns={"is_adult": "num_adults", "is_child": "num_children"}
            )
        )

        # Ensure household is indexed properly
        household_indexed = household.set_index("household_id")

        # Get counts aligned with household order
        num_adults = hh_counts.loc[
            household_indexed.index, "num_adults"
        ].values
        num_children = hh_counts.loc[
            household_indexed.index, "num_children"
        ].values

        # Map ACS tenure to SPM tenure codes
        tenure_codes = map_tenure_acs_to_spm(household_indexed["TEN"].values)

        # Calculate SPM thresholds using national-level values (no geographic
        # adjustment). Geographic adjustments will be applied later in the
        # pipeline when households are assigned to specific areas.
        acs["spm_unit_spm_threshold"] = calculate_spm_thresholds_national(
            num_adults=num_adults,
            num_children=num_children,
            tenure_codes=tenure_codes,
            year=time_period,
        )

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
    census_acs = CensusACS_2022
    url = "release://PolicyEngine/policyengine-us-data/1.13.0/acs_2022.h5"


if __name__ == "__main__":
    ACS_2022().generate()
