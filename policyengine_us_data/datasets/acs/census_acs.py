from io import BytesIO
import logging
from typing import List
from zipfile import ZipFile
import pandas as pd
from policyengine_core.data import Dataset
import requests
from tqdm import tqdm
from policyengine_us_data.storage import STORAGE_FOLDER

logging.getLogger().setLevel(logging.INFO)

PERSON_COLUMNS = [
    "SERIALNO",  # Household ID
    "SPORDER",  # Person number within household
    "PWGTP",  # Person weight
    "AGEP",  # Age
    "CIT",  # Citizenship
    "MAR",  # Marital status
    "WAGP",  # Wage/salary
    "SSP",  # Social security income
    "SSIP",  # Supplemental security income
    "SEX",  # Sex
    "SEMP",  # Self-employment income
    "SCHL",  # Educational attainment
    "RETP",  # Retirement income
    "PAP",  # Public assistance income
    "OIP",  # Other income
    "PERNP",  # Total earnings
    "PINCP",  # Total income
    "POVPIP",  # Income-to-poverty line percentage
    "RAC1P",  # Race
]

HOUSEHOLD_COLUMNS = [
    "SERIALNO",  # Household ID
    "PUMA",  # PUMA area code
    "ST",  # State code
    "ADJHSG",  # Adjustment factor for housing dollar amounts
    "ADJINC",  # Adjustment factor for income
    "WGTP",  # Household weight
    "NP",  # Number of persons in household
    "BDSP",  # Number of bedrooms
    "ELEP",  # Electricity monthly cost
    "FULP",  # Fuel monthly cost
    "GASP",  # Gas monthly cost
    "RMSP",  # Number of rooms
    "RNTP",  # Monthly rent
    "TEN",  # Tenure
    "VEH",  # Number of vehicles
    "FINCP",  # Total income
    "GRNTP",  # Gross rent
    "TAXAMT",  # Property taxes
]


class CensusACS(Dataset):
    data_format = Dataset.TABLES

    def generate(self) -> None:
        spm_url = f"https://www2.census.gov/programs-surveys/supplemental-poverty-measure/datasets/spm/spm_{self.time_period}_pu.dta"
        person_url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{self.time_period}/1-Year/csv_pus.zip"
        household_url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{self.time_period}/1-Year/csv_hus.zip"

        with pd.HDFStore(self.file_path, mode="w") as storage:
            household = self.process_household_data(
                household_url, "psam_hus", HOUSEHOLD_COLUMNS
            )
            person = self.process_person_data(
                person_url, "psam_pus", PERSON_COLUMNS
            )
            person = person[person.SERIALNO.isin(household.SERIALNO)]
            household = household[household.SERIALNO.isin(person.SERIALNO)]
            storage["household"] = household
            storage["person"] = person

    @staticmethod
    def process_household_data(
        url: str, prefix: str, columns: List[str]
    ) -> pd.DataFrame:
        req = requests.get(url, stream=True)
        with BytesIO() as f:
            pbar = tqdm()
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    pbar.update(len(chunk))
                    f.write(chunk)
            f.seek(0)
            zf = ZipFile(f)
            a = pd.read_csv(
                zf.open(prefix + "a.csv"),
                usecols=columns,
                dtype={"SERIALNO": str},
            )
            b = pd.read_csv(
                zf.open(prefix + "b.csv"),
                usecols=columns,
                dtype={"SERIALNO": str},
            )
        res = pd.concat([a, b]).fillna(0)
        res.columns = res.columns.str.upper()

        # Ensure correct data types
        res["ST"] = res["ST"].astype(int)

        return res

    @staticmethod
    def process_person_data(
        url: str, prefix: str, columns: List[str]
    ) -> pd.DataFrame:
        req = requests.get(url, stream=True)
        with BytesIO() as f:
            pbar = tqdm()
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    pbar.update(len(chunk))
                    f.write(chunk)
            f.seek(0)
            zf = ZipFile(f)
            a = pd.read_csv(
                zf.open(prefix + "a.csv"),
                usecols=columns,
                dtype={"SERIALNO": str},
            )
            b = pd.read_csv(
                zf.open(prefix + "b.csv"),
                usecols=columns,
                dtype={"SERIALNO": str},
            )
        res = pd.concat([a, b]).fillna(0)
        res.columns = res.columns.str.upper()

        # Ensure correct data types
        res["SPORDER"] = res["SPORDER"].astype(int)

        return res

    @staticmethod
    def create_spm_unit_table(
        storage: pd.HDFStore, person: pd.DataFrame
    ) -> None:
        SPM_UNIT_COLUMNS = [
            "CAPHOUSESUB",
            "CAPWKCCXPNS",
            "CHILDCAREXPNS",
            "EITC",
            "ENGVAL",
            "EQUIVSCALE",
            "FEDTAX",
            "FEDTAXBC",
            "FICA",
            "GEOADJ",
            "MEDXPNS",
            "NUMADULTS",
            "NUMKIDS",
            "NUMPER",
            "POOR",
            "POVTHRESHOLD",
            "RESOURCES",
            "SCHLUNCH",
            "SNAPSUB",
            "STTAX",
            "TENMORTSTATUS",
            "TOTVAL",
            "WCOHABIT",
            "WICVAL",
            "WKXPNS",
            "WUI_LT15",
            "ID",
        ]
        spm_table = (
            person[["SPM_" + column for column in SPM_UNIT_COLUMNS]]
            .groupby(person.SPM_ID)
            .first()
        )

        original_person_table = storage["person"]
        original_person_table.to_csv("person.csv")
        person.to_csv("spm_person.csv")

        # Ensure SERIALNO is treated as string
        JOIN_COLUMNS = ["SERIALNO", "SPORDER"]
        original_person_table["SERIALNO"] = original_person_table[
            "SERIALNO"
        ].astype(str)
        original_person_table["SPORDER"] = original_person_table[
            "SPORDER"
        ].astype(int)
        person["SERIALNO"] = person["SERIALNO"].astype(str)
        person["SPORDER"] = person["SPORDER"].astype(int)

        # Add SPM_ID from the SPM person table to the original person table.
        combined_person_table = pd.merge(
            original_person_table,
            person[JOIN_COLUMNS + ["SPM_ID"]],
            on=JOIN_COLUMNS,
        )

        storage["person_matched"] = combined_person_table
        storage["spm_unit"] = spm_table


class CensusACS_2022(CensusACS):
    label = "Census ACS (2022)"
    name = "census_acs_2022.h5"
    file_path = STORAGE_FOLDER / "census_acs_2022.h5"
    time_period = 2022
