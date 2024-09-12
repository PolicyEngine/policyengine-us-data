from io import BytesIO
import logging
from typing import List
from zipfile import ZipFile
import pandas as pd
from policyengine_core.data import Dataset
import requests
from tqdm import tqdm
from policyengine_us_data.data_storage import STORAGE_FOLDER

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
]


class RawACS(Dataset):
    name = "raw_acs"
    label = "Raw ACS"
    data_format = Dataset.TABLES
    years = []  # This will be populated as datasets are generated
    file_path = STORAGE_FOLDER / "raw_acs_{year}.h5"

    @staticmethod
    def file(year: int):
        return STORAGE_FOLDER / f"raw_acs_{year}.h5"

    def generate(self, year: int) -> None:
        year = int(year)
        if year in self.years:
            self.remove(year)

        spm_url = f"https://www2.census.gov/programs-surveys/supplemental-poverty-measure/datasets/spm/spm_{year}_pu.dta"
        person_url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_pus.zip"
        household_url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_hus.zip"

        try:
            with pd.HDFStore(self.file(year)) as storage:
                logging.info(f"Downloading household file")
                household = self.concat_zipped_csvs(
                    household_url, "psam_hus", HOUSEHOLD_COLUMNS
                )
                storage["household"] = household

                logging.info(f"Downloading person file")
                person = self.concat_zipped_csvs(
                    person_url, "psam_pus", PERSON_COLUMNS
                )
                storage["person"] = person

                logging.info(f"Downloading SPM unit file")
                spm_person = pd.read_stata(spm_url).fillna(0)
                spm_person.columns = spm_person.columns.str.upper()
                self.create_spm_unit_table(storage, spm_person)

            self.years.append(
                year
            )  # Add the year to the list of available years
            logging.info(f"Successfully generated Raw ACS data for {year}")
        except Exception as e:
            self.remove(year)
            logging.error(
                f"Attempted to extract and save the CSV files, but encountered an error: {e}"
            )
            raise e

    @staticmethod
    def concat_zipped_csvs(
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
            logging.info(f"Loading the first half of the dataset")
            a = pd.read_csv(zf.open(prefix + "a.csv"), usecols=columns)
            logging.info(f"Loading the second half of the dataset")
            b = pd.read_csv(zf.open(prefix + "b.csv"), usecols=columns)
        logging.info(f"Concatenating datasets")
        res = pd.concat([a, b]).fillna(0)
        res.columns = res.columns.str.upper()
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

        # Convert SERIALNO to string in both DataFrames
        JOIN_COLUMNS = ["SERIALNO", "SPORDER"]
        original_person_table[JOIN_COLUMNS] = original_person_table[
            JOIN_COLUMNS
        ].astype(int)
        person[JOIN_COLUMNS] = person[JOIN_COLUMNS].astype(int)

        # Add SPM_ID from the SPM person table to the original person table.
        combined_person_table = pd.merge(
            original_person_table,
            person[JOIN_COLUMNS + ["SPM_ID"]],
            on=JOIN_COLUMNS,
        )

        storage["person"] = combined_person_table
        storage["spm_unit"] = spm_table

    def load(self, year: int) -> dict:
        if not self.file(year).exists():
            raise FileNotFoundError(
                f"Raw ACS data for {year} not found. Please generate it first."
            )

        with pd.HDFStore(self.file(year), mode="r") as store:
            return {
                "person": store["person"],
                "household": store["household"],
                "spm_unit": store["spm_unit"],
            }

    def remove(self, year: int) -> None:
        if self.file(year).exists():
            self.file(year).unlink()
        if year in self.years:
            self.years.remove(year)
