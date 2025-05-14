from policyengine_core.data import Dataset
from tqdm import tqdm
import requests
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
from policyengine_us_data.storage import STORAGE_FOLDER


class CensusCPS(Dataset):
    """Dataset containing CPS ASEC tables in the Census format."""

    time_period: int
    """Year of the dataset."""

    def generate(self):
        if self._cps_download_url is None:
            raise ValueError(
                f"No raw CPS data URL known for year {self.time_period}."
            )

        url = self._cps_download_url

        spm_unit_columns = SPM_UNIT_COLUMNS
        if self.time_period <= 2020:
            spm_unit_columns = [
                col for col in spm_unit_columns if col != "SPM_BBSUBVAL"
            ]

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(
            response.headers.get("content-length", 200e6)
        )
        progress_bar = tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc="Downloading ASEC",
        )
        if response.status_code == 404:
            raise FileNotFoundError(
                "Received a 404 response when fetching the data."
            )
        with BytesIO() as file:
            content_length_actual = 0
            for data in response.iter_content(int(1e6)):
                progress_bar.update(len(data))
                content_length_actual += len(data)
                file.write(data)
            progress_bar.set_description("Downloaded ASEC")
            progress_bar.total = content_length_actual
            progress_bar.close()
            zipfile = ZipFile(file)
            spm_unit_columns = SPM_UNIT_COLUMNS
            if self.time_period <= 2020:
                spm_unit_columns = [
                    col for col in spm_unit_columns if col != "SPM_BBSUBVAL"
                ]
            with pd.HDFStore(self.file_path, mode="w") as storage:
                file_year = int(self.time_period) + 1
                file_year_code = str(file_year)[-2:]
                if file_year_code == "19":
                    # In the 2018 CPS, the file is within prod/data/2019
                    # instead of at the top level.
                    file_prefix = "cpspb/asec/prod/data/2019/"
                else:
                    file_prefix = ""
                with zipfile.open(
                    f"{file_prefix}pppub{file_year_code}.csv"
                ) as f:
                    storage["person"] = pd.read_csv(
                        f,
                        usecols=PERSON_COLUMNS
                        + spm_unit_columns
                        + TAX_UNIT_COLUMNS,
                    ).fillna(0)
                    person = storage["person"]
                with zipfile.open(
                    f"{file_prefix}ffpub{file_year_code}.csv"
                ) as f:
                    person_family_id = person.PH_SEQ * 10 + person.PF_SEQ
                    family = pd.read_csv(f).fillna(0)
                    family_id = family.FH_SEQ * 10 + family.FFPOS
                    family = family[family_id.isin(person_family_id)]
                    storage["family"] = family
                with zipfile.open(
                    f"{file_prefix}hhpub{file_year_code}.csv"
                ) as f:
                    person_household_id = person.PH_SEQ
                    household = pd.read_csv(f).fillna(0)
                    household_id = household.H_SEQ
                    household = household[
                        household_id.isin(person_household_id)
                    ]
                    storage["household"] = household
                storage["tax_unit"] = self._create_tax_unit_table(person)
                storage["spm_unit"] = self._create_spm_unit_table(
                    person, self.time_period
                )

    @property
    def _cps_download_url(self) -> str:
        return CPS_URL_BY_YEAR.get(self.time_period)

    def _create_tax_unit_table(self, person: pd.DataFrame) -> pd.DataFrame:
        tax_unit_df = person[TAX_UNIT_COLUMNS].groupby(person.TAX_ID).sum()
        tax_unit_df["TAX_ID"] = tax_unit_df.index
        return tax_unit_df

    def _create_spm_unit_table(
        self, person: pd.DataFrame, time_period: int
    ) -> pd.DataFrame:
        spm_unit_columns = SPM_UNIT_COLUMNS
        if time_period <= 2020:
            spm_unit_columns = [
                col for col in spm_unit_columns if col != "SPM_BBSUBVAL"
            ]
        return person[spm_unit_columns].groupby(person.SPM_ID).first()


class CensusCPS_2023(CensusCPS):
    time_period = 2023
    label = "Census CPS (2023)"
    name = "census_cps_2023"
    file_path = STORAGE_FOLDER / "census_cps_2023.h5"
    data_format = Dataset.TABLES


class CensusCPS_2022(CensusCPS):
    time_period = 2022
    label = "Census CPS (2022)"
    name = "census_cps_2022"
    file_path = STORAGE_FOLDER / "census_cps_2022.h5"
    data_format = Dataset.TABLES


class CensusCPS_2021(CensusCPS):
    time_period = 2021
    label = "Census CPS (2021)"
    name = "census_cps_2021"
    file_path = STORAGE_FOLDER / "census_cps_2021.h5"
    data_format = Dataset.TABLES


class CensusCPS_2020(CensusCPS):
    time_period = 2020
    label = "Census CPS (2020)"
    name = "census_cps_2020"
    file_path = STORAGE_FOLDER / "census_cps_2020.h5"
    data_format = Dataset.TABLES


class CensusCPS_2019(CensusCPS):
    time_period = 2019
    label = "Census CPS (2019)"
    name = "census_cps_2019"
    file_path = STORAGE_FOLDER / "census_cps_2019.h5"
    data_format = Dataset.TABLES


class CensusCPS_2018(CensusCPS):
    time_period = 2018
    label = "Census CPS (2018)"
    name = "census_cps_2018"
    file_path = STORAGE_FOLDER / "census_cps_2018.h5"
    data_format = Dataset.TABLES


CPS_URL_BY_YEAR = {
    2018: "https://www2.census.gov/programs-surveys/cps/datasets/2019/march/asecpub19csv.zip",
    2019: "https://www2.census.gov/programs-surveys/cps/datasets/2020/march/asecpub20csv.zip",
    2020: "https://www2.census.gov/programs-surveys/cps/datasets/2021/march/asecpub21csv.zip",
    2021: "https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asecpub22csv.zip",
    2022: "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
    2023: "https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asecpub24csv.zip",
}


TAX_UNIT_COLUMNS = [
    "ACTC_CRD",
    "AGI",
    "CTC_CRD",
    "EIT_CRED",
    "FEDTAX_AC",
    "FEDTAX_BC",
    "MARG_TAX",
    "STATETAX_A",
    "STATETAX_B",
    "TAX_INC",
]

SPM_UNIT_COLUMNS = [
    "ACTC",
    "BBSUBVAL",
    "CAPHOUSESUB",
    "CAPWKCCXPNS",
    "CHILDCAREXPNS",
    "CHILDSUPPD",
    "EITC",
    "ENGVAL",
    "EQUIVSCALE",
    "FAMTYPE",
    "FEDTAX",
    "FEDTAXBC",
    "FICA",
    "GEOADJ",
    "HAGE",
    "HHISP",
    "HMARITALSTATUS",
    "HRACE",
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
    "WEIGHT",
    "WFOSTER22",
    "WICVAL",
    "WKXPNS",
    "WNEWHEAD",
    "WNEWPARENT",
    "WUI_LT15",
    "ID",
]
SPM_UNIT_COLUMNS = ["SPM_" + column for column in SPM_UNIT_COLUMNS]
PERSON_COLUMNS = [
    "PH_SEQ",
    "PF_SEQ",
    "P_SEQ",
    "TAX_ID",
    "SPM_ID",
    "A_FNLWGT",
    "A_LINENO",
    "A_SPOUSE",
    "A_AGE",
    "A_SEX",
    "PEDISEYE",
    "MRK",
    "WSAL_VAL",
    "INT_VAL",
    "SEMP_VAL",
    "FRSE_VAL",
    "DIV_VAL",
    "RNT_VAL",
    "SS_VAL",
    "UC_VAL",
    "ANN_VAL",
    "PNSN_VAL",
    "OI_OFF",
    "OI_VAL",
    "CSP_VAL",
    "PAW_VAL",
    "SSI_VAL",
    "RETCB_VAL",
    "CAP_VAL",
    "WICYN",
    "VET_VAL",
    "WC_VAL",
    "DIS_VAL1",
    "DIS_VAL2",
    "CHSP_VAL",
    "PHIP_VAL",
    "MOOP",
    "PEDISDRS",
    "PEDISEAR",
    "PEDISOUT",
    "PEDISPHY",
    "PEDISREM",
    "PEPAR1",
    "PEPAR2",
    "DIS_SC1",
    "DIS_SC2",
    "DST_SC1",
    "DST_SC2",
    "DST_SC1_YNG",
    "DST_SC2_YNG",
    "DST_VAL1",
    "DST_VAL2",
    "DST_VAL1_YNG",
    "DST_VAL2_YNG",
    "PRDTRACE",
    "PRDTHSP",
    "A_MARITL",
    "PERIDNUM",
    "I_ERNVAL",
    "I_SEVAL",
    "A_HSCOL",
    "HRSWK",
    "WKSWORK",
    "PHIP_VAL",
    "POTC_VAL",
    "PMED_VAL",
    "PEMCPREM",
]
