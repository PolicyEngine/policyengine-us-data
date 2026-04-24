from policyengine_core.data import Dataset
from tqdm import tqdm
import requests
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.tax_unit_construction import (
    construct_tax_units,
)


OPTIONAL_PERSON_COLUMNS = {
    "NOW_COV",
    "NOW_DIR",
    "NOW_MRK",
    "NOW_MRKS",
    "NOW_MRKUN",
    "NOW_NONM",
    "NOW_PRIV",
    "NOW_PUB",
    "NOW_GRP",
    "NOW_CAID",
    "NOW_MCAID",
    "NOW_PCHIP",
    "NOW_OTHMT",
    "NOW_MCARE",
    "NOW_MIL",
    "NOW_CHAMPVA",
    "NOW_VACARE",
    "NOW_IHSFLG",
    "PTOTVAL",
}


def _resolve_person_usecols(
    available_columns, spm_unit_columns: list[str]
) -> list[str]:
    requested_columns = PERSON_COLUMNS + spm_unit_columns + TAX_UNIT_COLUMNS
    available_columns = set(available_columns)
    missing_required = sorted(
        column
        for column in requested_columns
        if column not in available_columns and column not in OPTIONAL_PERSON_COLUMNS
    )
    if missing_required:
        raise KeyError(
            "Missing required CPS person columns: " + ", ".join(missing_required[:10])
        )
    return [column for column in requested_columns if column in available_columns]


def _fill_missing_optional_person_columns(person: pd.DataFrame) -> pd.DataFrame:
    for column in OPTIONAL_PERSON_COLUMNS:
        if column not in person.columns:
            person[column] = 0
    return person


class CensusCPS(Dataset):
    """Dataset containing CPS ASEC tables in the Census format."""

    time_period: int
    """Year of the dataset."""

    tax_unit_construction_mode: str = "policyengine"
    """Mode used when constructing tax units from CPS person records."""

    def generate(self):
        if self._cps_download_url is None:
            raise ValueError(f"No raw CPS data URL known for year {self.time_period}.")

        url = self._cps_download_url

        spm_unit_columns = SPM_UNIT_COLUMNS
        if self.time_period <= 2020:
            spm_unit_columns = [
                col for col in spm_unit_columns if col != "SPM_BBSUBVAL"
            ]

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 200e6))
        progress_bar = tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc="Downloading ASEC",
        )
        if response.status_code == 404:
            raise FileNotFoundError("Received a 404 response when fetching the data.")
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
                person_path = f"{file_prefix}pppub{file_year_code}.csv"
                with zipfile.open(person_path) as f:
                    person_columns = pd.read_csv(f, nrows=0).columns
                person_usecols = _resolve_person_usecols(
                    person_columns, spm_unit_columns
                )
                with zipfile.open(person_path) as f:
                    person = pd.read_csv(
                        f,
                        usecols=person_usecols,
                    ).fillna(0)
                person = _fill_missing_optional_person_columns(person)
                tax_unit = self._create_tax_unit_table(person)
                storage["person"] = person
                with zipfile.open(f"{file_prefix}ffpub{file_year_code}.csv") as f:
                    person_family_id = person.PH_SEQ * 10 + person.PF_SEQ
                    family = pd.read_csv(f).fillna(0)
                    family_id = family.FH_SEQ * 10 + family.FFPOS
                    family = family[family_id.isin(person_family_id)]
                    storage["family"] = family
                with zipfile.open(f"{file_prefix}hhpub{file_year_code}.csv") as f:
                    person_household_id = person.PH_SEQ
                    household = pd.read_csv(f).fillna(0)
                    household_id = household.H_SEQ
                    household = household[household_id.isin(person_household_id)]
                    storage["household"] = household
                storage["tax_unit"] = tax_unit
                storage["spm_unit"] = self._create_spm_unit_table(
                    person, self.time_period
                )

    @property
    def _cps_download_url(self) -> str:
        return CPS_URL_BY_YEAR.get(self.time_period)

    def _create_tax_unit_table(
        self,
        person: pd.DataFrame,
        mode: str | None = None,
    ) -> pd.DataFrame:
        person["CENSUS_TAX_ID"] = person["TAX_ID"]
        mode = mode or self.tax_unit_construction_mode
        constructed_person, tax_unit_df = construct_tax_units(
            person=person,
            year=self.time_period,
            mode=mode,
        )
        person["TAX_ID"] = constructed_person["TAX_ID"].values
        return tax_unit_df[["TAX_ID"]]

    def _create_spm_unit_table(
        self, person: pd.DataFrame, time_period: int
    ) -> pd.DataFrame:
        spm_unit_columns = SPM_UNIT_COLUMNS
        if time_period <= 2020:
            spm_unit_columns = [
                col for col in spm_unit_columns if col != "SPM_BBSUBVAL"
            ]
        return person[spm_unit_columns].groupby(person.SPM_ID).first()


class CensusCPS_2024(CensusCPS):
    time_period = 2024
    label = "Census CPS (2024)"
    name = "census_cps_2024"
    file_path = STORAGE_FOLDER / "census_cps_2024.h5"
    data_format = Dataset.TABLES


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
    2024: "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asecpub25csv.zip",
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
    "PECOHAB",
    "SPM_ID",
    "A_FNLWGT",
    "A_LINENO",
    "A_SPOUSE",
    "A_EXPRRP",
    "A_FAMREL",
    "A_FAMTYP",
    "A_AGE",
    "A_SEX",
    "A_ENRLW",
    "A_FTPT",
    "PEDISEYE",
    "NOW_COV",
    "NOW_DIR",
    "NOW_MRK",
    "NOW_MRKS",
    "NOW_MRKUN",
    "NOW_NONM",
    "NOW_PRIV",
    "NOW_PUB",
    "NOW_GRP",
    "NOW_CAID",
    "NOW_MCAID",
    "NOW_PCHIP",
    "NOW_OTHMT",
    "NOW_MCARE",
    "NOW_MIL",
    "NOW_CHAMPVA",
    "NOW_VACARE",
    "NOW_IHSFLG",
    "WSAL_VAL",
    "INT_VAL",
    "SEMP_VAL",
    "FRSE_VAL",
    "DIV_VAL",
    "RNT_VAL",
    "SS_VAL",
    "UC_VAL",
    "LKWEEKS",  # Weeks looking for work during the year (Census variable)
    "ANN_VAL",
    "PNSN_VAL",
    "PTOTVAL",
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
    "PRCITSHP",
    "POCCU2",
    "PEINUSYR",
    "MCARE",
    "PEN_SC1",
    "PEN_SC2",
    "RESNSS1",
    "RESNSS2",
    "IHSFLG",
    "CAID",
    "CHAMPVA",
    "PEIO1COW",
    "A_MJOCC",
    "SS_YN",
    "PEAFEVER",
    "SSI_YN",
    "RESNSSI1",
    "RESNSSI2",
    "PENATVTY",
    "PEIOOCC",
    "MIL",
    "A_HRS1",
]
