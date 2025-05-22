from policyengine_core.data import Dataset
from tqdm import tqdm
from typing import List, Optional, Union
import requests
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import os
from policyengine_us_data.storage import STORAGE_FOLDER


class SummarizedFedSCF(Dataset):
    """Dataset containing Survey of Consumer Finances data from the Federal Reserve."""

    time_period: int
    """Year of the dataset."""

    def load(self):
        """Loads the raw SCF dataset.

        Returns:
            pd.DataFrame: The raw SCF data.
        """
        # Check if file exists
        if not os.path.exists(self.file_path):
            print(f"Raw SCF dataset file not found. Generating it.")
            self.generate()

        # Open the HDF store and return the DataFrame
        with pd.HDFStore(self.file_path, mode="r") as storage:
            return storage["data"]

    def generate(self):
        if self._scf_download_url is None:
            raise ValueError(
                f"No raw SCF data URL known for year {self.time_period}."
            )

        url = self._scf_download_url

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(
            response.headers.get("content-length", 200e6)
        )
        progress_bar = tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc="Downloading SCF",
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
            progress_bar.set_description("Downloaded SCF")
            progress_bar.total = content_length_actual
            progress_bar.close()

            zipfile = ZipFile(file)
            with pd.HDFStore(self.file_path, mode="w") as storage:
                # Find the Stata file, which should be the only .dta file in the zip
                dta_files = [
                    f for f in zipfile.namelist() if f.endswith(".dta")
                ]
                if not dta_files:
                    raise FileNotFoundError(
                        "No .dta file found in the SCF zip archive."
                    )
                # Usually there's only one .dta file, but we'll handle multiple just in case
                for dta_file in dta_files:
                    with zipfile.open(dta_file) as f:
                        # Read the Stata file with pandas
                        data = pd.read_stata(f)
                        # Add year column
                        data["year"] = self.time_period
                        # Store in HDF file
                        storage["data"] = data

    @property
    def _scf_download_url(self) -> str:
        return SummarizedSCF_URL_BY_YEAR.get(self.time_period)


class SummarizedFedSCF_2022(SummarizedFedSCF):
    time_period = 2022
    label = "Federal Reserve SCF (2022)"
    name = "fed_scf_2022"
    file_path = STORAGE_FOLDER / "fed_scf_2022.h5"
    data_format = Dataset.TABLES


class SummarizedFedSCF_2019(SummarizedFedSCF):
    time_period = 2019
    label = "Federal Reserve SCF (2019)"
    name = "fed_scf_2019"
    file_path = STORAGE_FOLDER / "fed_scf_2019.h5"
    data_format = Dataset.TABLES


class SummarizedFedSCF_2016(SummarizedFedSCF):
    time_period = 2016
    label = "Federal Reserve SCF (2016)"
    name = "fed_scf_2016"
    file_path = STORAGE_FOLDER / "fed_scf_2016.h5"
    data_format = Dataset.TABLES


# URLs for the SCF data by year
SummarizedSCF_URL_BY_YEAR = {
    2016: "https://www.federalreserve.gov/econres/files/scfp2016s.zip",
    2019: "https://www.federalreserve.gov/econres/files/scfp2019s.zip",
    2022: "https://www.federalreserve.gov/econres/files/scfp2022s.zip",
}
