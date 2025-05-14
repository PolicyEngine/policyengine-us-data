from policyengine_core.data import Dataset
import requests
import zipfile
import io
import pandas as pd
from policyengine_us_data.storage import STORAGE_FOLDER
import h5py

class CensusCPSOrg(Dataset):
    file_path = STORAGE_FOLDER / "census_cps_org_2024.h5"
    name = "census_cps_org_2024"
    label = "Census CPS Org (2024)"
    time_period = 2024
    data_format = Dataset.ARRAYS

    def generate(self):

        # Download from https://microdata.epi.org/epi_cpsorg_1979_2025.zip
        # Extract the file and read the epi_cpsorg_2024.dta with pandas

        url = "https://microdata.epi.org/epi_cpsorg_1979_2025.zip"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download file: {response.status_code}")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open("epi_cpsorg_2024.dta") as f:
                df = pd.read_stata(f)
                with h5py.File(self.file_path, "w") as h5f:
                    for col in df.columns:
                        h5f.create_dataset(col, data=df[col].values)