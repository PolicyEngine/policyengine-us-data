from policyengine_core.data import Dataset
import requests
import zipfile
import io
import pandas as pd
from policyengine_us_data.storage import STORAGE_FOLDER
import h5py
from tqdm import tqdm
import huggingface_hub

class CensusCPSOrg(Dataset):
    file_path = STORAGE_FOLDER / "census_cps_org_2024.h5"
    name = "census_cps_org_2024"
    label = "Census CPS Org (2024)"
    time_period = 2024
    data_format = Dataset.TABLES

    def generate(self):

        # Download from https://microdata.epi.org/epi_cpsorg_1979_2025.zip
        # Extract the file and read the epi_cpsorg_2024.dta with pandas
        DOWNLOAD_FROM_CENSUS = False
        if DOWNLOAD_FROM_CENSUS:
            url = "https://microdata.epi.org/epi_cpsorg_1979_2025.zip"
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading CPS Org data")
            content = b''
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                content += data
            progress_bar.close()
            response.content = content
            if response.status_code != 200:
                raise Exception(f"Failed to download file: {response.status_code}")
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open("epi_cpsorg_2024.dta") as f:
                    df = pd.read_stata(f)
        else:
            huggingface_hub.hf_hub_download(
                repo_id="policyengine/policyengine-us-data",
                filename="epi_cpsorg_2024.dta",
                repo_type="model",
                local_dir=STORAGE_FOLDER,
            )
            df = pd.read_stata(STORAGE_FOLDER / "epi_cpsorg_2024.dta")
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except:
                df[col] = df[col].astype(str)
        with pd.HDFStore(self.file_path, "a") as f:
                f.put("main", df)