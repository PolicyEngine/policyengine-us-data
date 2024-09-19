from policyengine_core.data import Dataset
from policyengine_us_data.storage import STORAGE_FOLDER
from pathlib import Path


class IRS_PUF(Dataset):
    """Dataset containing IRS PUF tables."""

    puf_file_path: Path
    puf_demographics_file_path: Path
    data_format = Dataset.TABLES

    def generate(self):
        import pandas as pd

        puf_file_path = Path(self.puf_file_path).expanduser().resolve()
        puf_demographics_file_path = (
            Path(self.puf_demographics_file_path).expanduser().resolve()
        )

        if not puf_file_path.exists():
            raise FileNotFoundError(
                f"PUF file not found at {puf_file_path}. Either put it there, or change {Path(__file__)} point to a different path."
            )

        if not puf_demographics_file_path.exists():
            raise FileNotFoundError(
                f"PUF demographics file not found at {puf_demographics_file_path}. Either put it there, or change {Path(__file__)} point to a different path."
            )

        with pd.HDFStore(self.file_path, mode="w") as storage:
            storage.put("puf", pd.read_csv(puf_file_path))
            storage.put(
                "puf_demographics", pd.read_csv(puf_demographics_file_path)
            )


class IRS_PUF_2015(IRS_PUF):
    name = "irs_puf_2015"
    label = "IRS PUF (2015)"
    time_period = 2015
    puf_file_path = STORAGE_FOLDER / "puf_2015.csv"
    puf_demographics_file_path = STORAGE_FOLDER / "demographics_2015.csv"
    file_path = STORAGE_FOLDER / "irs_puf_2015.h5"
