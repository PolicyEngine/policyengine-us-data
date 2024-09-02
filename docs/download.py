from policyengine_us_data.utils.github import download
from policyengine_us_data.data_storage import STORAGE_FOLDER

download(
    "PolicyEngine",
    "policyengine-us-data",
    "release",
    "enhanced_cps_2024.h5",
    STORAGE_FOLDER / "enhanced_cps_2024.h5",
)

download(
    "PolicyEngine",
    "policyengine-us-data",
    "release",
    "cps_2024.h5",
    STORAGE_FOLDER / "cps_2024.h5",
)

download(
    "PolicyEngine",
    "irs-soi-puf",
    "release",
    "puf_2024.h5",
    STORAGE_FOLDER / "puf_2024.h5",
)
