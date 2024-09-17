from policyengine_us_data.utils.github import upload
from pathlib import Path

FOLDER = Path(__file__).parent

upload(
    "PolicyEngine",
    "policyengine-us-data",
    "release",
    "enhanced_cps_2024.h5",
    FOLDER / "enhanced_cps_2024.h5",
)

upload(
    "PolicyEngine",
    "policyengine-us-data",
    "release",
    "cps_2024.h5",
    FOLDER / "cps_2024.h5",
)

upload(
    "PolicyEngine",
    "irs-soi-puf",
    "release",
    "puf_2024.h5",
    FOLDER / "puf_2024.h5",
)
