import os
from pathlib import Path

# The `us` library excludes DC from its state lists by default.
# Setting this before `import us` ensures DC appears in
# STATES_AND_TERRITORIES so lookup("DC") works and DC is
# processed alongside the 50 states.
os.environ["DC_STATEHOOD"] = "1"

STORAGE_FOLDER = Path(__file__).parent
CALIBRATION_FOLDER = STORAGE_FOLDER / "calibration_targets"
DOCS_FOLDER = STORAGE_FOLDER.parent.parent / "docs"
