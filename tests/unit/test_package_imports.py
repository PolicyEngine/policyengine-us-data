import subprocess
import sys
from pathlib import Path

import policyengine_us_data


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_create_database_tables_imports_cleanly_in_fresh_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import policyengine_us_data.db.create_database_tables",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_package_root_lazily_exports_dataset_classes():
    assert policyengine_us_data.EnhancedCPS_2024.__name__ == "EnhancedCPS_2024"
    assert policyengine_us_data.ExtendedCPS_2024.__name__ == "ExtendedCPS_2024"
    assert policyengine_us_data.CPS_2024.__name__ == "CPS_2024"
    assert policyengine_us_data.PUF_2024.__name__ == "PUF_2024"
