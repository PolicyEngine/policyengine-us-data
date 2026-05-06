import json
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


def test_validate_national_h5_imports_cleanly_in_fresh_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import policyengine_us_data.calibration.validate_national_h5",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_deep_cps_long_term_import_skips_dataset_class_dependencies():
    script = """
import importlib
import json
import sys

importlib.import_module("policyengine_us_data.datasets.cps.long_term.ssa_data")
blocked = ["policyengine_core"]
print(json.dumps({name: name in sys.modules for name in blocked}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"policyengine_core": False}


def test_package_root_lazily_exports_dataset_classes():
    assert policyengine_us_data.EnhancedCPS_2024.__name__ == "EnhancedCPS_2024"
    assert policyengine_us_data.ExtendedCPS_2024.__name__ == "ExtendedCPS_2024"
    assert policyengine_us_data.CPS_2024.__name__ == "CPS_2024"
    assert policyengine_us_data.PUF_2024.__name__ == "PUF_2024"
