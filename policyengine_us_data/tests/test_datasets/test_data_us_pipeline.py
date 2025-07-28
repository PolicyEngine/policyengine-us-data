"""
Tests to ensure Microsimulations can run with the us-data direct dependencies only.
"""

import subprocess
import sys
import tempfile
import os


def test_microsimulation_runs():
    # Create a test script that only imports direct dependencies
    test_script = """
import sys
# Remove optional dependencies from sys.modules if present
optional_deps = ["black",
    "pytest",
    "quantile-forest",
    "tabulate",
    "furo",
    "jupyter-book",
    "yaml-changelog",
    "build",
    "tomli",
    "itables"]
for dep in optional_deps:
    if dep in sys.modules:
        del sys.modules[dep]

from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps import EnhancedCPS_2024

sim = Microsimulation(dataset=EnhancedCPS_2024)
result = sim.calculate("employment_income", map_to="household", period=2025)
print("SUCCESS")
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(test_script)
        f.flush()

        # Run in subprocess with minimal environment
        result = subprocess.run(
            [sys.executable, f.name], capture_output=True, text=True
        )

        os.unlink(f.name)

        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "SUCCESS" in result.stdout
