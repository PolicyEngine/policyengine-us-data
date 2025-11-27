"""Tests for GEOADJ calculation."""

import importlib.util
import sys
from pathlib import Path

import pytest
import numpy as np

# Load modules directly to avoid full package import (which requires policyengine_core)
spm_dir = Path(__file__).parent.parent.parent / "policyengine_us_data" / "spm"

# Add the spm directory to path so relative imports work
sys.path.insert(0, str(spm_dir))


def load_module(name, path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


district_geoadj = load_module(
    "district_geoadj", spm_dir / "district_geoadj.py"
)
calculate_geoadj_from_rent = district_geoadj.calculate_geoadj_from_rent
get_district_geoadj = district_geoadj.get_district_geoadj


class TestGeoadjFormula:
    """Test the GEOADJ formula."""

    def test_national_equals_one(self):
        """When local = national rent, GEOADJ should be 1.0."""
        geoadj = calculate_geoadj_from_rent(1500, 1500)
        assert geoadj == pytest.approx(1.0)

    def test_high_cost_area(self):
        """High cost area (2x national) should give GEOADJ > 1."""
        # If local rent is 2x national:
        # GEOADJ = 2.0 * 0.492 + 0.508 = 1.492
        geoadj = calculate_geoadj_from_rent(3000, 1500)
        assert geoadj == pytest.approx(1.492)

    def test_low_cost_area(self):
        """Low cost area (0.5x national) should give GEOADJ < 1."""
        # If local rent is 0.5x national:
        # GEOADJ = 0.5 * 0.492 + 0.508 = 0.754
        geoadj = calculate_geoadj_from_rent(750, 1500)
        assert geoadj == pytest.approx(0.754)

    def test_vectorized(self):
        """Should handle array inputs."""
        local_rents = np.array([1500, 2250, 1000])
        national = 1500

        geoadj = calculate_geoadj_from_rent(local_rents, national)

        assert len(geoadj) == 3
        assert geoadj[0] == pytest.approx(1.0)  # Equal to national
        assert geoadj[1] > 1.0  # 1.5x national
        assert geoadj[2] < 1.0  # 0.67x national


class TestDistrictGeoadj:
    """Test district GEOADJ lookup (requires cached data or API)."""

    def test_get_district_geoadj_default(self):
        """Should return 1.0 for unknown district codes."""
        try:
            import census  # noqa: F401
        except ImportError:
            pytest.skip("census package required for this test")

        # Non-existent district should return 1.0
        geoadj = get_district_geoadj("9999", year=2022)
        assert geoadj == 1.0
