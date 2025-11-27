"""Tests for CE base threshold calculation."""

import importlib.util
import sys
from pathlib import Path

import pytest

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


# Load modules in dependency order
ce_threshold = load_module("ce_threshold", spm_dir / "ce_threshold.py")
calculate_base_thresholds = ce_threshold.calculate_base_thresholds

district_geoadj = load_module(
    "district_geoadj", spm_dir / "district_geoadj.py"
)

local_threshold = load_module(
    "local_threshold", spm_dir / "local_threshold.py"
)
spm_equivalence_scale = local_threshold.spm_equivalence_scale


class TestBaseThresholds:
    """Test base threshold retrieval and forecasting."""

    def test_published_2024_thresholds(self):
        """Should return correct published 2024 thresholds."""
        thresholds = calculate_base_thresholds(2024)

        assert thresholds["renter"] == 39430
        assert thresholds["owner_with_mortgage"] == 39068
        assert thresholds["owner_without_mortgage"] == 32586

    def test_published_2023_thresholds(self):
        """Should return correct published 2023 thresholds."""
        thresholds = calculate_base_thresholds(2023)

        assert thresholds["renter"] == 36606

    def test_forecasted_2025_higher_than_2024(self):
        """Forecasted 2025 should be higher than 2024."""
        t_2024 = calculate_base_thresholds(2024)
        t_2025 = calculate_base_thresholds(2025)

        assert t_2025["renter"] > t_2024["renter"]

    def test_tenure_ordering(self):
        """Renters should have highest threshold, owner w/o mortgage lowest."""
        thresholds = calculate_base_thresholds(2024)

        # Owner without mortgage has lowest housing costs
        assert (
            thresholds["owner_without_mortgage"]
            < thresholds["owner_with_mortgage"]
        )
        assert thresholds["owner_with_mortgage"] <= thresholds["renter"]


class TestEquivalenceScale:
    """Test SPM equivalence scale calculation."""

    def test_reference_family_normalized(self):
        """Reference family (2A2C) should equal 1.0 when normalized."""
        scale = spm_equivalence_scale(2, 2, normalize=True)
        assert scale == pytest.approx(1.0)

    def test_reference_family_raw(self):
        """Reference family (2A2C) should equal 2.1 raw."""
        scale = spm_equivalence_scale(2, 2, normalize=False)
        assert scale == pytest.approx(2.1)

    def test_single_adult(self):
        """Single adult = 1.0 / 2.1 ≈ 0.476."""
        scale = spm_equivalence_scale(1, 0, normalize=True)
        assert scale == pytest.approx(1.0 / 2.1, rel=0.01)

    def test_larger_family(self):
        """Larger families scale up appropriately."""
        # 3 adults, 4 children = 1 + 0.5*2 + 0.3*4 = 3.2
        scale = spm_equivalence_scale(3, 4, normalize=False)
        assert scale == pytest.approx(3.2)

    def test_vectorized(self):
        """Should handle array inputs."""
        import numpy as np

        adults = np.array([1, 2, 2, 3])
        children = np.array([0, 0, 2, 4])

        scales = spm_equivalence_scale(adults, children, normalize=False)

        assert len(scales) == 4
        assert scales[0] == pytest.approx(1.0)  # 1A0C
        assert scales[1] == pytest.approx(1.5)  # 2A0C
        assert scales[2] == pytest.approx(2.1)  # 2A2C
        assert scales[3] == pytest.approx(3.2)  # 3A4C
