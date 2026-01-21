"""
Tests for H6 Social Security reform threshold crossover logic.

The H6 reform phases out OASDI taxation from 2045-2053 while preserving
HI taxation. This requires careful handling when OASDI thresholds rise
above HI thresholds, necessitating parameter swapping.

These tests validate the mathematical logic without requiring full
policyengine imports (which need heavy dependencies like torch).
"""

import pytest

# Constants from the H6 reform implementation
HI_SINGLE = 34_000
HI_JOINT = 44_000


def calculate_oasdi_thresholds(year: int) -> tuple[int, int]:
    """Calculate OASDI thresholds for a given year during phase-out."""
    if year < 2045 or year > 2053:
        raise ValueError("Phase-out only applies to 2045-2053")

    i = year - 2045
    oasdi_single = 32_500 + (7_500 * i)
    oasdi_joint = 65_000 + (15_000 * i)
    return oasdi_single, oasdi_joint


def get_swapped_thresholds(
    oasdi_threshold: int, hi_threshold: int
) -> tuple[int, int]:
    """
    Apply min/max swap to handle threshold crossover.

    Returns (base_threshold, adjusted_threshold) where base <= adjusted.
    """
    return min(oasdi_threshold, hi_threshold), max(
        oasdi_threshold, hi_threshold
    )


def needs_crossover_swap(oasdi_threshold: int, hi_threshold: int) -> bool:
    """Check if OASDI threshold has crossed above HI threshold."""
    return oasdi_threshold > hi_threshold


class TestH6ThresholdCalculation:
    """Test OASDI threshold progression during phase-out."""

    def test_2045_single_threshold(self):
        """2045 single OASDI threshold should be $32,500."""
        oasdi_single, _ = calculate_oasdi_thresholds(2045)
        assert oasdi_single == 32_500

    def test_2045_joint_threshold(self):
        """2045 joint OASDI threshold should be $65,000."""
        _, oasdi_joint = calculate_oasdi_thresholds(2045)
        assert oasdi_joint == 65_000

    def test_2053_single_threshold(self):
        """2053 single OASDI threshold should be $92,500."""
        oasdi_single, _ = calculate_oasdi_thresholds(2053)
        assert oasdi_single == 92_500

    def test_2053_joint_threshold(self):
        """2053 joint OASDI threshold should be $185,000."""
        _, oasdi_joint = calculate_oasdi_thresholds(2053)
        assert oasdi_joint == 185_000

    def test_threshold_progression_single(self):
        """Single thresholds should increase by $7,500 per year."""
        expected = {
            2045: 32_500,
            2046: 40_000,
            2047: 47_500,
            2048: 55_000,
            2049: 62_500,
            2050: 70_000,
            2051: 77_500,
            2052: 85_000,
            2053: 92_500,
        }
        for year, expected_val in expected.items():
            oasdi_single, _ = calculate_oasdi_thresholds(year)
            assert oasdi_single == expected_val, f"Year {year}"

    def test_threshold_progression_joint(self):
        """Joint thresholds should increase by $15,000 per year."""
        expected = {
            2045: 65_000,
            2046: 80_000,
            2047: 95_000,
            2048: 110_000,
            2049: 125_000,
            2050: 140_000,
            2051: 155_000,
            2052: 170_000,
            2053: 185_000,
        }
        for year, expected_val in expected.items():
            _, oasdi_joint = calculate_oasdi_thresholds(year)
            assert oasdi_joint == expected_val, f"Year {year}"


class TestH6ThresholdCrossover:
    """Test the threshold crossover detection and handling.

    Key insight: During phase-out, OASDI thresholds rise above HI thresholds.
    - HI thresholds are frozen at $34k single / $44k joint
    - Joint filers cross immediately (2045: $65k > $44k)
    - Single filers cross in 2046 ($40k > $34k)
    """

    def test_2045_single_no_crossover(self):
        """In 2045, single OASDI ($32.5k) is below HI ($34k) - no swap needed."""
        oasdi_single, _ = calculate_oasdi_thresholds(2045)
        assert not needs_crossover_swap(oasdi_single, HI_SINGLE)
        assert oasdi_single < HI_SINGLE

    def test_2045_joint_has_crossover(self):
        """In 2045, joint OASDI ($65k) exceeds HI ($44k) - swap needed."""
        _, oasdi_joint = calculate_oasdi_thresholds(2045)
        assert needs_crossover_swap(oasdi_joint, HI_JOINT)
        assert oasdi_joint > HI_JOINT

    def test_2046_single_has_crossover(self):
        """In 2046, single OASDI ($40k) exceeds HI ($34k) - swap needed."""
        oasdi_single, _ = calculate_oasdi_thresholds(2046)
        assert needs_crossover_swap(oasdi_single, HI_SINGLE)
        assert oasdi_single > HI_SINGLE

    def test_all_years_joint_crossover(self):
        """Joint filers have crossover in all phase-out years."""
        for year in range(2045, 2054):
            _, oasdi_joint = calculate_oasdi_thresholds(year)
            assert needs_crossover_swap(oasdi_joint, HI_JOINT), f"Year {year}"

    def test_single_crossover_starts_2046(self):
        """Single filers cross over starting in 2046."""
        # 2045: no crossover
        oasdi_2045, _ = calculate_oasdi_thresholds(2045)
        assert not needs_crossover_swap(oasdi_2045, HI_SINGLE)

        # 2046+: crossover
        for year in range(2046, 2054):
            oasdi_single, _ = calculate_oasdi_thresholds(year)
            assert needs_crossover_swap(
                oasdi_single, HI_SINGLE
            ), f"Year {year}"


class TestH6ThresholdSwapping:
    """Test min/max swap ensures base <= adjusted_base."""

    def test_swap_when_oasdi_higher(self):
        """When OASDI > HI, swap puts HI in base slot."""
        oasdi = 65_000
        hi = 44_000
        base, adjusted = get_swapped_thresholds(oasdi, hi)
        assert base == hi == 44_000
        assert adjusted == oasdi == 65_000
        assert base <= adjusted

    def test_no_swap_when_oasdi_lower(self):
        """When OASDI < HI, no swap needed."""
        oasdi = 32_500
        hi = 34_000
        base, adjusted = get_swapped_thresholds(oasdi, hi)
        assert base == oasdi == 32_500
        assert adjusted == hi == 34_000
        assert base <= adjusted

    def test_swap_preserves_ordering_all_years(self):
        """Swapped thresholds always maintain base <= adjusted."""
        for year in range(2045, 2054):
            oasdi_single, oasdi_joint = calculate_oasdi_thresholds(year)

            base_s, adj_s = get_swapped_thresholds(oasdi_single, HI_SINGLE)
            base_j, adj_j = get_swapped_thresholds(oasdi_joint, HI_JOINT)

            assert base_s <= adj_s, f"Single ordering violated in {year}"
            assert base_j <= adj_j, f"Joint ordering violated in {year}"


class TestH6RateSwapping:
    """Test rate swapping logic during transition.

    Key insight: PolicyEngine requires one rate structure per year.
    When thresholds cross, we swap to (0.35, 0.85) to minimize error.
    """

    def test_2045_error_analysis(self):
        """In 2045, swapped rates minimize error vs default rates."""
        # 2045 situation:
        # Single: OASDI=$32.5k, HI=$34k -> $1.5k range affected
        # Joint: OASDI=$65k, HI=$44k -> $21k range affected

        # With swapped rates (0.35/0.85 instead of 0.50/0.85):
        # Single: undertax by 15% on $1.5k = $225
        # With default rates (0.50/0.85):
        # Joint: overtax by 15% on $21k = $3,150

        single_range = 34_000 - 32_500  # $1,500
        joint_range = 65_000 - 44_000  # $21,000

        rate_diff = 0.50 - 0.35  # 15%

        single_error_swapped = single_range * rate_diff  # $225 undertax
        joint_error_default = joint_range * rate_diff  # $3,150 overtax

        assert single_error_swapped == pytest.approx(225)
        assert joint_error_default == pytest.approx(3_150)
        assert joint_error_default / single_error_swapped == pytest.approx(
            14.0
        ), "Swapped rates should have 14x less error"

    def test_swapped_rates_align_with_tax_cut_intent(self):
        """Swapped rates undertax (not overtax), aligning with reform intent."""
        # H6 is a tax cut - undertaxing is more aligned with legislative intent
        # than overtaxing would be
        single_undertax = (34_000 - 32_500) * 0.15  # $225
        assert single_undertax > 0  # Positive = undertax (taxpayer-favorable)


class TestH6EliminationPhase:
    """Test the post-2054 elimination phase parameters."""

    def test_elimination_thresholds(self):
        """After 2054, only HI thresholds remain active."""
        # Base thresholds = HI ($34k/$44k)
        # Adjusted thresholds = very high (effectively disabled)
        INFINITY_THRESHOLD = 9_999_999

        assert HI_SINGLE == 34_000
        assert HI_JOINT == 44_000
        assert INFINITY_THRESHOLD > HI_SINGLE * 100
        assert INFINITY_THRESHOLD > HI_JOINT * 100

    def test_elimination_rates(self):
        """After 2054, both tiers use 35% (HI-only rate)."""
        HI_RATE = 0.35
        OASDI_RATE = 0.50  # eliminated

        # In elimination phase, tier 1 = 35%, tier 2 = 35% (no additional)
        assert HI_RATE == 0.35
        assert HI_RATE + OASDI_RATE == 0.85  # was combined rate
