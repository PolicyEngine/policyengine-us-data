import numpy as np
import pytest

from policyengine_us_data.datasets.cps.medicaid_cost import (
    allocate_medicaid_cost_if_enrolled_by_slcsp,
    family_tier_slcsp_person_share,
)


def test_allocate_medicaid_cost_if_enrolled_matches_state_targets():
    person_slcsp = np.array([100.0, 200.0, 300.0, 400.0])
    medicaid_enrolled = np.array([True, True, False, True])
    person_weight = np.array([1.0, 2.0, 1.0, 4.0])
    state_codes = np.array(["CA", "CA", "CA", "NY"])
    state_spending = {"CA": 5_000.0, "NY": 8_000.0}

    costs = allocate_medicaid_cost_if_enrolled_by_slcsp(
        person_slcsp=person_slcsp,
        medicaid_enrolled=medicaid_enrolled,
        person_weight=person_weight,
        state_codes=state_codes,
        state_spending=state_spending,
    )

    ca_baseline_cost = np.sum(costs[:2] * person_weight[:2])
    ny_baseline_cost = costs[3] * person_weight[3]
    assert ca_baseline_cost == pytest.approx(5_000.0)
    assert ny_baseline_cost == pytest.approx(8_000.0)
    assert costs[2] / costs[0] == pytest.approx(3)


def test_allocate_medicaid_cost_if_enrolled_fills_missing_slcsp_with_state_mean():
    person_slcsp = np.array([0.0, 200.0, 400.0])
    medicaid_enrolled = np.array([True, True, False])
    person_weight = np.array([1.0, 1.0, 1.0])
    state_codes = np.array([b"CA", b"CA", b"CA"])

    costs = allocate_medicaid_cost_if_enrolled_by_slcsp(
        person_slcsp=person_slcsp,
        medicaid_enrolled=medicaid_enrolled,
        person_weight=person_weight,
        state_codes=state_codes,
        state_spending={"CA": 900.0},
    )

    assert np.sum(costs[:2] * person_weight[:2]) == pytest.approx(900.0)
    assert costs[0] == pytest.approx(costs[1] * 1.5)
    assert costs[2] == pytest.approx(costs[1] * 2)


def test_allocate_medicaid_cost_if_enrolled_requires_aligned_inputs():
    with pytest.raises(ValueError, match="same length"):
        allocate_medicaid_cost_if_enrolled_by_slcsp(
            person_slcsp=np.array([100.0]),
            medicaid_enrolled=np.array([True, False]),
            person_weight=np.array([1.0]),
            state_codes=np.array(["CA"]),
            state_spending={"CA": 100.0},
        )


def test_family_tier_slcsp_person_share_allocates_ny_tax_unit_premium():
    share = family_tier_slcsp_person_share(
        state_codes=np.array(["NY", "NY", "NY", "NY"]),
        base_cost=np.array([500.0, 500.0, 500.0, 500.0]),
        age=np.array([40, 38, 10, 8]),
        is_tax_unit_dependent=np.array([False, False, True, True]),
        person_tax_unit_id=np.array([1, 1, 1, 1]),
        tax_unit_id=np.array([1]),
        fallback=np.array([999.0, 999.0, 999.0, 999.0]),
        time_period=2026,
    )

    np.testing.assert_allclose(share, np.full(4, 500.0 * 2.85 / 4))


def test_family_tier_slcsp_person_share_uses_ny_child_only_tier():
    share = family_tier_slcsp_person_share(
        state_codes=np.array(["NY", "NY"]),
        base_cost=np.array([500.0, 500.0]),
        age=np.array([12, 10]),
        is_tax_unit_dependent=np.array([True, True]),
        person_tax_unit_id=np.array([1, 1]),
        tax_unit_id=np.array([1]),
        fallback=np.array([999.0, 999.0]),
        time_period=2026,
    )

    np.testing.assert_allclose(share, np.full(2, 500.0 * 0.412 / 2))


def test_family_tier_slcsp_person_share_preserves_vt_child_only_fallback():
    fallback = np.array([500.0, 500.0])

    share = family_tier_slcsp_person_share(
        state_codes=np.array(["VT", "VT"]),
        base_cost=np.array([500.0, 500.0]),
        age=np.array([12, 10]),
        is_tax_unit_dependent=np.array([True, True]),
        person_tax_unit_id=np.array([1, 1]),
        tax_unit_id=np.array([1]),
        fallback=fallback,
        time_period=2026,
    )

    np.testing.assert_allclose(share, fallback)


def test_family_tier_slcsp_person_share_allocates_vt_family_tier_premium():
    share = family_tier_slcsp_person_share(
        state_codes=np.array(["VT", "VT", "VT"]),
        base_cost=np.array([500.0, 500.0, 500.0]),
        age=np.array([40, 10, 8]),
        is_tax_unit_dependent=np.array([False, True, True]),
        person_tax_unit_id=np.array([1, 1, 1]),
        tax_unit_id=np.array([1]),
        fallback=np.array([999.0, 999.0, 999.0]),
        time_period=2026,
    )

    np.testing.assert_allclose(share, np.full(3, 500.0 * 1.93 / 3))
