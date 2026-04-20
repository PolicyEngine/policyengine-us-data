"""Unit tests for the Marketplace plan benchmark ratio back-out."""

from __future__ import annotations

import numpy as np

from policyengine_us_data.datasets.cps.cps import (
    MARKETPLACE_PLAN_BENCHMARK_RATIO_MAX,
    MARKETPLACE_PLAN_BENCHMARK_RATIO_MIN,
    compute_marketplace_plan_benchmark_ratio,
)


def test_silver_plan_back_out_yields_unit_ratio() -> None:
    reported_premium = np.array([2_000.0])
    aca_ptc = np.array([4_000.0])
    slcsp = np.array([6_000.0])
    takes_up_aca = np.array([True])

    result = compute_marketplace_plan_benchmark_ratio(
        reported_premium=reported_premium,
        aca_ptc=aca_ptc,
        slcsp=slcsp,
        takes_up_aca=takes_up_aca,
    )

    np.testing.assert_allclose(result, [1.0])


def test_bronze_plan_back_out_yields_sub_silver_ratio() -> None:
    reported_premium = np.array([800.0])
    aca_ptc = np.array([4_000.0])
    slcsp = np.array([6_000.0])
    takes_up_aca = np.array([True])

    result = compute_marketplace_plan_benchmark_ratio(
        reported_premium=reported_premium,
        aca_ptc=aca_ptc,
        slcsp=slcsp,
        takes_up_aca=takes_up_aca,
    )

    np.testing.assert_allclose(result, [0.8])


def test_gold_plan_back_out_yields_above_silver_ratio() -> None:
    reported_premium = np.array([3_500.0])
    aca_ptc = np.array([4_000.0])
    slcsp = np.array([6_000.0])
    takes_up_aca = np.array([True])

    result = compute_marketplace_plan_benchmark_ratio(
        reported_premium=reported_premium,
        aca_ptc=aca_ptc,
        slcsp=slcsp,
        takes_up_aca=takes_up_aca,
    )

    np.testing.assert_allclose(result, [1.25])


def test_non_marketplace_households_keep_default_ratio() -> None:
    reported_premium = np.array([500.0, 2_000.0])
    aca_ptc = np.array([0.0, 0.0])
    slcsp = np.array([5_000.0, 6_000.0])
    takes_up_aca = np.array([False, False])

    result = compute_marketplace_plan_benchmark_ratio(
        reported_premium=reported_premium,
        aca_ptc=aca_ptc,
        slcsp=slcsp,
        takes_up_aca=takes_up_aca,
    )

    np.testing.assert_allclose(result, [1.0, 1.0])


def test_zero_slcsp_returns_default_ratio_even_for_marketplace_taker() -> None:
    reported_premium = np.array([1_000.0])
    aca_ptc = np.array([0.0])
    slcsp = np.array([0.0])
    takes_up_aca = np.array([True])

    result = compute_marketplace_plan_benchmark_ratio(
        reported_premium=reported_premium,
        aca_ptc=aca_ptc,
        slcsp=slcsp,
        takes_up_aca=takes_up_aca,
    )

    np.testing.assert_allclose(result, [1.0])


def test_ratios_are_clipped_to_configured_window() -> None:
    reported_premium = np.array([0.0, 10_000.0])
    aca_ptc = np.array([0.0, 0.0])
    slcsp = np.array([6_000.0, 6_000.0])
    takes_up_aca = np.array([True, True])

    result = compute_marketplace_plan_benchmark_ratio(
        reported_premium=reported_premium,
        aca_ptc=aca_ptc,
        slcsp=slcsp,
        takes_up_aca=takes_up_aca,
    )

    np.testing.assert_allclose(
        result,
        [
            MARKETPLACE_PLAN_BENCHMARK_RATIO_MIN,
            MARKETPLACE_PLAN_BENCHMARK_RATIO_MAX,
        ],
    )
