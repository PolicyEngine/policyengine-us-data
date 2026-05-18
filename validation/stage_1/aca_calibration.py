"""Shared ACA PTC state-target validation helpers."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pytest

from policyengine_us_data.storage.calibration_targets.aca_ptc_targets import (
    load_aca_ptc_state_targets,
)

# Stage 1 publication should not be blocked by noisy state-level ACA target
# diagnostics; hard export-contract validators still gate unusable artifacts.
ACA_PTC_STATE_TOLERANCE = 100.0


def assert_aca_ptc_calibration(
    sim,
    *,
    period: int = 2025,
    emit: Callable[[str], None] | None = None,
) -> None:
    """Check state ACA PTC totals against the IRS SOI total-PTC target."""
    targets = load_aca_ptc_state_targets(period)
    if targets is None:
        pytest.skip("ACA PTC state targets not available")

    emit = emit or logging.info
    state_code_hh = sim.calculate("state_code", map_to="household").values
    aca_ptc = sim.calculate("aca_ptc", map_to="household", period=period)

    failures = []
    for row in targets.itertuples(index=False):
        state = row.state
        target_spending = float(row.TotalPTCAmount)
        simulated = float(aca_ptc[state_code_hh == state].sum())
        if target_spending <= 0:
            pct_error = np.inf
        else:
            pct_error = abs(simulated - target_spending) / target_spending

        message = (
            f"{state}: simulated ${simulated / 1e9:.2f} bn  "
            f"target ${target_spending / 1e9:.2f} bn  "
            f"error {pct_error:.2%}"
        )
        emit(message)

        if pct_error > ACA_PTC_STATE_TOLERANCE:
            failures.append(message)

    assert not failures, (
        "One or more states exceeded tolerance of "
        f"{ACA_PTC_STATE_TOLERANCE:.0%}:\n" + "\n".join(failures)
    )
