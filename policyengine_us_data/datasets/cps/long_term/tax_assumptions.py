from __future__ import annotations

import math
from typing import Any

from policyengine_us.reforms.ssa.trustees_core_thresholds import (
    TRUSTEES_CORE_THRESHOLD_ASSUMPTION,
    create_trustees_core_thresholds_reform as create_trustees_core_thresholds_reform,
)


def _round_amount(amount: float, rounding: dict | None) -> float:
    if not rounding:
        return amount

    interval = float(rounding["interval"])
    rounding_type = rounding["type"]

    if rounding_type == "downwards":
        return math.floor(amount / interval) * interval
    if rounding_type == "nearest":
        return math.floor(amount / interval + 0.5) * interval

    raise ValueError(f"Unsupported rounding type: {rounding_type}")


def _uprating_parameter_name(parameter) -> str | None:
    metadata = getattr(parameter, "metadata", {})
    uprating = metadata.get("uprating")
    if isinstance(uprating, dict):
        return uprating.get("parameter")
    return uprating


def _iter_updatable_parameters(
    root,
    *,
    uprating_parameter: str | None = None,
) -> list:
    candidates = [root]
    if hasattr(root, "get_descendants"):
        candidates.extend(root.get_descendants())

    result = []
    for candidate in candidates:
        if candidate.__class__.__name__ != "Parameter":
            continue
        uprating_name = _uprating_parameter_name(candidate)
        if uprating_name is None:
            continue
        if uprating_parameter is not None and uprating_name != uprating_parameter:
            continue
        result.append(candidate)
    return result


def _apply_wage_growth_to_parameter(
    parameter,
    *,
    nawi,
    start_year: int,
    end_year: int,
) -> None:
    metadata = getattr(parameter, "metadata", {})
    uprating = metadata.get("uprating")
    rounding = uprating.get("rounding") if isinstance(uprating, dict) else None

    for year in range(start_year, end_year + 1):
        previous_value = float(parameter(f"{year - 1}-01-01"))
        wage_growth = float(nawi(f"{year - 1}-01-01")) / float(
            nawi(f"{year - 2}-01-01")
        )
        updated_value = _round_amount(previous_value * wage_growth, rounding)
        parameter.update(
            period=f"year:{year}-01-01:1",
            value=updated_value,
        )


def create_wage_indexed_full_irs_uprating_reform(
    *,
    start_year: int = 2035,
    end_year: int = 2100,
):
    """Diagnostic sensitivity: wage-index every IRS parameter on the IRS CPI path."""
    from policyengine_us.model_api import Reform

    def modify_parameters(parameters):
        nawi = parameters.gov.ssa.nawi
        seen = set()
        for parameter in _iter_updatable_parameters(
            parameters.gov.irs,
            uprating_parameter="gov.irs.uprating",
        ):
            if parameter.name in seen:
                continue
            seen.add(parameter.name)
            _apply_wage_growth_to_parameter(
                parameter,
                nawi=nawi,
                start_year=start_year,
                end_year=end_year,
            )
        return parameters

    class reform(Reform):
        def apply(self):
            self.modify_parameters(modify_parameters)

    return reform


def get_long_run_tax_assumption_metadata(
    name: str,
    *,
    end_year: int,
) -> dict[str, Any]:
    if name != TRUSTEES_CORE_THRESHOLD_ASSUMPTION["name"]:
        raise ValueError(f"Unknown long-run tax assumption: {name}")

    metadata = dict(TRUSTEES_CORE_THRESHOLD_ASSUMPTION)
    metadata["end_year"] = int(end_year)
    return metadata
