from __future__ import annotations

import math
from typing import Any


TRUSTEES_CORE_THRESHOLD_ASSUMPTION = {
    "name": "trustees-core-thresholds-v1",
    "description": (
        "Best-public Trustees tax-side approximation: keep Social Security "
        "benefit-tax thresholds fixed, but wage-index core ordinary federal "
        "tax thresholds after 2034."
    ),
    "source": "SSA 2025 Trustees Report V.C.7",
    "start_year": 2035,
    "parameter_groups": [
        "ordinary_income_brackets",
        "standard_deduction",
        "aged_blind_standard_deduction",
        "capital_gains_thresholds",
        "amt_thresholds",
    ],
}


def round_amount(amount: float, rounding: dict | None) -> float:
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


def iter_updatable_parameters(
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


def apply_wage_growth_to_parameter(
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
        updated_value = round_amount(previous_value * wage_growth, rounding)
        parameter.update(
            period=f"year:{year}-01-01:1",
            value=updated_value,
        )


def create_wage_indexed_core_thresholds_reform(
    *,
    start_year: int = 2035,
    end_year: int = 2100,
):
    from policyengine_us.model_api import Reform

    def modify_parameters(parameters):
        nawi = parameters.gov.ssa.nawi
        roots = [
            parameters.gov.irs.income.bracket.thresholds,
            parameters.gov.irs.deductions.standard.amount,
            parameters.gov.irs.deductions.standard.aged_or_blind.amount,
            parameters.gov.irs.capital_gains.thresholds,
            parameters.gov.irs.income.amt.brackets,
            parameters.gov.irs.income.amt.exemption.amount,
            parameters.gov.irs.income.amt.exemption.phase_out.start,
            parameters.gov.irs.income.amt.exemption.separate_limit,
        ]

        seen = set()
        for root in roots:
            for parameter in iter_updatable_parameters(root):
                if parameter.name in seen:
                    continue
                seen.add(parameter.name)
                apply_wage_growth_to_parameter(
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


def create_wage_indexed_full_irs_uprating_reform(
    *,
    start_year: int = 2035,
    end_year: int = 2100,
):
    from policyengine_us.model_api import Reform

    def modify_parameters(parameters):
        nawi = parameters.gov.ssa.nawi
        seen = set()
        for parameter in iter_updatable_parameters(
            parameters.gov.irs,
            uprating_parameter="gov.irs.uprating",
        ):
            if parameter.name in seen:
                continue
            seen.add(parameter.name)
            apply_wage_growth_to_parameter(
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

