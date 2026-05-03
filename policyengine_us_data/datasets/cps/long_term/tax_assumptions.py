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
    "not_default_current_law": True,
}

try:
    from policyengine_us.reforms.ssa.trustees_core_thresholds import (
        TRUSTEES_CORE_THRESHOLD_ASSUMPTION as _PE_TRUSTEES_CORE_THRESHOLD_ASSUMPTION,
        create_trustees_core_thresholds_reform as _pe_create_trustees_core_thresholds_reform,
    )
except ImportError:
    _pe_create_trustees_core_thresholds_reform = None
else:
    TRUSTEES_CORE_THRESHOLD_ASSUMPTION = dict(_PE_TRUSTEES_CORE_THRESHOLD_ASSUMPTION)


# Fallback used until the PolicyEngine US dependency includes the native
# trustees-core-thresholds-v1 reform.
TRUSTEES_AVERAGE_WAGE_GROWTH_PCT = {
    2034: 3.85,
    2035: 3.72,
    2036: 3.65,
    2037: 3.66,
    2038: 3.66,
    2039: 3.68,
    2040: 3.65,
    2041: 3.63,
    2042: 3.62,
    2043: 3.60,
    2044: 3.57,
    2045: 3.55,
    2046: 3.53,
    2047: 3.53,
    2048: 3.53,
    2049: 3.52,
    2050: 3.51,
    2051: 3.51,
    2052: 3.51,
    2053: 3.50,
    2054: 3.50,
    2055: 3.49,
    2056: 3.50,
    2057: 3.51,
    2058: 3.52,
    2059: 3.52,
    2060: 3.53,
    2061: 3.53,
    2062: 3.54,
    2063: 3.55,
    2064: 3.55,
    2065: 3.55,
    2066: 3.55,
    2067: 3.56,
    2068: 3.55,
    2069: 3.55,
    2070: 3.56,
    2071: 3.55,
    2072: 3.55,
    2073: 3.55,
    2074: 3.55,
    2075: 3.56,
    2076: 3.56,
    2077: 3.56,
    2078: 3.56,
    2079: 3.56,
    2080: 3.55,
    2081: 3.56,
    2082: 3.56,
    2083: 3.56,
    2084: 3.56,
    2085: 3.56,
    2086: 3.57,
    2087: 3.56,
    2088: 3.56,
    2089: 3.56,
    2090: 3.56,
    2091: 3.56,
    2092: 3.56,
    2093: 3.55,
    2094: 3.55,
    2095: 3.55,
    2096: 3.55,
    2097: 3.55,
    2098: 3.55,
    2099: 3.55,
    2100: 3.55,
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


def _get_parameter_by_name(parameters, name: str):
    current = parameters
    for part in name.split("."):
        current = getattr(current, part)
    return current


def _trustees_wage_growth_for_tax_year(year: int) -> float:
    wage_year = year - 1
    try:
        return 1 + TRUSTEES_AVERAGE_WAGE_GROWTH_PCT[wage_year] / 100
    except KeyError as error:
        raise ValueError(
            "No SSA Trustees average-wage growth rate for "
            f"tax year {year} (calendar wage year {wage_year})."
        ) from error


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
    parameters,
    start_year: int,
    end_year: int,
    projection_base_year: int = 2026,
) -> None:
    metadata = getattr(parameter, "metadata", {})
    uprating = metadata.get("uprating")
    rounding = uprating.get("rounding") if isinstance(uprating, dict) else None
    uprating_name = _uprating_parameter_name(parameter)
    if uprating_name is None:
        return

    # Validate that the referenced default uprating parameter exists.
    _get_parameter_by_name(parameters, uprating_name)
    values_by_year = {}

    for year in range(projection_base_year + 1, start_year):
        values_by_year[year] = float(parameter(f"{year}-01-01"))

    for year in range(start_year, end_year + 1):
        if year - 1 in values_by_year:
            previous_value = values_by_year[year - 1]
        else:
            previous_value = float(parameter(f"{year - 1}-01-01"))
        wage_growth = _trustees_wage_growth_for_tax_year(year)
        updated_value = round_amount(previous_value * wage_growth, rounding)
        values_by_year[year] = updated_value

    for year, value in values_by_year.items():
        parameter.update(
            period=f"year:{year}-01-01:1",
            value=value,
        )


def _parameters_have_long_run_projection(parameters, end_year: int) -> bool:
    parameter = parameters.gov.irs.income.bracket.thresholds.children["1"].SINGLE
    return any(
        value.instant_str == f"{end_year}-01-01" for value in parameter.values_list
    )


def create_wage_indexed_core_thresholds_reform(
    *,
    start_year: int = 2035,
    end_year: int = 2100,
):
    if _pe_create_trustees_core_thresholds_reform is not None:
        return _pe_create_trustees_core_thresholds_reform(
            start_year=start_year,
            end_year=end_year,
        )

    from policyengine_us.model_api import Reform

    def modify_parameters(parameters):
        if not _parameters_have_long_run_projection(parameters, end_year):
            return parameters

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
                    parameters=parameters,
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
        if not _parameters_have_long_run_projection(parameters, end_year):
            return parameters

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
                parameters=parameters,
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
