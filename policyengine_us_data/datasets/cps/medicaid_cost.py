from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import numpy as np


@lru_cache(maxsize=1)
def _policyengine_us_parameters():
    from policyengine_us import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem().parameters


def add_medicaid_cost_if_enrolled_to_dataset(dataset) -> None:
    """Add person-level conditional Medicaid cost to an array-format dataset."""

    data = dataset.load_dataset()

    from policyengine_us import Microsimulation

    simulation = Microsimulation(dataset=dataset)
    values = calculate_medicaid_cost_if_enrolled(
        simulation=simulation,
        time_period=dataset.time_period,
    )
    data["medicaid_cost_if_enrolled"] = values.astype(np.float32)
    dataset.save_dataset(data)


def add_medicaid_cost_if_enrolled_to_time_period_data(
    data: dict,
    time_period: int,
    microsimulation_cls=None,
    dataset_cls=None,
) -> dict:
    """Add conditional Medicaid cost to time-period-array data."""

    if microsimulation_cls is None:
        from policyengine_us import Microsimulation

        microsimulation_cls = Microsimulation

    if dataset_cls is None:
        from policyengine_core.data import Dataset
        from policyengine_us_data.storage import STORAGE_FOLDER

        class InMemoryTimePeriodDataset(Dataset):
            name = "medicaid_cost_if_enrolled_build"
            label = "Medicaid cost build"
            data_format = Dataset.TIME_PERIOD_ARRAYS
            file_path = STORAGE_FOLDER / "medicaid_cost_if_enrolled_build.h5"

            def __init__(self, source_data: dict, source_time_period: int):
                self._data = source_data
                self.time_period = source_time_period
                super().__init__()

            def load(self):
                return self._data

            def load_dataset(self):
                return self._data

        dataset_cls = InMemoryTimePeriodDataset

    simulation = microsimulation_cls(dataset=dataset_cls(data, time_period))
    values = calculate_medicaid_cost_if_enrolled(
        simulation=simulation,
        time_period=time_period,
    )
    data["medicaid_cost_if_enrolled"] = {time_period: values.astype(np.float32)}
    return data


def calculate_medicaid_cost_if_enrolled(
    simulation: Any, time_period: int
) -> np.ndarray:
    """Return SLCSP-indexed Medicaid cost if each person enrolled.

    The allocator uses only an SLCSP-derived age/location premium index for
    within-state variation. It normalizes by state against current Medicaid
    enrollees so baseline weighted Medicaid costs match state spending totals,
    while non-enrollees still receive a conditional cost for reform analysis.
    """

    person_slcsp = calculate_person_slcsp_cost_index(simulation, time_period)
    medicaid_enrolled = _calculate(
        simulation,
        "medicaid_enrolled",
        time_period,
        map_to="person",
    ).astype(bool)
    person_weight = _calculate(
        simulation,
        "person_weight",
        time_period,
        map_to="person",
    ).astype(float)
    state_codes = _as_str_array(
        _calculate(
            simulation,
            "state_code_str",
            time_period,
            map_to="person",
        )
    )
    spending = medicaid_spending_by_state(time_period)
    return allocate_medicaid_cost_if_enrolled_by_slcsp(
        person_slcsp=person_slcsp,
        medicaid_enrolled=medicaid_enrolled,
        person_weight=person_weight,
        state_codes=state_codes,
        state_spending=spending,
    )


def calculate_person_slcsp_cost_index(
    simulation: Any,
    time_period: int,
) -> np.ndarray:
    """Return an SLCSP premium index for each person."""

    age = _calculate(simulation, "age", time_period, map_to="person").astype(float)
    is_tax_unit_dependent = _calculate(
        simulation,
        "is_tax_unit_dependent",
        time_period,
        map_to="person",
    ).astype(bool)
    person_tax_unit_id = _calculate(
        simulation,
        "person_tax_unit_id",
        time_period,
        map_to="person",
    )
    tax_unit_id = _calculate(
        simulation,
        "tax_unit_id",
        time_period,
        map_to="tax_unit",
    )
    state_codes = _as_str_array(
        _calculate(simulation, "state_code_str", time_period, map_to="person")
    )
    rating_area = _calculate(
        simulation,
        "slcsp_rating_area_default",
        time_period,
        map_to="person",
    ).astype(int)
    base_cost = slcsp_age_0_by_state_rating_area(
        state_codes,
        rating_area,
        time_period,
    )
    age_rated_index = base_cost * age_curve_multiplier(age, state_codes, time_period)
    return np.clip(
        family_tier_slcsp_person_share(
            state_codes=state_codes,
            base_cost=base_cost,
            age=age,
            is_tax_unit_dependent=is_tax_unit_dependent,
            person_tax_unit_id=person_tax_unit_id,
            tax_unit_id=tax_unit_id,
            fallback=age_rated_index,
            time_period=time_period,
        ),
        0,
        None,
    )


def slcsp_age_0_by_state_rating_area(
    state_codes: np.ndarray,
    rating_areas: np.ndarray,
    time_period: int,
) -> np.ndarray:
    """Return the age-0 SLCSP premium by state and county-level rating area."""

    parameters = _policyengine_us_parameters()(f"{int(time_period)}-01-01")
    costs = parameters.gov.aca.state_rating_area_cost
    state_codes = _as_str_array(state_codes)
    rating_areas = np.asarray(rating_areas, dtype=int)
    output = np.zeros(len(state_codes), dtype=float)
    for state_value in np.unique(state_codes):
        state = str(state_value)
        state_mask = state_codes == state
        if state not in STATE_CODES:
            continue
        state_costs = costs[state]
        for rating_area in np.unique(rating_areas[state_mask]):
            safe_rating_area = str(int(rating_area))
            try:
                cost = float(state_costs[safe_rating_area])
            except KeyError:
                cost = float(state_costs["1"])
            output[state_mask & (rating_areas == rating_area)] = cost
    return output


def age_curve_multiplier(
    age: np.ndarray,
    state_codes: np.ndarray,
    time_period: int,
) -> np.ndarray:
    """Return ACA SLCSP age-curve multipliers by person."""

    parameters = _policyengine_us_parameters()(f"{int(time_period)}-01-01")
    curves = parameters.gov.aca.age_curves
    age = np.asarray(age, dtype=float)
    state_codes = _as_str_array(state_codes)
    result = np.asarray(curves.default.calc(age), dtype=float)
    state_specific_curves = {
        "AL": curves.al,
        "DC": curves.dc,
        "MA": curves.ma,
        "MN": curves.mn,
        "MS": curves.ms,
        "NY": curves.ny,
        "OR": curves["or"],
        "UT": curves.ut,
        "VT": curves.vt,
    }
    for state, curve in state_specific_curves.items():
        result = np.where(
            state_codes == state, _calculate_age_curve(curve, age), result
        )
    return result


def _calculate_age_curve(curve, age: np.ndarray) -> np.ndarray:
    if hasattr(curve, "calc"):
        return np.asarray(curve.calc(age), dtype=float)
    return np.full(len(age), float(curve), dtype=float)


def family_tier_slcsp_person_share(
    *,
    state_codes: np.ndarray,
    base_cost: np.ndarray,
    age: np.ndarray,
    is_tax_unit_dependent: np.ndarray,
    person_tax_unit_id: np.ndarray,
    tax_unit_id: np.ndarray,
    fallback: np.ndarray,
    time_period: int,
) -> np.ndarray:
    """Return tax-unit SLCSP shares for NY/VT family-tier states."""

    state_codes = _as_str_array(state_codes)
    output = np.asarray(fallback, dtype=float).copy()
    family_tier_mask = np.isin(state_codes, FAMILY_TIER_STATES)
    if not family_tier_mask.any():
        return output

    base_cost = np.asarray(base_cost, dtype=float)
    age = np.asarray(age, dtype=float)
    is_tax_unit_dependent = np.asarray(is_tax_unit_dependent, dtype=bool)
    person_tax_unit_id = np.asarray(person_tax_unit_id)
    tax_unit_id = np.asarray(tax_unit_id)

    parameters = _policyengine_us_parameters()(f"{int(time_period)}-01-01").gov.aca
    max_child_age = float(parameters.slcsp.max_child_age)
    dependent_child_age_threshold = float(
        parameters.family_tier_dependent_child_age_threshold
    )

    for unit_id in tax_unit_id:
        members = person_tax_unit_id == unit_id
        members = members & family_tier_mask
        if not members.any():
            continue
        member_states = state_codes[members]
        state = str(member_states[0])
        if state not in FAMILY_TIER_STATES:
            continue

        dependent_child = (age[members] <= max_child_age) | (
            is_tax_unit_dependent[members]
            & (age[members] < dependent_child_age_threshold)
        )
        adult_count = int(np.count_nonzero(~dependent_child))
        child_count = int(np.count_nonzero(dependent_child))
        member_count = adult_count + child_count
        if member_count == 0:
            continue

        multiplier = _family_tier_multiplier(
            state=state,
            adult_count=adult_count,
            child_count=child_count,
            parameters=parameters,
        )
        if multiplier is None:
            continue

        positive_base_cost = base_cost[members][base_cost[members] > 0]
        if not positive_base_cost.size:
            continue
        output[members] = float(np.mean(positive_base_cost)) * multiplier / member_count

    return output


def _family_tier_multiplier(
    *,
    state: str,
    adult_count: int,
    child_count: int,
    parameters,
) -> float | None:
    ratings = (
        parameters.family_tier_ratings.ny
        if state == "NY"
        else parameters.family_tier_ratings.vt
    )
    extra_adults = max(adult_count - 2, 0)
    one_adult = float(ratings.ONE_ADULT)

    if adult_count == 0:
        if state == "NY" and child_count > 0:
            return float(ratings.CHILD_ONLY)
        # Vermont has no child-only family-tier multiplier; retain fallback.
        return None
    if child_count == 0:
        base = one_adult if adult_count == 1 else float(ratings.TWO_ADULTS)
    elif adult_count == 1:
        base = float(ratings.ONE_ADULT_AND_ONE_OR_MORE_CHILDREN)
    else:
        base = float(ratings.TWO_ADULTS_AND_ONE_OR_MORE_CHILDREN)
    return base + extra_adults * one_adult


def medicaid_spending_by_state(time_period: int) -> dict[str, float]:
    """Return total Medicaid spending targets by state."""

    spending = _policyengine_us_parameters()(
        f"{int(time_period)}-01-01"
    ).calibration.gov.hhs.medicaid.totals.spending
    return {state: float(spending[state]) for state in STATE_CODES}


def allocate_medicaid_cost_if_enrolled_by_slcsp(
    *,
    person_slcsp: np.ndarray,
    medicaid_enrolled: np.ndarray,
    person_weight: np.ndarray,
    state_codes: np.ndarray,
    state_spending: Mapping[str, float],
) -> np.ndarray:
    """Allocate state Medicaid spending using only SLCSP as the cost index."""

    person_slcsp = np.nan_to_num(np.asarray(person_slcsp, dtype=float), nan=0)
    person_slcsp = np.clip(person_slcsp, 0, None)
    medicaid_enrolled = np.asarray(medicaid_enrolled, dtype=bool)
    person_weight = np.nan_to_num(np.asarray(person_weight, dtype=float), nan=0)
    state_codes = _as_str_array(state_codes)

    if not (
        len(person_slcsp)
        == len(medicaid_enrolled)
        == len(person_weight)
        == len(state_codes)
    ):
        raise ValueError("Medicaid cost allocator inputs must have the same length.")

    output = np.zeros(len(person_slcsp), dtype=float)
    positive_slcsp = person_slcsp > 0
    national_fallback = (
        float(np.mean(person_slcsp[positive_slcsp])) if positive_slcsp.any() else 1.0
    )

    for state, target in state_spending.items():
        state_mask = state_codes == state
        if not state_mask.any() or target <= 0:
            continue

        state_index = _fill_missing_slcsp(
            person_slcsp[state_mask],
            fallback=national_fallback,
        )
        enrolled_within_state = medicaid_enrolled[state_mask]
        if not enrolled_within_state.any():
            continue

        denominator = float(
            np.sum(
                person_weight[state_mask][enrolled_within_state]
                * state_index[enrolled_within_state]
            )
        )
        if denominator <= 0:
            continue

        output[state_mask] = float(target) * state_index / denominator

    return output


def _fill_missing_slcsp(values: np.ndarray, *, fallback: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values > 0
    if positive.any():
        fallback = float(np.mean(values[positive]))
    return np.where(positive, values, fallback)


def _calculate(
    simulation: Any,
    variable: str,
    time_period: int,
    *,
    map_to: str | None = None,
) -> np.ndarray:
    kwargs: dict[str, Any] = {"period": time_period}
    if map_to is not None:
        kwargs["map_to"] = map_to
    result = simulation.calculate(variable, **kwargs)
    if hasattr(result, "values"):
        result = result.values
    return np.asarray(result)


def _as_str_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind == "S":
        return np.char.decode(values.astype("S"), "utf-8")
    if values.dtype.kind == "O":
        return np.asarray(
            [
                value.decode("utf-8")
                if isinstance(value, (bytes, bytearray))
                else str(value)
                for value in values
            ]
        )
    return values.astype(str)


FAMILY_TIER_STATES = ("NY", "VT")

STATE_CODES = (
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "DC",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
)
