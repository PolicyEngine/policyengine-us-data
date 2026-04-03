from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from policyengine_core.data.dataset import Dataset
from policyengine_us import Microsimulation


SS_COMPONENTS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
)
PAYROLL_COMPONENTS = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
)
PAYROLL_TRANSFER_COMPONENTS = PAYROLL_COMPONENTS + ("w2_wages_from_qualified_business",)
ENTITY_ID_COLUMNS = {
    "household": ("household_id", "person_household_id"),
    "family": ("family_id", "person_family_id"),
    "tax_unit": ("tax_unit_id", "person_tax_unit_id"),
    "spm_unit": ("spm_unit_id", "person_spm_unit_id"),
    "marital_unit": ("marital_unit_id", "person_marital_unit_id"),
}
PERSON_ID_COLUMN = "person_id"


ConstraintState = Literal["any", "positive", "nonpositive"]


@dataclass(frozen=True)
class AgeShiftCloneRule:
    name: str
    min_max_age: int
    max_max_age: int
    age_shift: int
    ss_state: ConstraintState = "any"
    payroll_state: ConstraintState = "any"
    clone_weight_scale: float = 0.25


@dataclass(frozen=True)
class CompositePayrollRule:
    name: str
    recipient_min_max_age: int
    recipient_max_max_age: int
    donor_min_max_age: int
    donor_max_max_age: int
    recipient_ss_state: ConstraintState = "positive"
    recipient_payroll_state: ConstraintState = "nonpositive"
    donor_ss_state: ConstraintState = "nonpositive"
    donor_payroll_state: ConstraintState = "positive"
    payroll_transfer_scale: float = 1.0
    clone_weight_scale: float = 0.25


@dataclass(frozen=True)
class SinglePersonSyntheticGridRule:
    name: str
    template_min_max_age: int
    template_max_max_age: int
    target_ages: tuple[int, ...]
    ss_quantiles: tuple[float, ...]
    payroll_quantiles: tuple[float, ...]
    template_ss_state: ConstraintState = "positive"
    template_payroll_state: ConstraintState = "any"
    payroll_donor_min_max_age: int = 45
    payroll_donor_max_max_age: int = 64
    payroll_donor_ss_state: ConstraintState = "nonpositive"
    payroll_donor_payroll_state: ConstraintState = "positive"
    payroll_scale_factors: tuple[float, ...] = (1.0,)
    clone_weight_scale: float = 0.1


@dataclass(frozen=True)
class MixedAgeAppendRule:
    name: str
    recipient_min_max_age: int
    recipient_max_max_age: int
    donor_min_max_age: int
    donor_max_max_age: int
    recipient_ss_state: ConstraintState = "positive"
    recipient_payroll_state: ConstraintState = "any"
    donor_ss_state: ConstraintState = "nonpositive"
    donor_payroll_state: ConstraintState = "positive"
    donor_age_shift: int = 0
    clone_weight_scale: float = 0.15


@dataclass(frozen=True)
class SupportAugmentationProfile:
    name: str
    description: str
    rules: tuple[
        AgeShiftCloneRule
        | CompositePayrollRule
        | SinglePersonSyntheticGridRule
        | MixedAgeAppendRule,
        ...,
    ]


LATE_CLONE_V1 = SupportAugmentationProfile(
    name="late-clone-v1",
    description=(
        "Age-shifted donor households to expand late-year support for older "
        "beneficiary, older beneficiary-plus-payroll, and payroll-only households."
    ),
    rules=(
        AgeShiftCloneRule(
            name="ss_only_65_74_to_75_84",
            min_max_age=65,
            max_max_age=74,
            age_shift=10,
            ss_state="positive",
            payroll_state="nonpositive",
            clone_weight_scale=0.35,
        ),
        AgeShiftCloneRule(
            name="ss_only_75_84_to_85_plus",
            min_max_age=75,
            max_max_age=84,
            age_shift=10,
            ss_state="positive",
            payroll_state="nonpositive",
            clone_weight_scale=0.5,
        ),
        AgeShiftCloneRule(
            name="ss_pay_65_74_to_75_84",
            min_max_age=65,
            max_max_age=74,
            age_shift=10,
            ss_state="positive",
            payroll_state="positive",
            clone_weight_scale=0.35,
        ),
        AgeShiftCloneRule(
            name="ss_pay_75_84_to_85_plus",
            min_max_age=75,
            max_max_age=84,
            age_shift=10,
            ss_state="positive",
            payroll_state="positive",
            clone_weight_scale=0.5,
        ),
        AgeShiftCloneRule(
            name="pay_only_55_64_to_65_74",
            min_max_age=55,
            max_max_age=64,
            age_shift=10,
            ss_state="nonpositive",
            payroll_state="positive",
            clone_weight_scale=0.2,
        ),
    ),
)

LATE_CLONE_V2 = SupportAugmentationProfile(
    name="late-clone-v2",
    description=(
        "More aggressive age-shifted donor households that test whether the "
        "late-year infeasibility is driven by missing older payroll-rich support."
    ),
    rules=(
        *LATE_CLONE_V1.rules,
        AgeShiftCloneRule(
            name="pay_only_45_64_to_75_84",
            min_max_age=45,
            max_max_age=64,
            age_shift=20,
            ss_state="nonpositive",
            payroll_state="positive",
            clone_weight_scale=0.15,
        ),
        AgeShiftCloneRule(
            name="pay_only_55_64_to_85_plus",
            min_max_age=55,
            max_max_age=64,
            age_shift=30,
            ss_state="nonpositive",
            payroll_state="positive",
            clone_weight_scale=0.1,
        ),
        AgeShiftCloneRule(
            name="ss_pay_65_74_to_85_plus",
            min_max_age=65,
            max_max_age=74,
            age_shift=20,
            ss_state="positive",
            payroll_state="positive",
            clone_weight_scale=0.2,
        ),
    ),
)

LATE_COMPOSITE_V1 = SupportAugmentationProfile(
    name="late-composite-v1",
    description=(
        "Composite synthetic households that preserve older beneficiary age/SS "
        "structure while injecting payroll from younger payroll-rich donors."
    ),
    rules=(
        CompositePayrollRule(
            name="ss_only_75_84_plus_payroll_from_55_64",
            recipient_min_max_age=75,
            recipient_max_max_age=84,
            donor_min_max_age=55,
            donor_max_max_age=64,
            recipient_ss_state="positive",
            recipient_payroll_state="nonpositive",
            donor_ss_state="nonpositive",
            donor_payroll_state="positive",
            payroll_transfer_scale=1.0,
            clone_weight_scale=0.2,
        ),
        CompositePayrollRule(
            name="ss_only_85_plus_plus_payroll_from_55_64",
            recipient_min_max_age=85,
            recipient_max_max_age=85,
            donor_min_max_age=55,
            donor_max_max_age=64,
            recipient_ss_state="positive",
            recipient_payroll_state="nonpositive",
            donor_ss_state="nonpositive",
            donor_payroll_state="positive",
            payroll_transfer_scale=0.75,
            clone_weight_scale=0.15,
        ),
        CompositePayrollRule(
            name="ss_pay_75_84_boost_from_45_64",
            recipient_min_max_age=75,
            recipient_max_max_age=84,
            donor_min_max_age=45,
            donor_max_max_age=64,
            recipient_ss_state="positive",
            recipient_payroll_state="positive",
            donor_ss_state="nonpositive",
            donor_payroll_state="positive",
            payroll_transfer_scale=0.5,
            clone_weight_scale=0.15,
        ),
    ),
)

LATE_COMPOSITE_V2 = SupportAugmentationProfile(
    name="late-composite-v2",
    description=(
        "Extreme composite synthetic households for diagnosing whether the late "
        "frontier is limited by missing older payroll intensity."
    ),
    rules=(
        CompositePayrollRule(
            name="ss_only_75_84_plus_heavy_payroll_from_45_64",
            recipient_min_max_age=75,
            recipient_max_max_age=84,
            donor_min_max_age=45,
            donor_max_max_age=64,
            recipient_ss_state="positive",
            recipient_payroll_state="nonpositive",
            donor_ss_state="nonpositive",
            donor_payroll_state="positive",
            payroll_transfer_scale=3.0,
            clone_weight_scale=0.15,
        ),
        CompositePayrollRule(
            name="ss_only_85_plus_heavy_payroll_from_45_64",
            recipient_min_max_age=85,
            recipient_max_max_age=85,
            donor_min_max_age=45,
            donor_max_max_age=64,
            recipient_ss_state="positive",
            recipient_payroll_state="nonpositive",
            donor_ss_state="nonpositive",
            donor_payroll_state="positive",
            payroll_transfer_scale=2.0,
            clone_weight_scale=0.1,
        ),
        CompositePayrollRule(
            name="ss_pay_75_84_heavy_boost_from_45_64",
            recipient_min_max_age=75,
            recipient_max_max_age=84,
            donor_min_max_age=45,
            donor_max_max_age=64,
            recipient_ss_state="positive",
            recipient_payroll_state="positive",
            donor_ss_state="nonpositive",
            donor_payroll_state="positive",
            payroll_transfer_scale=1.5,
            clone_weight_scale=0.1,
        ),
    ),
)

LATE_SYNTHETIC_GRID_V1 = SupportAugmentationProfile(
    name="late-synthetic-grid-v1",
    description=(
        "Appended synthetic single-person older households on an explicit "
        "age/SS/payroll grid, preserving the base CPS support untouched."
    ),
    rules=(
        SinglePersonSyntheticGridRule(
            name="single_75_84_grid",
            template_min_max_age=75,
            template_max_max_age=84,
            target_ages=(77, 82),
            ss_quantiles=(0.25, 0.5, 0.75),
            payroll_quantiles=(0.25, 0.5, 0.75),
            template_ss_state="positive",
            template_payroll_state="any",
            payroll_donor_min_max_age=45,
            payroll_donor_max_max_age=64,
            clone_weight_scale=0.1,
        ),
        SinglePersonSyntheticGridRule(
            name="single_85_plus_grid",
            template_min_max_age=85,
            template_max_max_age=85,
            target_ages=(85,),
            ss_quantiles=(0.25, 0.5, 0.75),
            payroll_quantiles=(0.25, 0.5, 0.75),
            template_ss_state="positive",
            template_payroll_state="any",
            payroll_donor_min_max_age=45,
            payroll_donor_max_max_age=64,
            clone_weight_scale=0.08,
        ),
    ),
)

LATE_SYNTHETIC_GRID_V2 = SupportAugmentationProfile(
    name="late-synthetic-grid-v2",
    description=(
        "Appended synthetic older-worker grid with more extreme payroll levels "
        "to test whether older payroll intensity alone can move the late frontier."
    ),
    rules=(
        SinglePersonSyntheticGridRule(
            name="single_75_84_extreme_grid",
            template_min_max_age=75,
            template_max_max_age=84,
            target_ages=(77, 82),
            ss_quantiles=(0.25, 0.5, 0.75),
            payroll_quantiles=(0.25, 0.5, 0.75),
            payroll_scale_factors=(1.0, 2.0, 4.0),
            template_ss_state="positive",
            template_payroll_state="any",
            payroll_donor_min_max_age=45,
            payroll_donor_max_max_age=64,
            clone_weight_scale=0.08,
        ),
        SinglePersonSyntheticGridRule(
            name="single_85_plus_extreme_grid",
            template_min_max_age=85,
            template_max_max_age=85,
            target_ages=(85,),
            ss_quantiles=(0.25, 0.5, 0.75),
            payroll_quantiles=(0.25, 0.5, 0.75),
            payroll_scale_factors=(1.0, 2.0, 4.0),
            template_ss_state="positive",
            template_payroll_state="any",
            payroll_donor_min_max_age=45,
            payroll_donor_max_max_age=64,
            clone_weight_scale=0.06,
        ),
    ),
)

LATE_MIXED_HOUSEHOLD_V1 = SupportAugmentationProfile(
    name="late-mixed-household-v1",
    description=(
        "Append synthetic mixed-age households by combining older beneficiary "
        "households with a younger payroll-rich donor person as a separate "
        "subunit in the same household."
    ),
    rules=(
        MixedAgeAppendRule(
            name="older_75_84_plus_prime_age_earner",
            recipient_min_max_age=75,
            recipient_max_max_age=84,
            donor_min_max_age=35,
            donor_max_max_age=64,
            recipient_ss_state="positive",
            recipient_payroll_state="any",
            donor_ss_state="nonpositive",
            donor_payroll_state="positive",
            clone_weight_scale=0.12,
        ),
        MixedAgeAppendRule(
            name="older_85_plus_prime_age_earner",
            recipient_min_max_age=85,
            recipient_max_max_age=85,
            donor_min_max_age=35,
            donor_max_max_age=64,
            recipient_ss_state="positive",
            recipient_payroll_state="any",
            donor_ss_state="nonpositive",
            donor_payroll_state="positive",
            clone_weight_scale=0.08,
        ),
    ),
)


NAMED_SUPPORT_AUGMENTATION_PROFILES = {
    LATE_CLONE_V1.name: LATE_CLONE_V1,
    LATE_CLONE_V2.name: LATE_CLONE_V2,
    LATE_COMPOSITE_V1.name: LATE_COMPOSITE_V1,
    LATE_COMPOSITE_V2.name: LATE_COMPOSITE_V2,
    LATE_SYNTHETIC_GRID_V1.name: LATE_SYNTHETIC_GRID_V1,
    LATE_SYNTHETIC_GRID_V2.name: LATE_SYNTHETIC_GRID_V2,
    LATE_MIXED_HOUSEHOLD_V1.name: LATE_MIXED_HOUSEHOLD_V1,
}


def _period_column(name: str, base_year: int) -> str:
    return f"{name}__{base_year}"


def get_support_augmentation_profile(name: str) -> SupportAugmentationProfile:
    try:
        return NAMED_SUPPORT_AUGMENTATION_PROFILES[name]
    except KeyError as error:
        valid = ", ".join(sorted(NAMED_SUPPORT_AUGMENTATION_PROFILES))
        raise ValueError(
            f"Unknown support augmentation profile '{name}'. Valid profiles: {valid}"
        ) from error


def household_support_summary(
    input_df: pd.DataFrame,
    *,
    base_year: int,
) -> pd.DataFrame:
    household_id_col = _period_column("household_id", base_year)
    household_weight_col = _period_column("household_weight", base_year)
    age_col = _period_column("age", base_year)

    required = [household_id_col, household_weight_col, age_col]
    required.extend(_period_column(component, base_year) for component in SS_COMPONENTS)
    required.extend(
        _period_column(component, base_year) for component in PAYROLL_COMPONENTS
    )
    missing = [column for column in required if column not in input_df.columns]
    if missing:
        raise ValueError(
            "Input dataframe is missing required support columns: "
            + ", ".join(sorted(missing))
        )

    aggregations: dict[str, str] = {
        age_col: "max",
        household_weight_col: "max",
    }
    aggregations.update(
        {_period_column(component, base_year): "sum" for component in SS_COMPONENTS}
    )
    aggregations.update(
        {
            _period_column(component, base_year): "sum"
            for component in PAYROLL_COMPONENTS
        }
    )

    summary = (
        input_df.groupby(household_id_col, sort=False)
        .agg(aggregations)
        .rename(
            columns={
                age_col: "max_age",
                household_weight_col: "baseline_weight",
            }
        )
    )
    summary["ss_total"] = summary[
        [_period_column(component, base_year) for component in SS_COMPONENTS]
    ].sum(axis=1)
    summary["payroll_total"] = summary[
        [_period_column(component, base_year) for component in PAYROLL_COMPONENTS]
    ].sum(axis=1)
    summary["household_size"] = input_df.groupby(household_id_col, sort=False)[
        _period_column(PERSON_ID_COLUMN, base_year)
    ].count()
    return summary


def _match_state(values: pd.Series, state: ConstraintState) -> pd.Series:
    if state == "any":
        return pd.Series(True, index=values.index)
    if state == "positive":
        return values > 0
    if state == "nonpositive":
        return values <= 0
    raise ValueError(f"Unsupported state '{state}'")


def _age_range_mask(
    ages: pd.Series,
    *,
    min_max_age: int,
    max_max_age: int,
) -> pd.Series:
    if min_max_age == 85 and max_max_age == 85:
        return ages >= 85
    return ages.between(min_max_age, max_max_age)


def select_donor_households(
    summary: pd.DataFrame,
    rule: AgeShiftCloneRule,
) -> pd.Index:
    age_mask = _age_range_mask(
        summary["max_age"],
        min_max_age=rule.min_max_age,
        max_max_age=rule.max_max_age,
    )
    positive_weight_mask = summary["baseline_weight"] > 0
    ss_mask = _match_state(summary["ss_total"], rule.ss_state)
    payroll_mask = _match_state(summary["payroll_total"], rule.payroll_state)
    return summary.index[age_mask & positive_weight_mask & ss_mask & payroll_mask]


def select_households_for_composite_rule(
    summary: pd.DataFrame,
    *,
    min_max_age: int,
    max_max_age: int,
    ss_state: ConstraintState,
    payroll_state: ConstraintState,
) -> pd.Index:
    age_mask = _age_range_mask(
        summary["max_age"],
        min_max_age=min_max_age,
        max_max_age=max_max_age,
    )
    positive_weight_mask = summary["baseline_weight"] > 0
    ss_mask = _match_state(summary["ss_total"], ss_state)
    payroll_mask = _match_state(summary["payroll_total"], payroll_state)
    return summary.index[age_mask & positive_weight_mask & ss_mask & payroll_mask]


def _next_entity_id(values: pd.Series) -> int:
    non_null = values.dropna()
    if non_null.empty:
        return 1
    return int(non_null.max()) + 1


def _cast_mapped_ids(series: pd.Series, mapped: pd.Series) -> pd.Series:
    dtype = series.dtype
    if pd.api.types.is_integer_dtype(dtype):
        return mapped.astype(dtype)
    if pd.api.types.is_float_dtype(dtype):
        return mapped.astype(dtype)
    return mapped


def clone_households_with_age_shift(
    input_df: pd.DataFrame,
    *,
    base_year: int,
    household_ids: pd.Index,
    age_shift: int,
    clone_weight_scale: float,
    id_counters: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if household_ids.empty:
        return input_df.iloc[0:0].copy(), (
            id_counters.copy() if id_counters is not None else {}
        )

    household_id_col = _period_column("household_id", base_year)
    person_id_col = _period_column(PERSON_ID_COLUMN, base_year)
    age_col = _period_column("age", base_year)
    household_weight_col = _period_column("household_weight", base_year)
    person_weight_col = _period_column("person_weight", base_year)

    donors = input_df[input_df[household_id_col].isin(household_ids)].copy()

    next_ids = (
        id_counters.copy()
        if id_counters is not None
        else {
            entity_name: _next_entity_id(
                input_df[_period_column(columns[0], base_year)]
            )
            for entity_name, columns in ENTITY_ID_COLUMNS.items()
        }
    )
    if "person" not in next_ids:
        next_ids["person"] = _next_entity_id(input_df[person_id_col])

    household_map = {
        original_id: next_ids["household"] + offset
        for offset, original_id in enumerate(
            pd.unique(donors[household_id_col].dropna())
        )
    }
    next_ids["household"] += len(household_map)

    for entity_name, columns in ENTITY_ID_COLUMNS.items():
        column = _period_column(columns[0], base_year)
        if entity_name == "household" or column not in donors.columns:
            continue
        unique_ids = pd.unique(donors[column].dropna())
        mapping = {
            original_id: next_ids[entity_name] + offset
            for offset, original_id in enumerate(unique_ids)
        }
        next_ids[entity_name] += len(mapping)
        for raw_column in columns:
            mapped_column = _period_column(raw_column, base_year)
            if mapped_column not in donors.columns:
                continue
            mapped = donors[mapped_column].map(mapping)
            donors[mapped_column] = _cast_mapped_ids(donors[mapped_column], mapped)

    person_map = {
        original_id: next_ids["person"] + offset
        for offset, original_id in enumerate(pd.unique(donors[person_id_col].dropna()))
    }
    next_ids["person"] += len(person_map)
    donors[person_id_col] = _cast_mapped_ids(
        donors[person_id_col], donors[person_id_col].map(person_map)
    )

    for raw_column in ENTITY_ID_COLUMNS["household"]:
        mapped_column = _period_column(raw_column, base_year)
        donors[mapped_column] = _cast_mapped_ids(
            donors[mapped_column], donors[mapped_column].map(household_map)
        )

    donors[age_col] = np.minimum(donors[age_col].astype(float) + age_shift, 85)

    if household_weight_col in donors.columns:
        donors[household_weight_col] = (
            donors[household_weight_col].astype(float) * clone_weight_scale
        )
    if person_weight_col in donors.columns:
        donors[person_weight_col] = (
            donors[person_weight_col].astype(float) * clone_weight_scale
        )

    return donors, next_ids


def _household_component_totals(
    input_df: pd.DataFrame,
    *,
    base_year: int,
    components: tuple[str, ...],
) -> pd.DataFrame:
    household_id_col = _period_column("household_id", base_year)
    available = [
        _period_column(component, base_year)
        for component in components
        if _period_column(component, base_year) in input_df.columns
    ]
    if not available:
        return pd.DataFrame(index=pd.Index([], dtype=int))
    return input_df.groupby(household_id_col, sort=False)[available].sum()


def _quantile_pair_households(
    recipient_ids: pd.Index,
    donor_ids: pd.Index,
    summary: pd.DataFrame,
) -> list[tuple[int, int]]:
    if recipient_ids.empty or donor_ids.empty:
        return []
    recipient_order = (
        summary.loc[recipient_ids]
        .sort_values(["ss_total", "baseline_weight", "max_age"])
        .index.to_list()
    )
    donor_order = (
        summary.loc[donor_ids]
        .sort_values(["payroll_total", "baseline_weight", "max_age"])
        .index.to_list()
    )
    if len(donor_order) == 1:
        donor_positions = np.zeros(len(recipient_order), dtype=int)
    else:
        donor_positions = (
            np.linspace(
                0,
                len(donor_order) - 1,
                num=len(recipient_order),
            )
            .round()
            .astype(int)
        )
    return [
        (int(recipient_household_id), int(donor_order[position]))
        for recipient_household_id, position in zip(recipient_order, donor_positions)
    ]


def _select_payroll_target_row(
    household_rows: pd.DataFrame,
    *,
    base_year: int,
) -> Any:
    age_col = _period_column("age", base_year)
    employment_col = _period_column("employment_income_before_lsr", base_year)
    self_employment_col = _period_column("self_employment_income_before_lsr", base_year)
    adults = household_rows[household_rows[age_col] >= 18]
    if adults.empty:
        adults = household_rows
    existing_payroll = adults.get(employment_col, 0).astype(float) + adults.get(
        self_employment_col, 0
    ).astype(float)
    if existing_payroll.gt(0).any():
        return existing_payroll.idxmax()
    return adults[age_col].astype(float).idxmax()


def _clone_single_donor_person_row(
    donor_row: pd.Series,
    *,
    base_year: int,
    shared_household_id: int,
    household_weight: float,
    person_weight: float,
    donor_age_shift: int,
    id_counters: dict[str, int],
) -> tuple[pd.Series, dict[str, int]]:
    cloned = donor_row.copy()
    person_id_col = _period_column(PERSON_ID_COLUMN, base_year)
    household_weight_col = _period_column("household_weight", base_year)
    person_weight_col = _period_column("person_weight", base_year)
    age_col = _period_column("age", base_year)

    cloned[person_id_col] = id_counters["person"]
    id_counters["person"] += 1

    for raw_column in ENTITY_ID_COLUMNS["household"]:
        column = _period_column(raw_column, base_year)
        cloned[column] = shared_household_id

    for entity_name, columns in ENTITY_ID_COLUMNS.items():
        if entity_name == "household":
            continue
        entity_id = id_counters[entity_name]
        id_counters[entity_name] += 1
        for raw_column in columns:
            column = _period_column(raw_column, base_year)
            if column in cloned.index:
                cloned[column] = entity_id

    cloned[age_col] = min(float(cloned[age_col]) + donor_age_shift, 85)
    if household_weight_col in cloned.index:
        cloned[household_weight_col] = household_weight
    if person_weight_col in cloned.index:
        cloned[person_weight_col] = person_weight
    return cloned, id_counters


def synthesize_composite_households(
    input_df: pd.DataFrame,
    *,
    base_year: int,
    summary: pd.DataFrame,
    rule: CompositePayrollRule,
    id_counters: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, Any]]:
    recipient_ids = select_households_for_composite_rule(
        summary,
        min_max_age=rule.recipient_min_max_age,
        max_max_age=rule.recipient_max_max_age,
        ss_state=rule.recipient_ss_state,
        payroll_state=rule.recipient_payroll_state,
    )
    donor_ids = select_households_for_composite_rule(
        summary,
        min_max_age=rule.donor_min_max_age,
        max_max_age=rule.donor_max_max_age,
        ss_state=rule.donor_ss_state,
        payroll_state=rule.donor_payroll_state,
    )
    recipient_pairs = _quantile_pair_households(recipient_ids, donor_ids, summary)
    if not recipient_pairs:
        return (
            input_df.iloc[0:0].copy(),
            id_counters.copy() if id_counters is not None else {},
            {
                "rule": rule.name,
                "recipient_household_count": 0,
                "donor_household_count": int(len(donor_ids)),
                "composite_household_count": 0,
                "composite_person_count": 0,
                "payroll_transfer_scale": rule.payroll_transfer_scale,
            },
        )

    clone_df, next_ids = clone_households_with_age_shift(
        input_df,
        base_year=base_year,
        household_ids=pd.Index([recipient for recipient, _ in recipient_pairs]),
        age_shift=0,
        clone_weight_scale=rule.clone_weight_scale,
        id_counters=id_counters,
    )
    household_id_col = _period_column("household_id", base_year)
    original_household_col = _period_column("person_household_id", base_year)
    payroll_totals = _household_component_totals(
        input_df,
        base_year=base_year,
        components=PAYROLL_TRANSFER_COMPONENTS,
    )

    original_recipients = pd.unique(
        input_df[
            input_df[household_id_col].isin(
                [recipient for recipient, _ in recipient_pairs]
            )
        ][household_id_col]
    )
    cloned_household_ids = pd.unique(clone_df[household_id_col])
    cloned_mapping = {
        int(original): int(cloned)
        for original, cloned in zip(original_recipients, cloned_household_ids)
    }

    employment_col = _period_column("employment_income_before_lsr", base_year)
    self_employment_col = _period_column("self_employment_income_before_lsr", base_year)
    qbi_col = _period_column("w2_wages_from_qualified_business", base_year)

    for recipient_household_id, donor_household_id in recipient_pairs:
        cloned_household_id = cloned_mapping[int(recipient_household_id)]
        mask = clone_df[household_id_col] == cloned_household_id
        target_row = _select_payroll_target_row(
            clone_df.loc[mask],
            base_year=base_year,
        )
        donor_row = payroll_totals.loc[int(donor_household_id)]
        clone_df.loc[target_row, employment_col] = (
            clone_df.loc[target_row, employment_col]
            + float(donor_row.get(employment_col, 0.0)) * rule.payroll_transfer_scale
        )
        clone_df.loc[target_row, self_employment_col] = (
            clone_df.loc[target_row, self_employment_col]
            + float(donor_row.get(self_employment_col, 0.0))
            * rule.payroll_transfer_scale
        )
        if qbi_col in clone_df.columns and qbi_col in donor_row.index:
            clone_df.loc[target_row, qbi_col] = (
                clone_df.loc[target_row, qbi_col]
                + float(donor_row.get(qbi_col, 0.0)) * rule.payroll_transfer_scale
            )

    return (
        clone_df,
        next_ids,
        {
            "rule": rule.name,
            "recipient_household_count": int(len(recipient_ids)),
            "donor_household_count": int(len(donor_ids)),
            "composite_household_count": int(clone_df[household_id_col].nunique()),
            "composite_person_count": int(len(clone_df)),
            "payroll_transfer_scale": rule.payroll_transfer_scale,
        },
    )


def synthesize_single_person_grid_households(
    input_df: pd.DataFrame,
    *,
    base_year: int,
    summary: pd.DataFrame,
    rule: SinglePersonSyntheticGridRule,
    id_counters: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, Any]]:
    template_ids = select_households_for_composite_rule(
        summary[summary["household_size"] == 1],
        min_max_age=rule.template_min_max_age,
        max_max_age=rule.template_max_max_age,
        ss_state=rule.template_ss_state,
        payroll_state=rule.template_payroll_state,
    )
    donor_ids = select_households_for_composite_rule(
        summary[summary["household_size"] == 1],
        min_max_age=rule.payroll_donor_min_max_age,
        max_max_age=rule.payroll_donor_max_max_age,
        ss_state=rule.payroll_donor_ss_state,
        payroll_state=rule.payroll_donor_payroll_state,
    )
    if template_ids.empty or donor_ids.empty:
        return (
            input_df.iloc[0:0].copy(),
            id_counters.copy() if id_counters is not None else {},
            {
                "rule": rule.name,
                "template_household_count": int(len(template_ids)),
                "payroll_donor_household_count": int(len(donor_ids)),
                "synthetic_household_count": 0,
                "synthetic_person_count": 0,
            },
        )

    household_id_col = _period_column("household_id", base_year)
    age_col = _period_column("age", base_year)
    person_id_col = _period_column(PERSON_ID_COLUMN, base_year)
    household_weight_col = _period_column("household_weight", base_year)
    person_weight_col = _period_column("person_weight", base_year)
    employment_col = _period_column("employment_income_before_lsr", base_year)
    self_employment_col = _period_column("self_employment_income_before_lsr", base_year)
    qbi_col = _period_column("w2_wages_from_qualified_business", base_year)

    template_rows = (
        input_df[input_df[household_id_col].isin(template_ids)]
        .sort_values([age_col, household_id_col])
        .copy()
    )
    ss_values = summary.loc[template_ids, "ss_total"].astype(float)
    payroll_values = summary.loc[donor_ids, "payroll_total"].astype(float)
    ss_targets = [float(ss_values.quantile(q)) for q in rule.ss_quantiles]
    payroll_targets = [
        float(payroll_values.quantile(q) * scale_factor)
        for q in rule.payroll_quantiles
        for scale_factor in rule.payroll_scale_factors
    ]

    next_ids = (
        id_counters.copy()
        if id_counters is not None
        else {
            entity_name: _next_entity_id(
                input_df[_period_column(columns[0], base_year)]
            )
            for entity_name, columns in ENTITY_ID_COLUMNS.items()
        }
    )
    if "person" not in next_ids:
        next_ids["person"] = _next_entity_id(input_df[person_id_col])

    synthetic_rows = []
    template_records = template_rows.to_dict("records")
    template_index = 0
    for target_age in rule.target_ages:
        for ss_target in ss_targets:
            for payroll_target in payroll_targets:
                base_row = template_records[
                    template_index % len(template_records)
                ].copy()
                template_index += 1
                household_id = next_ids["household"]
                next_ids["household"] += 1
                person_id = next_ids["person"]
                next_ids["person"] += 1

                for entity_name, columns in ENTITY_ID_COLUMNS.items():
                    entity_id = next_ids[entity_name]
                    next_ids[entity_name] += 1
                    for raw_column in columns:
                        column = _period_column(raw_column, base_year)
                        base_row[column] = entity_id

                base_row[household_id_col] = household_id
                base_row[_period_column("person_household_id", base_year)] = (
                    household_id
                )
                base_row[person_id_col] = person_id
                base_row[age_col] = float(target_age)
                base_row[_period_column("social_security_retirement", base_year)] = (
                    float(ss_target)
                )
                base_row[_period_column("social_security_disability", base_year)] = 0.0
                base_row[_period_column("social_security_survivors", base_year)] = 0.0
                base_row[_period_column("social_security_dependents", base_year)] = 0.0
                base_row[employment_col] = float(payroll_target)
                base_row[self_employment_col] = 0.0
                if qbi_col in base_row:
                    base_row[qbi_col] = 0.0
                if household_weight_col in base_row:
                    base_row[household_weight_col] = (
                        float(base_row[household_weight_col]) * rule.clone_weight_scale
                    )
                if person_weight_col in base_row:
                    base_row[person_weight_col] = (
                        float(base_row[person_weight_col]) * rule.clone_weight_scale
                    )
                synthetic_rows.append(base_row)

    synthetic_df = pd.DataFrame(synthetic_rows, columns=input_df.columns)
    return (
        synthetic_df,
        next_ids,
        {
            "rule": rule.name,
            "template_household_count": int(len(template_ids)),
            "payroll_donor_household_count": int(len(donor_ids)),
            "synthetic_household_count": int(synthetic_df[household_id_col].nunique()),
            "synthetic_person_count": int(len(synthetic_df)),
            "target_age_count": int(len(rule.target_ages)),
            "ss_grid_size": int(len(rule.ss_quantiles)),
            "payroll_grid_size": int(len(rule.payroll_quantiles)),
            "payroll_scale_factor_count": int(len(rule.payroll_scale_factors)),
        },
    )


def synthesize_mixed_age_households(
    input_df: pd.DataFrame,
    *,
    base_year: int,
    summary: pd.DataFrame,
    rule: MixedAgeAppendRule,
    id_counters: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, Any]]:
    recipient_ids = select_households_for_composite_rule(
        summary,
        min_max_age=rule.recipient_min_max_age,
        max_max_age=rule.recipient_max_max_age,
        ss_state=rule.recipient_ss_state,
        payroll_state=rule.recipient_payroll_state,
    )
    donor_ids = select_households_for_composite_rule(
        summary,
        min_max_age=rule.donor_min_max_age,
        max_max_age=rule.donor_max_max_age,
        ss_state=rule.donor_ss_state,
        payroll_state=rule.donor_payroll_state,
    )
    recipient_pairs = _quantile_pair_households(recipient_ids, donor_ids, summary)
    if not recipient_pairs:
        return (
            input_df.iloc[0:0].copy(),
            id_counters.copy() if id_counters is not None else {},
            {
                "rule": rule.name,
                "recipient_household_count": int(len(recipient_ids)),
                "donor_household_count": int(len(donor_ids)),
                "synthetic_household_count": 0,
                "synthetic_person_count": 0,
            },
        )

    recipient_clone_df, next_ids = clone_households_with_age_shift(
        input_df,
        base_year=base_year,
        household_ids=pd.Index([recipient for recipient, _ in recipient_pairs]),
        age_shift=0,
        clone_weight_scale=rule.clone_weight_scale,
        id_counters=id_counters,
    )

    household_id_col = _period_column("household_id", base_year)
    person_weight_col = _period_column("person_weight", base_year)

    original_recipients = pd.unique(
        input_df[
            input_df[household_id_col].isin(
                [recipient for recipient, _ in recipient_pairs]
            )
        ][household_id_col]
    )
    cloned_household_ids = pd.unique(recipient_clone_df[household_id_col])
    cloned_mapping = {
        int(original): int(cloned)
        for original, cloned in zip(original_recipients, cloned_household_ids)
    }

    donor_person_rows: list[pd.Series] = []
    for recipient_household_id, donor_household_id in recipient_pairs:
        cloned_household_id = cloned_mapping[int(recipient_household_id)]
        cloned_household_rows = recipient_clone_df[
            recipient_clone_df[household_id_col] == cloned_household_id
        ]
        donor_rows = input_df[input_df[household_id_col] == int(donor_household_id)]
        donor_row_idx = _select_payroll_target_row(
            donor_rows,
            base_year=base_year,
        )
        household_weight = float(
            cloned_household_rows.iloc[0][_period_column("household_weight", base_year)]
        )
        person_weight = (
            float(cloned_household_rows.iloc[0][person_weight_col])
            if person_weight_col in cloned_household_rows.columns
            else household_weight
        )
        donor_row, next_ids = _clone_single_donor_person_row(
            donor_rows.loc[donor_row_idx],
            base_year=base_year,
            shared_household_id=cloned_household_id,
            household_weight=household_weight,
            person_weight=person_weight,
            donor_age_shift=rule.donor_age_shift,
            id_counters=next_ids,
        )
        donor_person_rows.append(donor_row)

    donor_df = pd.DataFrame(donor_person_rows, columns=input_df.columns)
    synthetic_df = pd.concat([recipient_clone_df, donor_df], ignore_index=True)
    return (
        synthetic_df,
        next_ids,
        {
            "rule": rule.name,
            "recipient_household_count": int(len(recipient_ids)),
            "donor_household_count": int(len(donor_ids)),
            "synthetic_household_count": int(synthetic_df[household_id_col].nunique()),
            "synthetic_person_count": int(len(synthetic_df)),
            "donor_age_shift": rule.donor_age_shift,
        },
    )


def augment_input_dataframe(
    input_df: pd.DataFrame,
    *,
    base_year: int,
    profile: str | SupportAugmentationProfile,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    profile_obj = (
        get_support_augmentation_profile(profile)
        if isinstance(profile, str)
        else profile
    )
    summary = household_support_summary(input_df, base_year=base_year)

    clone_frames = []
    rule_reports: list[dict[str, Any]] = []
    id_counters = {
        entity_name: _next_entity_id(input_df[_period_column(columns[0], base_year)])
        for entity_name, columns in ENTITY_ID_COLUMNS.items()
    }
    id_counters["person"] = _next_entity_id(
        input_df[_period_column(PERSON_ID_COLUMN, base_year)]
    )
    for rule in profile_obj.rules:
        if isinstance(rule, AgeShiftCloneRule):
            donor_households = select_donor_households(summary, rule)
            clone_df, id_counters = clone_households_with_age_shift(
                input_df,
                base_year=base_year,
                household_ids=donor_households,
                age_shift=rule.age_shift,
                clone_weight_scale=rule.clone_weight_scale,
                id_counters=id_counters,
            )
            clone_frames.append(clone_df)
            rule_reports.append(
                {
                    "rule": rule.name,
                    "donor_household_count": int(len(donor_households)),
                    "clone_household_count": int(
                        clone_df[_period_column("household_id", base_year)].nunique()
                    )
                    if not clone_df.empty
                    else 0,
                    "clone_person_count": int(len(clone_df)),
                    "age_shift": rule.age_shift,
                    "clone_weight_scale": rule.clone_weight_scale,
                }
            )
            continue

        if isinstance(rule, CompositePayrollRule):
            composite_df, id_counters, composite_report = (
                synthesize_composite_households(
                    input_df,
                    base_year=base_year,
                    summary=summary,
                    rule=rule,
                    id_counters=id_counters,
                )
            )
            clone_frames.append(composite_df)
            rule_reports.append(composite_report)
            continue

        if isinstance(rule, MixedAgeAppendRule):
            mixed_df, id_counters, mixed_report = synthesize_mixed_age_households(
                input_df,
                base_year=base_year,
                summary=summary,
                rule=rule,
                id_counters=id_counters,
            )
            clone_frames.append(mixed_df)
            rule_reports.append(mixed_report)
            continue

        synthetic_df, id_counters, synthetic_report = (
            synthesize_single_person_grid_households(
                input_df,
                base_year=base_year,
                summary=summary,
                rule=rule,
                id_counters=id_counters,
            )
        )
        clone_frames.append(synthetic_df)
        rule_reports.append(synthetic_report)

    if clone_frames:
        augmented_df = pd.concat([input_df, *clone_frames], ignore_index=True)
    else:
        augmented_df = input_df.copy()

    report = {
        "profile": profile_obj.name,
        "description": profile_obj.description,
        "base_household_count": int(
            input_df[_period_column("household_id", base_year)].nunique()
        ),
        "base_person_count": int(len(input_df)),
        "augmented_household_count": int(
            augmented_df[_period_column("household_id", base_year)].nunique()
        ),
        "augmented_person_count": int(len(augmented_df)),
        "rules": rule_reports,
    }
    return augmented_df, report


def build_augmented_dataset(
    *,
    base_dataset: str,
    base_year: int,
    profile: str | SupportAugmentationProfile,
) -> tuple[Dataset, dict[str, Any]]:
    sim = Microsimulation(dataset=base_dataset)
    input_df = sim.to_input_dataframe()
    augmented_df, report = augment_input_dataframe(
        input_df,
        base_year=base_year,
        profile=profile,
    )
    report["base_dataset"] = base_dataset
    return Dataset.from_dataframe(augmented_df, base_year), report
