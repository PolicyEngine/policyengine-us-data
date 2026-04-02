from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd
from policyengine_core.data.dataset import Dataset
from policyengine_us import Microsimulation

try:
    from .calibration import (
        assess_nonnegative_feasibility,
        calibrate_entropy,
        calibrate_entropy_bounded,
        densify_lp_solution,
    )
    from .projection_utils import (
        aggregate_age_targets,
        build_age_bins,
        validate_projected_social_security_cap,
    )
    from .ssa_data import (
        get_long_term_target_source,
        load_ssa_age_projections,
        load_ssa_benefit_projections,
        load_taxable_payroll_projections,
        set_long_term_target_source,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from calibration import (
        assess_nonnegative_feasibility,
        calibrate_entropy,
        calibrate_entropy_bounded,
        densify_lp_solution,
    )
    from projection_utils import (
        aggregate_age_targets,
        build_age_bins,
        validate_projected_social_security_cap,
    )
    from ssa_data import (
        get_long_term_target_source,
        load_ssa_age_projections,
        load_ssa_benefit_projections,
        load_taxable_payroll_projections,
        set_long_term_target_source,
    )


DEFAULT_DATASET = "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"
DEFAULT_YEAR = 2100
BASE_YEAR = 2024
ENTITY_ID_COLUMNS = {
    "household": ("household_id", "person_household_id"),
    "family": ("family_id", "person_family_id"),
    "tax_unit": ("tax_unit_id", "person_tax_unit_id"),
    "spm_unit": ("spm_unit_id", "person_spm_unit_id"),
    "marital_unit": ("marital_unit_id", "person_marital_unit_id"),
}
PERSON_ID_COLUMN = "person_id"
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
PAYROLL_UPRATING_FACTOR_COLUMN = "__pe_payroll_uprating_factor"
SS_UPRATING_FACTOR_COLUMN = "__pe_ss_uprating_factor"


@dataclass(frozen=True)
class SyntheticTemplate:
    name: str
    head_ages: tuple[int, ...]
    spouse_age_offsets: tuple[int | None, ...]
    dependent_age_sets: tuple[tuple[int, ...], ...]
    ss_source: str
    payroll_source: str
    pension_source: str
    dividend_source: str
    ss_split: tuple[float, float]
    payroll_split: tuple[float, float]
    ss_scale_factors: tuple[float, ...] = (1.0,)
    payroll_scale_factors: tuple[float, ...] = (1.0,)
    pension_scale_factors: tuple[float, ...] = (1.0,)
    dividend_scale_factors: tuple[float, ...] = (1.0,)


@dataclass(frozen=True)
class SyntheticCandidate:
    archetype: str
    head_age: int
    spouse_age: int | None
    dependent_ages: tuple[int, ...]
    head_wages: float
    spouse_wages: float
    head_ss: float
    spouse_ss: float
    pension_income: float
    dividend_income: float

    @property
    def payroll_total(self) -> float:
        return float(self.head_wages + self.spouse_wages)

    def taxable_payroll_total(self, payroll_cap: float) -> float:
        return float(
            min(self.head_wages, payroll_cap) + min(self.spouse_wages, payroll_cap)
        )

    @property
    def ss_total(self) -> float:
        return float(self.head_ss + self.spouse_ss)

    def ages(self) -> list[int]:
        values = [self.head_age]
        if self.spouse_age is not None:
            values.append(self.spouse_age)
        values.extend(self.dependent_ages)
        return values

    def filing_status(self) -> str:
        return "joint" if self.spouse_age is not None else "single"

    def taxable_benefits_proxy(self) -> float:
        benefits = self.ss_total
        if benefits <= 0:
            return 0.0
        provisional_income = (
            self.payroll_total
            + self.pension_income
            + self.dividend_income
            + 0.5 * benefits
        )
        if self.filing_status() == "joint":
            base = 32_000.0
            adjusted = 44_000.0
            lesser_cap = 6_000.0
        else:
            base = 25_000.0
            adjusted = 34_000.0
            lesser_cap = 4_500.0

        if provisional_income <= base:
            return 0.0
        if provisional_income <= adjusted:
            return min(0.5 * benefits, 0.5 * (provisional_income - base))
        return min(
            0.85 * benefits,
            0.85 * (provisional_income - adjusted) + min(0.5 * benefits, lesser_cap),
        )


TEMPLATES = (
    SyntheticTemplate(
        name="older_beneficiary_single",
        head_ages=(62, 67, 72, 77, 82, 85),
        spouse_age_offsets=(None,),
        dependent_age_sets=((),),
        ss_source="older_beneficiary",
        payroll_source="zero",
        pension_source="older_asset",
        dividend_source="older_asset",
        ss_split=(1.0, 0.0),
        payroll_split=(0.0, 0.0),
        ss_scale_factors=(0.75, 1.0, 1.25),
        pension_scale_factors=(0.0, 1.0),
        dividend_scale_factors=(0.0, 1.0),
    ),
    SyntheticTemplate(
        name="older_beneficiary_couple",
        head_ages=(62, 67, 72, 77, 82, 85),
        spouse_age_offsets=(-2, -5, -8),
        dependent_age_sets=((),),
        ss_source="older_couple_beneficiary",
        payroll_source="zero",
        pension_source="older_asset",
        dividend_source="older_asset",
        ss_split=(0.55, 0.45),
        payroll_split=(0.0, 0.0),
        ss_scale_factors=(0.75, 1.0, 1.25),
        pension_scale_factors=(0.0, 1.0),
        dividend_scale_factors=(0.0, 1.0),
    ),
    SyntheticTemplate(
        name="older_worker_single",
        head_ages=(62, 65, 67, 70, 75, 80),
        spouse_age_offsets=(None,),
        dependent_age_sets=((),),
        ss_source="older_worker",
        payroll_source="older_worker",
        pension_source="older_asset",
        dividend_source="older_asset",
        ss_split=(1.0, 0.0),
        payroll_split=(1.0, 0.0),
        ss_scale_factors=(0.5, 0.75, 1.0, 1.25),
        payroll_scale_factors=(0.5, 1.0, 1.5, 2.0),
        pension_scale_factors=(0.0, 1.0),
        dividend_scale_factors=(0.0, 1.0),
    ),
    SyntheticTemplate(
        name="older_worker_couple",
        head_ages=(62, 65, 67, 70, 75, 80),
        spouse_age_offsets=(-2, -5),
        dependent_age_sets=((),),
        ss_source="older_worker",
        payroll_source="older_worker",
        pension_source="older_asset",
        dividend_source="older_asset",
        ss_split=(0.55, 0.45),
        payroll_split=(0.55, 0.45),
        ss_scale_factors=(0.5, 0.75, 1.0, 1.25),
        payroll_scale_factors=(0.5, 1.0, 1.5, 2.0),
        pension_scale_factors=(0.0, 1.0),
        dividend_scale_factors=(0.0, 1.0),
    ),
    SyntheticTemplate(
        name="mixed_retiree_worker_couple",
        head_ages=(62, 67, 72, 77, 82, 85),
        spouse_age_offsets=(-10, -15, -20, -25, -35),
        dependent_age_sets=((),),
        ss_source="older_beneficiary",
        payroll_source="prime_worker",
        pension_source="older_asset",
        dividend_source="older_asset",
        ss_split=(1.0, 0.0),
        payroll_split=(0.0, 1.0),
        ss_scale_factors=(0.5, 0.75, 1.0, 1.25),
        payroll_scale_factors=(0.5, 1.0, 1.5, 2.0),
        pension_scale_factors=(0.0, 1.0),
        dividend_scale_factors=(0.0, 1.0),
    ),
    SyntheticTemplate(
        name="prime_worker_single",
        head_ages=(20, 22, 25, 27, 30, 35, 40, 45, 50, 55, 60, 64),
        spouse_age_offsets=(None,),
        dependent_age_sets=((),),
        ss_source="zero",
        payroll_source="prime_worker",
        pension_source="prime_asset",
        dividend_source="prime_asset",
        ss_split=(0.0, 0.0),
        payroll_split=(1.0, 0.0),
        payroll_scale_factors=(0.5, 1.0, 1.5, 2.0),
        pension_scale_factors=(0.0, 1.0),
        dividend_scale_factors=(0.0, 1.0),
    ),
    SyntheticTemplate(
        name="prime_worker_couple",
        head_ages=(25, 30, 35, 40, 45, 50, 55, 60),
        spouse_age_offsets=(-2, -5, -8),
        dependent_age_sets=((),),
        ss_source="zero",
        payroll_source="prime_worker",
        pension_source="prime_asset",
        dividend_source="prime_asset",
        ss_split=(0.0, 0.0),
        payroll_split=(0.6, 0.4),
        payroll_scale_factors=(0.5, 1.0, 1.5, 2.0),
        pension_scale_factors=(0.0, 1.0),
        dividend_scale_factors=(0.0, 1.0),
    ),
    SyntheticTemplate(
        name="prime_worker_family",
        head_ages=(25, 30, 35, 40, 45, 50, 55),
        spouse_age_offsets=(-2,),
        dependent_age_sets=((0,), (3,), (7,), (12,), (16,), (4, 9), (11, 16)),
        ss_source="zero",
        payroll_source="prime_worker_family",
        pension_source="prime_asset",
        dividend_source="prime_asset",
        ss_split=(0.0, 0.0),
        payroll_split=(0.6, 0.4),
        payroll_scale_factors=(0.5, 1.0, 1.5, 2.0),
        pension_scale_factors=(0.0, 1.0),
        dividend_scale_factors=(0.0, 1.0),
    ),
    SyntheticTemplate(
        name="older_plus_prime_worker_family",
        head_ages=(62, 67, 72, 77, 82, 85),
        spouse_age_offsets=(-15, -25, -35),
        dependent_age_sets=((0,), (7,), (15,), (4, 9), (11, 16)),
        ss_source="older_beneficiary",
        payroll_source="prime_worker_family",
        pension_source="older_asset",
        dividend_source="older_asset",
        ss_split=(1.0, 0.0),
        payroll_split=(0.0, 1.0),
        ss_scale_factors=(0.5, 0.75, 1.0, 1.25),
        payroll_scale_factors=(0.5, 1.0, 1.5, 2.0),
        pension_scale_factors=(0.0, 1.0),
        dividend_scale_factors=(0.0, 1.0),
    ),
    SyntheticTemplate(
        name="late_worker_couple",
        head_ages=(58, 60, 62, 64, 66, 68),
        spouse_age_offsets=(-2, -5),
        dependent_age_sets=((),),
        ss_source="older_worker",
        payroll_source="prime_worker",
        pension_source="prime_asset",
        dividend_source="prime_asset",
        ss_split=(0.6, 0.4),
        payroll_split=(0.6, 0.4),
        ss_scale_factors=(0.25, 0.5, 0.75, 1.0),
        payroll_scale_factors=(0.5, 1.0, 1.5, 2.0),
        pension_scale_factors=(0.0, 1.0),
        dividend_scale_factors=(0.0, 1.0),
    ),
    SyntheticTemplate(
        name="late_worker_single",
        head_ages=(58, 60, 62, 64, 66, 68),
        spouse_age_offsets=(None,),
        dependent_age_sets=((),),
        ss_source="older_worker",
        payroll_source="prime_worker",
        pension_source="prime_asset",
        dividend_source="prime_asset",
        ss_split=(1.0, 0.0),
        payroll_split=(1.0, 0.0),
        ss_scale_factors=(0.25, 0.5, 0.75, 1.0),
        payroll_scale_factors=(0.5, 1.0, 1.5, 2.0),
        pension_scale_factors=(0.0, 1.0),
        dividend_scale_factors=(0.0, 1.0),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype a fully synthetic late-year support using a minimal set of "
            "head/spouse/dependent features."
        )
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="Projection year to target.",
    )
    parser.add_argument(
        "--target-source",
        default=get_long_term_target_source(),
        help="Named long-run target source package.",
    )
    parser.add_argument(
        "--base-dataset",
        default=DEFAULT_DATASET,
        help="Base 2024 dataset used to derive comparison pools.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--epsilon-path",
        default="0.25,0.5,1.0",
        help=(
            "Comma-separated approximate error thresholds to probe with "
            "bounded entropy after the exact solve. Use an empty string to disable."
        ),
    )
    parser.add_argument(
        "--donor-probe-top-n",
        type=int,
        default=20,
        help="Number of top exact-fit synthetic candidates to probe against real donors.",
    )
    parser.add_argument(
        "--donor-probe-k",
        type=int,
        default=5,
        help="Number of nearest real donors to report for each probed synthetic candidate.",
    )
    return parser.parse_args()


def parse_epsilon_path(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _period_column(name: str, base_year: int) -> str:
    return f"{name}__{base_year}"


def classify_archetype(
    *,
    head_age: float,
    spouse_age: float | None,
    dependent_count: int,
    ss_total: float,
    payroll_total: float,
) -> str:
    older_head = head_age >= 65
    has_spouse = spouse_age is not None
    older_spouse = spouse_age is not None and spouse_age >= 65
    positive_ss = ss_total > 0
    positive_payroll = payroll_total > 0

    if older_head:
        if dependent_count > 0 and positive_payroll:
            return "older_plus_prime_worker_family"
        if has_spouse and spouse_age is not None and spouse_age < 65 and positive_payroll:
            return "mixed_retiree_worker_couple"
        if has_spouse and older_spouse:
            if positive_payroll and positive_ss:
                return "older_worker_couple"
            if positive_ss:
                return "older_beneficiary_couple"
            return "older_couple_other"
        if positive_payroll and positive_ss:
            return "older_worker_single"
        if positive_ss:
            return "older_beneficiary_single"
        return "older_single_other"

    if dependent_count > 0:
        return "prime_worker_family" if positive_payroll else "prime_other_family"
    if has_spouse:
        return "prime_worker_couple" if positive_payroll else "prime_other_couple"
    return "prime_worker_single" if positive_payroll else "prime_other_single"


def build_tax_unit_summary(
    dataset: str | Dataset,
    *,
    period: int,
) -> pd.DataFrame:
    sim = Microsimulation(dataset=dataset)
    input_df = sim.to_input_dataframe()

    person_df = pd.DataFrame(
        {
            "tax_unit_id": sim.calculate("person_tax_unit_id", period=period).values,
            "household_id": sim.calculate("person_household_id", period=period).values,
            "age": sim.calculate("age", period=period).values,
            "is_head": sim.calculate("is_tax_unit_head", period=period).values,
            "is_spouse": sim.calculate("is_tax_unit_spouse", period=period).values,
            "is_dependent": sim.calculate(
                "is_tax_unit_dependent", period=period
            ).values,
            "social_security": sim.calculate(
                "social_security", period=period
            ).values,
            "payroll": (
                sim.calculate(
                    "taxable_earnings_for_social_security", period=period
                ).values
                + sim.calculate(
                    "social_security_taxable_self_employment_income", period=period
                ).values
            ),
            "dividend_income": sim.calculate(
                "qualified_dividend_income", period=period
            ).values,
            "pension_income": sim.calculate(
                "taxable_pension_income", period=period
            ).values,
            "person_weight": input_df[f"person_weight__{period}"].astype(float).values,
            "household_weight": input_df[
                f"household_weight__{period}"
            ].astype(float).values,
        }
    )

    rows: list[dict[str, object]] = []
    for tax_unit_id, group in person_df.groupby("tax_unit_id", sort=False):
        heads = group[group["is_head"]]
        spouses = group[group["is_spouse"]]
        dependents = group[group["is_dependent"]]
        adults = group[group["age"] >= 18]
        head_age = float(
            heads["age"].iloc[0]
            if not heads.empty
            else adults["age"].max() if not adults.empty else group["age"].max()
        )
        spouse_age = (
            float(spouses["age"].iloc[0]) if not spouses.empty else None
        )
        dependent_count = int(len(dependents))
        adult_count = int((group["age"] >= 18).sum())
        dependent_ages = tuple(sorted(int(age) for age in dependents["age"].tolist()))
        head_payroll = float(heads["payroll"].sum()) if not heads.empty else 0.0
        spouse_payroll = float(spouses["payroll"].sum()) if not spouses.empty else 0.0
        head_ss = float(heads["social_security"].sum()) if not heads.empty else 0.0
        spouse_ss = (
            float(spouses["social_security"].sum()) if not spouses.empty else 0.0
        )
        row = {
            "tax_unit_id": int(tax_unit_id),
            "household_id": int(group["household_id"].iloc[0]),
            "head_age": head_age,
            "spouse_age": spouse_age,
            "adult_count": adult_count,
            "dependent_count": dependent_count,
            "dependent_ages": dependent_ages,
            "head_payroll": head_payroll,
            "spouse_payroll": spouse_payroll,
            "head_ss": head_ss,
            "spouse_ss": spouse_ss,
            "payroll_total": float(group["payroll"].sum()),
            "ss_total": float(group["social_security"].sum()),
            "dividend_income": float(group["dividend_income"].sum()),
            "pension_income": float(group["pension_income"].sum()),
            "support_count_weight": 1.0,
            "person_weight_proxy": float(group["person_weight"].max()),
            "household_weight_proxy": float(group["household_weight"].max()),
        }
        row["archetype"] = classify_archetype(
            head_age=row["head_age"],
            spouse_age=row["spouse_age"],
            dependent_count=row["dependent_count"],
            ss_total=row["ss_total"],
            payroll_total=row["payroll_total"],
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_actual_tax_unit_summary(base_dataset: str) -> pd.DataFrame:
    return build_tax_unit_summary(base_dataset, period=BASE_YEAR)


def attach_person_uprating_factors(
    input_df: pd.DataFrame,
    sim: Microsimulation,
    *,
    base_year: int,
    target_year: int,
) -> pd.DataFrame:
    df = input_df.copy()
    payroll_columns = [
        _period_column(component, base_year)
        for component in PAYROLL_COMPONENTS
        if _period_column(component, base_year) in df.columns
    ]
    ss_columns = [
        _period_column(component, base_year)
        for component in SS_COMPONENTS
        if _period_column(component, base_year) in df.columns
    ]
    base_payroll = (
        df[payroll_columns].astype(float).sum(axis=1).to_numpy()
        if payroll_columns
        else np.zeros(len(df), dtype=float)
    )
    base_ss = (
        df[ss_columns].astype(float).sum(axis=1).to_numpy()
        if ss_columns
        else np.zeros(len(df), dtype=float)
    )
    uprated_payroll = sum(
        sim.calculate(component, period=target_year).values.astype(float)
        for component in PAYROLL_COMPONENTS
    )
    uprated_ss = sum(
        sim.calculate(component, period=target_year).values.astype(float)
        for component in SS_COMPONENTS
    )
    df[PAYROLL_UPRATING_FACTOR_COLUMN] = np.where(
        base_payroll > 0,
        uprated_payroll / np.maximum(base_payroll, 1e-12),
        np.nan,
    )
    df[SS_UPRATING_FACTOR_COLUMN] = np.where(
        base_ss > 0,
        uprated_ss / np.maximum(base_ss, 1e-12),
        np.nan,
    )
    return df


def load_base_aggregates(base_dataset: str) -> dict[str, float]:
    sim = Microsimulation(dataset=base_dataset)
    household_series = sim.calculate("household_id", period=BASE_YEAR, map_to="household")
    weights = household_series.weights.values.astype(float)
    ss = sim.calculate("social_security", period=BASE_YEAR, map_to="household").values
    payroll = (
        sim.calculate(
            "taxable_earnings_for_social_security",
            period=BASE_YEAR,
            map_to="household",
        ).values
        + sim.calculate(
            "social_security_taxable_self_employment_income",
            period=BASE_YEAR,
            map_to="household",
        ).values
    )
    return {
        "weighted_ss_total": float(np.sum(ss * weights)),
        "weighted_payroll_total": float(np.sum(payroll * weights)),
    }


def quantile_levels(
    values: pd.Series,
    *,
    quantiles: tuple[float, ...],
    include_zero: bool = False,
    positive_only: bool = False,
) -> list[float]:
    series = values.astype(float)
    if positive_only:
        series = series[series > 0]
    if series.empty:
        levels = [0.0]
    else:
        levels = [float(series.quantile(q)) for q in quantiles]
    if include_zero:
        levels = [0.0, *levels]
    deduped = []
    for value in levels:
        rounded = round(value, 2)
        if rounded not in deduped:
            deduped.append(rounded)
    return [float(value) for value in deduped]


def _scale_levels(levels: list[float], scale: float) -> list[float]:
    return [round(level * scale, 2) for level in levels]


@lru_cache(maxsize=None)
def load_policyengine_social_security_cap(year: int) -> float:
    sim = Microsimulation(dataset=DEFAULT_DATASET)
    return validate_projected_social_security_cap(
        sim.tax_benefit_system.parameters,
        year,
    )


def allocate_taxable_payroll_wages(
    total_taxable_payroll: float,
    payroll_split: tuple[float, float],
    payroll_cap: float,
    *,
    has_spouse: bool,
) -> tuple[float, float]:
    total_taxable_payroll = max(float(total_taxable_payroll), 0.0)
    positive_earner_count = int(payroll_split[0] > 0) + int(
        has_spouse and payroll_split[1] > 0
    )
    if not has_spouse or positive_earner_count <= 1:
        if has_spouse and payroll_split[1] > payroll_split[0]:
            return 0.0, min(total_taxable_payroll, payroll_cap)
        return min(total_taxable_payroll, payroll_cap), 0.0

    total_taxable_payroll = min(total_taxable_payroll, positive_earner_count * payroll_cap)
    preferred_head = total_taxable_payroll * float(payroll_split[0])
    lower = max(0.0, total_taxable_payroll - payroll_cap)
    upper = min(payroll_cap, total_taxable_payroll)
    head_taxable = min(max(preferred_head, lower), upper)
    spouse_taxable = total_taxable_payroll - head_taxable
    return float(head_taxable), float(spouse_taxable)


def build_quantile_pools(
    actual_summary: pd.DataFrame,
    *,
    ss_scale: float,
    earnings_scale: float,
) -> dict[str, dict[str, list[float]]]:
    masks = {
        "older_beneficiary": (actual_summary["head_age"] >= 65)
        & (actual_summary["ss_total"] > 0),
        "older_couple_beneficiary": (actual_summary["head_age"] >= 65)
        & actual_summary["spouse_age"].fillna(-1).ge(65)
        & (actual_summary["ss_total"] > 0),
        "older_worker": (actual_summary["head_age"] >= 65)
        & (actual_summary["ss_total"] > 0)
        & (actual_summary["payroll_total"] > 0),
        "prime_worker": (actual_summary["head_age"] < 65)
        & (actual_summary["payroll_total"] > 0),
        "prime_worker_family": (actual_summary["head_age"] < 65)
        & (actual_summary["dependent_count"] > 0)
        & (actual_summary["payroll_total"] > 0),
        "older_asset": (actual_summary["head_age"] >= 65)
        & (
            (actual_summary["pension_income"] > 0)
            | (actual_summary["dividend_income"] > 0)
        ),
        "prime_asset": (actual_summary["head_age"] < 65)
        & (
            (actual_summary["pension_income"] > 0)
            | (actual_summary["dividend_income"] > 0)
        ),
        "zero": actual_summary["head_age"].notna(),
    }

    pools: dict[str, dict[str, list[float]]] = {}
    for name, mask in masks.items():
        subset = actual_summary[mask]
        pools[name] = {
            "ss": _scale_levels(
                quantile_levels(
                    subset["ss_total"],
                    quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
                    include_zero=(name == "zero"),
                    positive_only=(name != "zero"),
                ),
                ss_scale,
            ),
            "payroll": _scale_levels(
                quantile_levels(
                    subset["payroll_total"],
                    quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
                    include_zero=(
                        name
                        in {
                            "zero",
                            "older_beneficiary",
                            "older_couple_beneficiary",
                            "older_asset",
                        }
                    ),
                    positive_only=(name not in {"zero"}),
                ),
                earnings_scale,
            ),
            "pension": _scale_levels(
                quantile_levels(
                    subset["pension_income"],
                    quantiles=(0.25, 0.5, 0.75, 0.9),
                    include_zero=True,
                    positive_only=True,
                ),
                earnings_scale,
            ),
            "dividend": _scale_levels(
                quantile_levels(
                    subset["dividend_income"],
                    quantiles=(0.25, 0.5, 0.75, 0.9),
                    include_zero=True,
                    positive_only=True,
                ),
                earnings_scale,
            ),
        }
    return pools


def generate_synthetic_candidates(
    pools: dict[str, dict[str, list[float]]],
    *,
    payroll_cap: float,
) -> list[SyntheticCandidate]:
    candidates: list[SyntheticCandidate] = []
    for template in TEMPLATES:
        ss_levels = pools[template.ss_source]["ss"]
        payroll_levels = pools[template.payroll_source]["payroll"]
        pension_levels = pools[template.pension_source]["pension"]
        dividend_levels = pools[template.dividend_source]["dividend"]
        for head_age in template.head_ages:
            for spouse_offset in template.spouse_age_offsets:
                spouse_age = (
                    None if spouse_offset is None else max(18, head_age + spouse_offset)
                )
                for dependent_ages in template.dependent_age_sets:
                    for ss_total in ss_levels:
                        for ss_scale in template.ss_scale_factors:
                            for payroll_total in payroll_levels:
                                for payroll_scale in template.payroll_scale_factors:
                                    for pension_income in pension_levels:
                                        for pension_scale in template.pension_scale_factors:
                                            for dividend_income in dividend_levels:
                                                for dividend_scale in template.dividend_scale_factors:
                                                    scaled_ss_total = ss_total * ss_scale
                                                    scaled_payroll_total = payroll_total * payroll_scale
                                                    scaled_pension_income = pension_income * pension_scale
                                                    scaled_dividend_income = dividend_income * dividend_scale
                                                    head_ss = (
                                                        scaled_ss_total * template.ss_split[0]
                                                    )
                                                    spouse_ss = (
                                                        scaled_ss_total * template.ss_split[1]
                                                    )
                                                    head_wages, spouse_wages = allocate_taxable_payroll_wages(
                                                        scaled_payroll_total,
                                                        template.payroll_split,
                                                        payroll_cap,
                                                        has_spouse=spouse_age is not None,
                                                    )
                                                    candidates.append(
                                                        SyntheticCandidate(
                                                            archetype=template.name,
                                                            head_age=head_age,
                                                            spouse_age=spouse_age,
                                                            dependent_ages=tuple(dependent_ages),
                                                            head_wages=head_wages,
                                                            spouse_wages=spouse_wages,
                                                            head_ss=head_ss,
                                                            spouse_ss=spouse_ss,
                                                            pension_income=scaled_pension_income,
                                                            dividend_income=scaled_dividend_income,
                                                        )
                                                    )
    # Deduplicate exact duplicates caused by repeated quantiles.
    deduped: dict[tuple[object, ...], SyntheticCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.archetype,
            candidate.head_age,
            candidate.spouse_age,
            candidate.dependent_ages,
            round(candidate.head_wages, 2),
            round(candidate.spouse_wages, 2),
            round(candidate.head_ss, 2),
            round(candidate.spouse_ss, 2),
            round(candidate.pension_income, 2),
            round(candidate.dividend_income, 2),
        )
        deduped[key] = candidate
    return list(deduped.values())


def age_bucket_vector(ages: list[int], age_bins: list[tuple[int, int]]) -> np.ndarray:
    vector = np.zeros(len(age_bins), dtype=float)
    for age in ages:
        if age >= 85:
            vector[-1] += 1.0
            continue
        for idx, (start, end) in enumerate(age_bins):
            if start <= age < end:
                vector[idx] += 1.0
                break
    return vector


def build_role_composite_calibration_blueprint(
    augmentation_report: dict[str, object],
    *,
    year: int,
    age_bins: list[tuple[int, int]],
    hh_id_to_idx: dict[int, int],
    baseline_weights: np.ndarray,
    base_weight_scale: float = 0.5,
) -> dict[str, object] | None:
    """
    Build target-year calibration overrides for donor-composite clones.

    Donor-composite augmentation produces realized microdata rows that are close
    to the synthetic target support but not numerically identical. At the
    augmentation target year we can calibrate against the exact clone
    blueprints, using the synthetic solution as priors, while still applying
    the resulting household weights to the realized rows.
    """
    target_year = augmentation_report.get("target_year")
    clone_reports = augmentation_report.get("clone_household_reports")
    if target_year is None or int(target_year) != int(year):
        return None
    if not clone_reports:
        return None

    baseline_weights = np.asarray(baseline_weights, dtype=float)
    if baseline_weights.ndim != 1:
        raise ValueError("baseline_weights must be one-dimensional")

    prior_weights = np.maximum(
        baseline_weights * float(base_weight_scale),
        1e-12,
    )
    clone_total_prior_weight = max(float(baseline_weights.sum()), 1.0)
    age_overrides: dict[int, np.ndarray] = {}
    ss_overrides: dict[int, float] = {}
    payroll_overrides: dict[int, float] = {}
    applied_clone_households = 0

    for clone_report in clone_reports:
        household_id = int(clone_report["clone_household_id"])
        idx = hh_id_to_idx.get(household_id)
        if idx is None:
            continue
        ages = [int(clone_report["target_head_age"])]
        spouse_age = clone_report.get("target_spouse_age")
        if spouse_age is not None:
            ages.append(int(spouse_age))
        ages.extend(int(age) for age in clone_report.get("target_dependent_ages", []))
        age_overrides[idx] = age_bucket_vector(ages, age_bins)
        ss_overrides[idx] = float(clone_report["target_ss_total"])
        payroll_overrides[idx] = float(clone_report["target_payroll_total"])
        prior_weights[idx] = (
            clone_total_prior_weight
            * float(clone_report["per_clone_weight_share_pct"])
            / 100.0
        )
        applied_clone_households += 1

    return {
        "baseline_weights": np.maximum(prior_weights, 1e-12),
        "age_overrides": age_overrides,
        "ss_overrides": ss_overrides,
        "payroll_overrides": payroll_overrides,
        "summary": {
            "mode": "target_year_role_composite_blueprint",
            "target_year": int(target_year),
            "clone_household_count": int(applied_clone_households),
            "base_weight_scale": float(base_weight_scale),
            "clone_total_prior_weight": float(clone_total_prior_weight),
        },
    }


def _ages_from_summary_row(row: pd.Series) -> list[int]:
    ages = [int(round(float(row["head_age"])))]
    if pd.notna(row.get("spouse_age")):
        ages.append(int(round(float(row["spouse_age"]))))
    ages.extend(int(age) for age in row.get("dependent_ages", ()))
    return ages


def _clone_report_record(
    *,
    clone_df: pd.DataFrame,
    base_year: int,
    target_candidate: SyntheticCandidate,
    candidate_idx: int,
    target_weight_share_pct: float,
    clone_weight_scale: float,
    combination_count: int,
    older_donor_row: pd.Series | None,
    worker_donor_row: pd.Series | None,
) -> dict[str, object]:
    household_id_col = _period_column("household_id", base_year)
    tax_unit_id_col = _period_column("tax_unit_id", base_year)
    age_col = _period_column("age", base_year)
    payroll_columns = [
        _period_column(component, base_year)
        for component in PAYROLL_COMPONENTS
        if _period_column(component, base_year) in clone_df.columns
    ]
    ss_columns = [
        _period_column(component, base_year)
        for component in SS_COMPONENTS
        if _period_column(component, base_year) in clone_df.columns
    ]
    ages = sorted(int(round(age)) for age in clone_df[age_col].astype(float).tolist())
    return {
        "candidate_idx": int(candidate_idx),
        "archetype": target_candidate.archetype,
        "clone_household_id": int(clone_df[household_id_col].iloc[0]),
        "clone_tax_unit_id": int(clone_df[tax_unit_id_col].iloc[0]),
        "clone_person_count": int(len(clone_df)),
        "clone_ages": ages,
        "base_clone_payroll_total": float(clone_df[payroll_columns].sum().sum())
        if payroll_columns
        else 0.0,
        "base_clone_ss_total": float(clone_df[ss_columns].sum().sum())
        if ss_columns
        else 0.0,
        "target_weight_share_pct": float(target_weight_share_pct),
        "per_clone_weight_share_pct": float(
            target_weight_share_pct / max(combination_count, 1)
        ),
        "clone_weight_scale": float(clone_weight_scale),
        "target_head_age": int(target_candidate.head_age),
        "target_spouse_age": (
            int(target_candidate.spouse_age)
            if target_candidate.spouse_age is not None
            else None
        ),
        "target_dependent_ages": list(target_candidate.dependent_ages),
        "target_head_wages": float(target_candidate.head_wages),
        "target_spouse_wages": float(target_candidate.spouse_wages),
        "target_head_ss": float(target_candidate.head_ss),
        "target_spouse_ss": float(target_candidate.spouse_ss),
        "target_payroll_total": float(target_candidate.payroll_total),
        "target_ss_total": float(target_candidate.ss_total),
        "older_donor_tax_unit_id": (
            int(older_donor_row["tax_unit_id"]) if older_donor_row is not None else None
        ),
        "worker_donor_tax_unit_id": (
            int(worker_donor_row["tax_unit_id"]) if worker_donor_row is not None else None
        ),
        "older_donor_distance": (
            float(older_donor_row["distance"])
            if older_donor_row is not None and "distance" in older_donor_row
            else None
        ),
        "worker_donor_distance": (
            float(worker_donor_row["distance"])
            if worker_donor_row is not None and "distance" in worker_donor_row
            else None
        ),
    }


def summarize_realized_clone_translation(
    dataset: str | Dataset,
    *,
    period: int,
    augmentation_report: dict[str, object],
    age_bucket_size: int = 5,
) -> dict[str, object]:
    clone_reports = augmentation_report.get("clone_household_reports", [])
    if not clone_reports:
        return {
            "clone_household_count": 0,
            "matched_clone_household_count": 0,
            "unmatched_clone_household_count": 0,
            "per_clone": [],
            "by_archetype": [],
        }

    realized_summary = build_tax_unit_summary(dataset, period=period)
    realized_by_tax_unit = realized_summary.set_index("tax_unit_id", drop=False)
    age_bins = build_age_bins(85, bucket_size=age_bucket_size)
    per_clone: list[dict[str, object]] = []

    for clone_report in clone_reports:
        target_ages = [int(clone_report["target_head_age"])]
        if clone_report.get("target_spouse_age") is not None:
            target_ages.append(int(clone_report["target_spouse_age"]))
        target_ages.extend(int(age) for age in clone_report["target_dependent_ages"])
        target_vector = age_bucket_vector(target_ages, age_bins)

        clone_tax_unit_id = int(clone_report["clone_tax_unit_id"])
        if clone_tax_unit_id not in realized_by_tax_unit.index:
            per_clone.append(
                {
                    **clone_report,
                    "matched": False,
                    "realized_archetype": None,
                    "realized_ages": None,
                    "realized_ss_total": None,
                    "realized_payroll_total": None,
                    "age_bucket_l1": None,
                    "ss_pct_error": None,
                    "payroll_pct_error": None,
                }
            )
            continue

        realized_row = realized_by_tax_unit.loc[clone_tax_unit_id]
        realized_ages = _ages_from_summary_row(realized_row)
        realized_vector = age_bucket_vector(realized_ages, age_bins)
        realized_ss_total = float(realized_row["ss_total"])
        realized_payroll_total = float(realized_row["payroll_total"])
        target_ss_total = float(clone_report["target_ss_total"])
        target_payroll_total = float(clone_report["target_payroll_total"])
        per_clone.append(
            {
                **clone_report,
                "matched": True,
                "realized_archetype": realized_row["archetype"],
                "realized_ages": realized_ages,
                "realized_ss_total": realized_ss_total,
                "realized_payroll_total": realized_payroll_total,
                "realized_household_weight": float(realized_row["household_weight_proxy"]),
                "age_bucket_l1": float(np.abs(realized_vector - target_vector).sum()),
                "ss_pct_error": (
                    0.0
                    if abs(target_ss_total) < 1e-9
                    else (realized_ss_total - target_ss_total) / target_ss_total * 100
                ),
                "payroll_pct_error": (
                    0.0
                    if abs(target_payroll_total) < 1e-9
                    else (realized_payroll_total - target_payroll_total)
                    / target_payroll_total
                    * 100
                ),
            }
        )

    per_clone_df = pd.DataFrame(per_clone)
    matched_df = per_clone_df[per_clone_df["matched"]].copy()
    if matched_df.empty:
        return {
            "clone_household_count": int(len(per_clone_df)),
            "matched_clone_household_count": 0,
            "unmatched_clone_household_count": int(len(per_clone_df)),
            "per_clone": per_clone,
            "by_archetype": [],
        }

    by_archetype = (
        matched_df.groupby("archetype", sort=False)
        .agg(
            clone_household_count=("clone_tax_unit_id", "count"),
            avg_age_bucket_l1=("age_bucket_l1", "mean"),
            avg_ss_pct_error=("ss_pct_error", "mean"),
            avg_payroll_pct_error=("payroll_pct_error", "mean"),
        )
        .reset_index()
    )
    target_ss_total = float(matched_df["target_ss_total"].sum())
    target_payroll_total = float(matched_df["target_payroll_total"].sum())
    realized_ss_total = float(matched_df["realized_ss_total"].sum())
    realized_payroll_total = float(matched_df["realized_payroll_total"].sum())
    return {
        "clone_household_count": int(len(per_clone_df)),
        "matched_clone_household_count": int(len(matched_df)),
        "unmatched_clone_household_count": int(len(per_clone_df) - len(matched_df)),
        "target_ss_total": target_ss_total,
        "realized_ss_total": realized_ss_total,
        "aggregate_ss_pct_error": (
            0.0
            if abs(target_ss_total) < 1e-9
            else (realized_ss_total - target_ss_total) / target_ss_total * 100
        ),
        "target_payroll_total": target_payroll_total,
        "realized_payroll_total": realized_payroll_total,
        "aggregate_payroll_pct_error": (
            0.0
            if abs(target_payroll_total) < 1e-9
            else (realized_payroll_total - target_payroll_total)
            / target_payroll_total
            * 100
        ),
        "median_age_bucket_l1": float(matched_df["age_bucket_l1"].median()),
        "median_ss_pct_error": float(matched_df["ss_pct_error"].median()),
        "median_payroll_pct_error": float(matched_df["payroll_pct_error"].median()),
        "top_ss_over": matched_df.sort_values("ss_pct_error", ascending=False)
        .head(10)
        .to_dict("records"),
        "top_payroll_over": matched_df.sort_values("payroll_pct_error", ascending=False)
        .head(10)
        .to_dict("records"),
        "by_archetype": by_archetype.to_dict("records"),
        "per_clone": per_clone,
    }


def build_synthetic_constraint_problem(
    candidates: list[SyntheticCandidate],
    *,
    year: int,
    baseline_weights: np.ndarray | None = None,
) -> dict[str, object]:
    payroll_cap = load_policyengine_social_security_cap(year)
    age_targets = load_ssa_age_projections(start_year=year, end_year=year)
    age_bins = build_age_bins(n_ages=age_targets.shape[0], bucket_size=5)
    aggregated_age_targets = aggregate_age_targets(age_targets, age_bins)[:, 0]
    X = np.vstack([age_bucket_vector(candidate.ages(), age_bins) for candidate in candidates])
    ss_values = np.array([candidate.ss_total for candidate in candidates], dtype=float)
    payroll_values = np.array(
        [candidate.taxable_payroll_total(payroll_cap) for candidate in candidates],
        dtype=float,
    )
    if baseline_weights is None:
        baseline_weights = np.ones(len(candidates), dtype=float)
    else:
        baseline_weights = np.asarray(baseline_weights, dtype=float)
        if len(baseline_weights) != len(candidates):
            raise ValueError("baseline_weights must align with candidates")
    return {
        "age_bins": age_bins,
        "aggregated_age_targets": aggregated_age_targets,
        "X": X,
        "ss_values": ss_values,
        "ss_target": float(load_ssa_benefit_projections(year)),
        "payroll_values": payroll_values,
        "payroll_target": float(load_taxable_payroll_projections(year)),
        "payroll_cap": float(payroll_cap),
        "baseline_weights": baseline_weights,
    }


def build_constraint_matrix(problem: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    constraint_matrix = np.column_stack(
        [
            problem["X"],
            problem["ss_values"],
            problem["payroll_values"],
        ]
    )
    targets = np.concatenate(
        [
            problem["aggregated_age_targets"],
            np.array([problem["ss_target"], problem["payroll_target"]], dtype=float),
        ]
    )
    return constraint_matrix, targets


def build_scaled_actual_summary(
    actual_summary: pd.DataFrame,
    *,
    ss_scale: float,
    earnings_scale: float,
) -> pd.DataFrame:
    scaled = actual_summary.copy()
    scaled["scaled_head_payroll"] = scaled["head_payroll"] * earnings_scale
    scaled["scaled_spouse_payroll"] = scaled["spouse_payroll"] * earnings_scale
    scaled["scaled_payroll_total"] = scaled["payroll_total"] * earnings_scale
    scaled["scaled_head_ss"] = scaled["head_ss"] * ss_scale
    scaled["scaled_spouse_ss"] = scaled["spouse_ss"] * ss_scale
    scaled["scaled_ss_total"] = scaled["ss_total"] * ss_scale
    scaled["spouse_present"] = scaled["spouse_age"].notna()
    scaled["spouse_age_filled"] = scaled["spouse_age"].fillna(-1)
    return scaled


def _target_head_payroll_share(candidate: SyntheticCandidate) -> float:
    return _safe_split(
        candidate.head_wages,
        candidate.payroll_total,
        1.0 if candidate.spouse_age is None else 0.5,
    )


def _target_head_ss_share(candidate: SyntheticCandidate) -> float:
    return _safe_split(
        candidate.head_ss,
        candidate.ss_total,
        1.0 if candidate.spouse_age is None else 0.5,
    )


def _target_worker_age(candidate: SyntheticCandidate) -> float:
    if candidate.spouse_age is not None and candidate.spouse_wages > candidate.head_wages:
        return float(candidate.spouse_age)
    return float(candidate.head_age)


def match_older_role_donors(
    target_candidate: SyntheticCandidate,
    scaled_actual_summary: pd.DataFrame,
    *,
    donors_per_target: int,
) -> pd.DataFrame:
    subset = scaled_actual_summary[
        (scaled_actual_summary["ss_total"] > 0)
        & (scaled_actual_summary["head_age"] >= 55)
    ].copy()
    if subset.empty:
        return subset
    target_spouse_age = (
        -1
        if target_candidate.spouse_age is None or target_candidate.spouse_age < 65
        else target_candidate.spouse_age
    )
    target_head_ss_share = _target_head_ss_share(target_candidate)
    donor_head_ss_share = _safe_series_split(
        subset["scaled_head_ss"],
        subset["scaled_ss_total"],
        target_head_ss_share,
    )
    subset["distance"] = (
        (subset["head_age"] - target_candidate.head_age).abs() / 5.0
        + (subset["spouse_age_filled"] - target_spouse_age).abs() / 7.5
        + np.abs(
            np.log1p(subset["scaled_ss_total"]) - np.log1p(target_candidate.ss_total)
        )
        + 0.5
        * np.abs(donor_head_ss_share - target_head_ss_share)
        + 0.15
        * subset["archetype"]
        .isin(
            {
                "older_beneficiary_single",
                "older_beneficiary_couple",
                "older_worker_single",
                "older_worker_couple",
                "mixed_retiree_worker_couple",
                "older_plus_prime_worker_family",
            }
        )
        .rsub(1)
        .astype(float)
    )
    return subset.nsmallest(donors_per_target, "distance").copy()


def match_worker_role_donors(
    target_candidate: SyntheticCandidate,
    scaled_actual_summary: pd.DataFrame,
    *,
    donors_per_target: int,
) -> pd.DataFrame:
    subset = scaled_actual_summary[
        scaled_actual_summary["payroll_total"] > 0
    ].copy()
    target_dependent_count = len(target_candidate.dependent_ages)
    target_spouse_present = target_candidate.spouse_age is not None
    if target_dependent_count > 0:
        family_subset = subset[subset["dependent_count"] == target_dependent_count]
        if not family_subset.empty:
            subset = family_subset.copy()
    if target_spouse_present:
        spouse_subset = subset[subset["spouse_present"]]
        if not spouse_subset.empty:
            subset = spouse_subset.copy()
    target_worker_age = _target_worker_age(target_candidate)
    target_head_payroll_share = _target_head_payroll_share(target_candidate)
    donor_head_payroll_share = _safe_series_split(
        subset["scaled_head_payroll"],
        subset["scaled_payroll_total"],
        target_head_payroll_share,
    )
    subset["distance"] = (
        (subset["head_age"] - target_worker_age).abs() / 5.0
        + (subset["dependent_count"] - target_dependent_count).abs() * 0.75
        + np.abs(
            np.log1p(subset["scaled_payroll_total"])
            - np.log1p(target_candidate.payroll_total)
        )
        + 0.5
        * np.abs(donor_head_payroll_share - target_head_payroll_share)
        + 0.25 * subset["spouse_present"].ne(target_spouse_present).astype(float)
    )
    return subset.nsmallest(donors_per_target, "distance").copy()


def build_role_donor_composite_candidate(
    target_candidate: SyntheticCandidate,
    *,
    older_donor_row: pd.Series | None,
    worker_donor_row: pd.Series | None,
    earnings_scale: float,
) -> SyntheticCandidate:
    target_head_payroll_share = _target_head_payroll_share(target_candidate)
    target_head_ss_share = _target_head_ss_share(target_candidate)
    if (
        worker_donor_row is not None
        and target_candidate.head_wages > 0
        and target_candidate.spouse_wages > 0
    ):
        head_payroll_share = _safe_split(
            float(worker_donor_row["scaled_head_payroll"]),
            float(worker_donor_row["scaled_payroll_total"]),
            target_head_payroll_share,
        )
    else:
        head_payroll_share = target_head_payroll_share
    if (
        older_donor_row is not None
        and target_candidate.head_ss > 0
        and target_candidate.spouse_ss > 0
    ):
        head_ss_share = _safe_split(
            float(older_donor_row["scaled_head_ss"]),
            float(older_donor_row["scaled_ss_total"]),
            target_head_ss_share,
        )
    else:
        head_ss_share = target_head_ss_share

    payroll_total = target_candidate.payroll_total
    ss_total = target_candidate.ss_total
    pension_income = 0.0
    dividend_income = 0.0
    if older_donor_row is not None:
        pension_income += float(older_donor_row["pension_income"]) * earnings_scale
        dividend_income += float(older_donor_row["dividend_income"]) * earnings_scale
    if worker_donor_row is not None and target_candidate.ss_total <= 0:
        pension_income += float(worker_donor_row["pension_income"]) * earnings_scale
        dividend_income += float(worker_donor_row["dividend_income"]) * earnings_scale

    return SyntheticCandidate(
        archetype=f"{target_candidate.archetype}_role_donor",
        head_age=target_candidate.head_age,
        spouse_age=target_candidate.spouse_age,
        dependent_ages=target_candidate.dependent_ages,
        head_wages=payroll_total * head_payroll_share,
        spouse_wages=payroll_total * (1.0 - head_payroll_share),
        head_ss=ss_total * head_ss_share,
        spouse_ss=ss_total * (1.0 - head_ss_share),
        pension_income=pension_income,
        dividend_income=dividend_income,
    )


def build_role_donor_composites(
    candidates: list[SyntheticCandidate],
    weights: np.ndarray,
    actual_summary: pd.DataFrame,
    *,
    ss_scale: float,
    earnings_scale: float,
    top_n_targets: int,
    older_donors_per_target: int,
    worker_donors_per_target: int,
    max_older_distance: float = 3.0,
    max_worker_distance: float = 3.0,
) -> tuple[list[SyntheticCandidate], np.ndarray, dict[str, object]]:
    exact_df = summarize_exact_candidates(candidates, weights)
    target_df = exact_df[exact_df["synthetic_weight"] > 0].head(top_n_targets).copy()
    scaled_actual = build_scaled_actual_summary(
        actual_summary,
        ss_scale=ss_scale,
        earnings_scale=earnings_scale,
    )

    composite_candidates: list[SyntheticCandidate] = []
    composite_weights: list[float] = []
    composite_records: list[dict[str, object]] = []
    skipped_targets: list[dict[str, object]] = []

    total_weight = max(float(weights.sum()), 1.0)

    for _, target_row in target_df.iterrows():
        target_candidate = candidates[int(target_row["candidate_idx"])]
        older_donors = [None]
        if target_candidate.ss_total > 0:
            matched = match_older_role_donors(
                target_candidate,
                scaled_actual,
                donors_per_target=older_donors_per_target,
            )
            usable = matched[matched["distance"] <= max_older_distance].copy()
            if usable.empty:
                skipped_targets.append(
                    {
                        "candidate_idx": int(target_row["candidate_idx"]),
                        "archetype": target_candidate.archetype,
                        "reason": "no_older_donor",
                        "best_distance": float(matched["distance"].min())
                        if not matched.empty
                        else None,
                    }
                )
                continue
            older_donors = [row for _, row in usable.iterrows()]

        worker_donors = [None]
        if target_candidate.payroll_total > 0:
            matched = match_worker_role_donors(
                target_candidate,
                scaled_actual,
                donors_per_target=worker_donors_per_target,
            )
            usable = matched[matched["distance"] <= max_worker_distance].copy()
            if usable.empty:
                skipped_targets.append(
                    {
                        "candidate_idx": int(target_row["candidate_idx"]),
                        "archetype": target_candidate.archetype,
                        "reason": "no_worker_donor",
                        "best_distance": float(matched["distance"].min())
                        if not matched.empty
                        else None,
                    }
                )
                continue
            worker_donors = [row for _, row in usable.iterrows()]

        target_weight = float(target_row["synthetic_weight"])
        combination_count = max(len(older_donors) * len(worker_donors), 1)
        per_candidate_weight = target_weight / combination_count
        for older_donor in older_donors:
            for worker_donor in worker_donors:
                composite_candidates.append(
                    build_role_donor_composite_candidate(
                        target_candidate,
                        older_donor_row=older_donor,
                        worker_donor_row=worker_donor,
                        earnings_scale=earnings_scale,
                    )
                )
                composite_weights.append(per_candidate_weight)
                composite_records.append(
                    {
                        "composite_idx": int(len(composite_candidates) - 1),
                        "candidate_idx": int(target_row["candidate_idx"]),
                        "archetype": target_candidate.archetype,
                        "older_tax_unit_id": (
                            None
                            if older_donor is None
                            else int(older_donor["tax_unit_id"])
                        ),
                        "worker_tax_unit_id": (
                            None
                            if worker_donor is None
                            else int(worker_donor["tax_unit_id"])
                        ),
                        "older_distance": (
                            None
                            if older_donor is None
                            else float(older_donor["distance"])
                        ),
                        "worker_distance": (
                            None
                            if worker_donor is None
                            else float(worker_donor["distance"])
                        ),
                        "assigned_weight_share_pct": float(
                            per_candidate_weight / total_weight * 100
                        ),
                    }
                )

    prior_weights = np.asarray(composite_weights, dtype=float)
    probe_summary = summarize_solution(
        composite_candidates,
        prior_weights,
        actual_summary,
    )
    return (
        composite_candidates,
        prior_weights,
        {
            "top_n_targets": int(top_n_targets),
            "older_donors_per_target": int(older_donors_per_target),
            "worker_donors_per_target": int(worker_donors_per_target),
            "max_older_distance": float(max_older_distance),
            "max_worker_distance": float(max_worker_distance),
            "skipped_targets": skipped_targets,
            "composite_records": composite_records,
            "prior_summary": probe_summary,
        },
    )


def summarize_exact_candidates(
    candidates: list[SyntheticCandidate],
    weights: np.ndarray,
) -> pd.DataFrame:
    candidate_rows = []
    for idx, (candidate, weight) in enumerate(zip(candidates, weights)):
        candidate_rows.append(
            {
                "candidate_idx": idx,
                **asdict(candidate),
                "dependent_count": len(candidate.dependent_ages),
                "payroll_total": candidate.payroll_total,
                "ss_total": candidate.ss_total,
                "synthetic_weight": float(weight),
            }
        )
    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        "synthetic_weight",
        ascending=False,
    )
    total_weight = max(float(candidate_df["synthetic_weight"].sum()), 1.0)
    candidate_df["weight_share_pct"] = (
        candidate_df["synthetic_weight"] / total_weight * 100
    )
    return candidate_df


def match_real_donors_for_target(
    target_row: pd.Series,
    scaled_actual_summary: pd.DataFrame,
    *,
    donors_per_target: int,
) -> pd.DataFrame:
    target_spouse_present = pd.notna(target_row["spouse_age"])
    target_spouse_age = -1 if not target_spouse_present else target_row["spouse_age"]
    target_adult_count = 2 if target_spouse_present else 1
    subset = scaled_actual_summary[
        scaled_actual_summary["spouse_present"].eq(target_spouse_present)
        & scaled_actual_summary["adult_count"].eq(target_adult_count)
        & scaled_actual_summary["dependent_count"].eq(int(target_row["dependent_count"]))
    ].copy()
    if subset.empty:
        subset = scaled_actual_summary[
            scaled_actual_summary["adult_count"].ge(target_adult_count)
        ].copy()
    if subset.empty:
        subset = scaled_actual_summary.copy()
    subset["distance"] = (
        (subset["head_age"] - target_row["head_age"]).abs() / 5.0
        + (
            subset["spouse_age_filled"] - target_spouse_age
        ).abs()
        / 5.0
        + (subset["dependent_count"] - target_row["dependent_count"]).abs() * 0.75
        + np.abs(
            np.log1p(subset["scaled_payroll_total"])
            - np.log1p(float(target_row["payroll_total"]))
        )
        + np.abs(
            np.log1p(subset["scaled_ss_total"])
            - np.log1p(float(target_row["ss_total"]))
        )
        + 0.25 * subset["archetype"].ne(target_row["archetype"]).astype(float)
    )
    nearest = subset.nsmallest(donors_per_target, "distance").copy()
    nearest["target_candidate_idx"] = int(target_row["candidate_idx"])
    nearest["target_archetype"] = target_row["archetype"]
    nearest["target_weight_share_pct"] = float(target_row["weight_share_pct"])
    nearest["target_head_age"] = int(target_row["head_age"])
    nearest["target_spouse_age"] = (
        None if not target_spouse_present else int(target_row["spouse_age"])
    )
    nearest["target_dependent_count"] = int(target_row["dependent_count"])
    nearest["target_payroll_total"] = float(target_row["payroll_total"])
    nearest["target_ss_total"] = float(target_row["ss_total"])
    nearest["required_head_age_shift"] = (
        nearest["target_head_age"] - nearest["head_age"]
    )
    nearest["required_spouse_age_shift"] = np.where(
        nearest["target_spouse_age"].isna(),
        np.nan,
        nearest["target_spouse_age"] - nearest["spouse_age_filled"],
    )
    return nearest


def summarize_donor_probe(
    candidates: list[SyntheticCandidate],
    weights: np.ndarray,
    actual_summary: pd.DataFrame,
    *,
    ss_scale: float,
    earnings_scale: float,
    top_n_targets: int,
    donors_per_target: int,
) -> dict[str, object]:
    exact_df = summarize_exact_candidates(candidates, weights)
    target_df = exact_df[exact_df["synthetic_weight"] > 0].head(top_n_targets).copy()
    scaled_actual = build_scaled_actual_summary(
        actual_summary,
        ss_scale=ss_scale,
        earnings_scale=earnings_scale,
    )
    donor_matches = []
    for _, target_row in target_df.iterrows():
        donor_matches.append(
            match_real_donors_for_target(
                target_row,
                scaled_actual,
                donors_per_target=donors_per_target,
            )
        )
    donor_df = pd.concat(donor_matches, ignore_index=True)
    nearest_only = donor_df.sort_values(
        ["target_candidate_idx", "distance"],
        ascending=[True, True],
    ).groupby("target_candidate_idx", as_index=False).first()
    distance_summary = {
        "median_best_distance": float(nearest_only["distance"].median()),
        "targets_with_best_distance_le_1": int((nearest_only["distance"] <= 1.0).sum()),
        "targets_with_best_distance_le_2": int((nearest_only["distance"] <= 2.0).sum()),
        "targets_with_best_distance_gt_3": int((nearest_only["distance"] > 3.0).sum()),
    }
    outlier_targets = nearest_only[nearest_only["distance"] > 3.0].copy()
    return {
        "top_n_targets": int(top_n_targets),
        "donors_per_target": int(donors_per_target),
        "distance_summary": distance_summary,
        "nearest_targets": nearest_only[
            [
                "target_candidate_idx",
                "target_archetype",
                "target_weight_share_pct",
                "target_head_age",
                "target_spouse_age",
                "target_dependent_count",
                "target_payroll_total",
                "target_ss_total",
                "tax_unit_id",
                "archetype",
                "head_age",
                "spouse_age",
                "dependent_count",
                "scaled_payroll_total",
                "scaled_ss_total",
                "required_head_age_shift",
                "required_spouse_age_shift",
                "distance",
            ]
        ].to_dict("records"),
        "outlier_targets": outlier_targets[
            [
                "target_candidate_idx",
                "target_archetype",
                "target_weight_share_pct",
                "target_head_age",
                "target_spouse_age",
                "target_dependent_count",
                "target_payroll_total",
                "target_ss_total",
                "distance",
            ]
        ].to_dict("records"),
    }


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


def _scale_person_components(
    row: pd.Series,
    columns: tuple[str, ...],
    target_total: float,
) -> pd.Series:
    available = [column for column in columns if column in row.index]
    if not available:
        return row
    target_total = float(target_total)
    if target_total <= 0:
        for column in available:
            row[column] = 0.0
        return row
    current_total = float(sum(float(row[column]) for column in available))
    if current_total > 0:
        scale = target_total / current_total
        for column in available:
            row[column] = float(row[column]) * scale
        return row
    row[available[0]] = target_total
    for column in available[1:]:
        row[column] = 0.0
    return row


def _target_base_total_for_row(
    row: pd.Series,
    *,
    target_total: float,
    factor_column: str,
    fallback_factor: float,
) -> float:
    target_total = float(target_total)
    if target_total <= 0:
        return 0.0
    factor = row.get(factor_column, np.nan)
    if pd.isna(factor) or float(factor) <= 0:
        factor = fallback_factor
    return target_total / max(float(factor), 1e-12)


def _clone_tax_unit_rows_to_target(
    donor_rows: pd.DataFrame,
    *,
    base_year: int,
    target_candidate: SyntheticCandidate,
    ss_scale: float,
    earnings_scale: float,
    id_counters: dict[str, int],
    clone_weight_scale: float,
    clone_weight_divisor: int,
) -> tuple[pd.DataFrame, dict[str, int]] | tuple[None, dict[str, int]]:
    age_col = _period_column("age", base_year)
    household_weight_col = _period_column("household_weight", base_year)
    person_weight_col = _period_column("person_weight", base_year)
    person_id_col = _period_column(PERSON_ID_COLUMN, base_year)

    adults = donor_rows[donor_rows[age_col] >= 18].sort_values(age_col, ascending=False)
    dependents = donor_rows[donor_rows[age_col] < 18].sort_values(age_col, ascending=False)
    target_has_spouse = target_candidate.spouse_age is not None
    target_adult_count = 2 if target_has_spouse else 1
    if len(adults) < target_adult_count or len(dependents) != len(target_candidate.dependent_ages):
        return None, id_counters

    cloned = donor_rows.copy()
    household_id = id_counters["household"]
    id_counters["household"] += 1
    for entity_name, columns in ENTITY_ID_COLUMNS.items():
        entity_id = id_counters[entity_name]
        id_counters[entity_name] += 1
        for raw_column in columns:
            column = _period_column(raw_column, base_year)
            if column in cloned.columns:
                cloned[column] = entity_id if entity_name != "household" else household_id
    cloned[_period_column("household_id", base_year)] = household_id
    cloned[_period_column("person_household_id", base_year)] = household_id

    person_ids = range(id_counters["person"], id_counters["person"] + len(cloned))
    id_counters["person"] += len(cloned)
    cloned[person_id_col] = _cast_mapped_ids(
        cloned[person_id_col],
        pd.Series(list(person_ids), index=cloned.index),
    )

    if household_weight_col in cloned.columns:
        cloned[household_weight_col] = (
            cloned[household_weight_col].astype(float)
            * clone_weight_scale
            / max(clone_weight_divisor, 1)
        )
    if person_weight_col in cloned.columns:
        cloned[person_weight_col] = (
            cloned[person_weight_col].astype(float)
            * clone_weight_scale
            / max(clone_weight_divisor, 1)
        )

    adult_indices = adults.index.tolist()
    head_idx = adult_indices[0]
    spouse_idx = adult_indices[1] if target_has_spouse else None
    dependent_indices = dependents.index.tolist()

    cloned.loc[head_idx, age_col] = float(target_candidate.head_age)
    if spouse_idx is not None:
        cloned.loc[spouse_idx, age_col] = float(target_candidate.spouse_age)
    for dep_idx, dep_age in zip(dependent_indices, target_candidate.dependent_ages):
        cloned.loc[dep_idx, age_col] = float(dep_age)

    payroll_columns = tuple(_period_column(component, base_year) for component in PAYROLL_COMPONENTS)
    ss_columns = tuple(_period_column(component, base_year) for component in SS_COMPONENTS)
    qbi_col = _period_column("w2_wages_from_qualified_business", base_year)

    target_head_payroll = _target_base_total_for_row(
        cloned.loc[head_idx],
        target_total=float(target_candidate.head_wages),
        factor_column=PAYROLL_UPRATING_FACTOR_COLUMN,
        fallback_factor=earnings_scale,
    )
    target_spouse_payroll = _target_base_total_for_row(
        cloned.loc[spouse_idx] if spouse_idx is not None else pd.Series(dtype=float),
        target_total=float(target_candidate.spouse_wages),
        factor_column=PAYROLL_UPRATING_FACTOR_COLUMN,
        fallback_factor=earnings_scale,
    )
    target_head_ss = _target_base_total_for_row(
        cloned.loc[head_idx],
        target_total=float(target_candidate.head_ss),
        factor_column=SS_UPRATING_FACTOR_COLUMN,
        fallback_factor=ss_scale,
    )
    target_spouse_ss = _target_base_total_for_row(
        cloned.loc[spouse_idx] if spouse_idx is not None else pd.Series(dtype=float),
        target_total=float(target_candidate.spouse_ss),
        factor_column=SS_UPRATING_FACTOR_COLUMN,
        fallback_factor=ss_scale,
    )

    cloned.loc[head_idx] = _scale_person_components(
        cloned.loc[head_idx].copy(),
        payroll_columns,
        target_head_payroll,
    )
    cloned.loc[head_idx] = _scale_person_components(
        cloned.loc[head_idx].copy(),
        ss_columns,
        target_head_ss,
    )
    if spouse_idx is not None:
        cloned.loc[spouse_idx] = _scale_person_components(
            cloned.loc[spouse_idx].copy(),
            payroll_columns,
            target_spouse_payroll,
        )
        cloned.loc[spouse_idx] = _scale_person_components(
            cloned.loc[spouse_idx].copy(),
            ss_columns,
            target_spouse_ss,
        )

    for dep_idx in dependent_indices:
        cloned.loc[dep_idx] = _scale_person_components(
            cloned.loc[dep_idx].copy(),
            payroll_columns,
            0.0,
        )
        cloned.loc[dep_idx] = _scale_person_components(
            cloned.loc[dep_idx].copy(),
            ss_columns,
            0.0,
        )
        if qbi_col in cloned.columns:
            cloned.loc[dep_idx, qbi_col] = 0.0

    if qbi_col in cloned.columns and head_idx in cloned.index:
        cloned.loc[head_idx, qbi_col] = 0.0
        if spouse_idx is not None:
            cloned.loc[spouse_idx, qbi_col] = 0.0

    return cloned, id_counters


def _compose_role_donor_rows_to_target(
    older_donor_rows: pd.DataFrame | None,
    worker_donor_rows: pd.DataFrame | None,
    *,
    base_year: int,
    target_candidate: SyntheticCandidate,
    ss_scale: float,
    earnings_scale: float,
    id_counters: dict[str, int],
    clone_weight_scale: float,
    clone_weight_divisor: int,
) -> tuple[pd.DataFrame, dict[str, int]] | tuple[None, dict[str, int]]:
    age_col = _period_column("age", base_year)
    household_weight_col = _period_column("household_weight", base_year)
    person_weight_col = _period_column("person_weight", base_year)
    person_id_col = _period_column(PERSON_ID_COLUMN, base_year)

    def _adult_rows(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame(columns=[] if df is None else df.columns)
        return df[df[age_col] >= 18].sort_values(age_col, ascending=False)

    def _dependent_rows(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame(columns=[] if df is None else df.columns)
        return df[df[age_col] < 18].sort_values(age_col, ascending=False)

    older_adults = _adult_rows(older_donor_rows)
    worker_adults = _adult_rows(worker_donor_rows)
    worker_dependents = _dependent_rows(worker_donor_rows)

    worker_payroll_factor = (
        float(
            np.nanmedian(
                worker_adults[PAYROLL_UPRATING_FACTOR_COLUMN].astype(float).to_numpy()
            )
        )
        if not worker_adults.empty
        and PAYROLL_UPRATING_FACTOR_COLUMN in worker_adults.columns
        else np.nan
    )
    older_ss_factor = (
        float(
            np.nanmedian(
                older_adults[SS_UPRATING_FACTOR_COLUMN].astype(float).to_numpy()
            )
        )
        if not older_adults.empty and SS_UPRATING_FACTOR_COLUMN in older_adults.columns
        else np.nan
    )
    payroll_reference_factor = (
        worker_payroll_factor
        if np.isfinite(worker_payroll_factor) and worker_payroll_factor > 0
        else earnings_scale
    )
    ss_reference_factor = (
        older_ss_factor
        if np.isfinite(older_ss_factor) and older_ss_factor > 0
        else ss_scale
    )

    selected_rows: list[pd.Series] = []
    head_target_older = target_candidate.head_ss > 0 or target_candidate.head_age >= 65
    head_source_rows = older_adults if head_target_older and not older_adults.empty else worker_adults
    if head_source_rows.empty:
        return None, id_counters
    head_row = head_source_rows.iloc[0].copy()
    selected_rows.append(head_row)

    spouse_row = None
    if target_candidate.spouse_age is not None:
        if target_candidate.spouse_age >= 65 and len(older_adults) >= 2:
            spouse_row = older_adults.iloc[1].copy()
        elif not worker_adults.empty:
            worker_candidates = worker_adults.iloc[1:] if worker_adults.index[0] == head_row.name else worker_adults
            if worker_candidates.empty:
                worker_candidates = worker_adults
            spouse_idx = (worker_candidates[age_col] - target_candidate.spouse_age).abs().idxmin()
            spouse_row = worker_candidates.loc[spouse_idx].copy()
        elif len(older_adults) >= 2:
            spouse_row = older_adults.iloc[1].copy()
        if spouse_row is None:
            fallback_spouse_pool = worker_adults if not worker_adults.empty else older_adults
            if fallback_spouse_pool.empty:
                return None, id_counters
            spouse_row = fallback_spouse_pool.iloc[0].copy()
        selected_rows.append(spouse_row)

    if len(target_candidate.dependent_ages) > 0:
        dependent_rows = [row.copy() for _, row in worker_dependents.iterrows()]
        if not dependent_rows:
            fallback_source = None
            if worker_donor_rows is not None and not worker_donor_rows.empty:
                fallback_source = worker_donor_rows.sort_values(age_col, ascending=True).iloc[
                    0
                ].copy()
            elif older_donor_rows is not None and not older_donor_rows.empty:
                fallback_source = older_donor_rows.sort_values(age_col, ascending=True).iloc[
                    0
                ].copy()
            if fallback_source is None:
                return None, id_counters
            dependent_rows = [fallback_source.copy()]
        while len(dependent_rows) < len(target_candidate.dependent_ages):
            dependent_rows.append(dependent_rows[-1].copy())
        selected_rows.extend(dependent_rows[: len(target_candidate.dependent_ages)])

    # Reset duplicate donor indices so later row-specific retargeting only touches
    # the intended clone row.
    cloned = pd.DataFrame(selected_rows).reset_index(drop=True).copy()
    household_id = id_counters["household"]
    id_counters["household"] += 1
    for entity_name, columns in ENTITY_ID_COLUMNS.items():
        entity_id = id_counters[entity_name]
        id_counters[entity_name] += 1
        for raw_column in columns:
            column = _period_column(raw_column, base_year)
            if column in cloned.columns:
                cloned[column] = entity_id if entity_name != "household" else household_id
    cloned[_period_column("household_id", base_year)] = household_id
    cloned[_period_column("person_household_id", base_year)] = household_id

    person_ids = range(id_counters["person"], id_counters["person"] + len(cloned))
    id_counters["person"] += len(cloned)
    cloned[person_id_col] = _cast_mapped_ids(
        cloned[person_id_col],
        pd.Series(list(person_ids), index=cloned.index),
    )

    if household_weight_col in cloned.columns:
        cloned[household_weight_col] = (
            cloned[household_weight_col].astype(float)
            * clone_weight_scale
            / max(clone_weight_divisor, 1)
        )
    if person_weight_col in cloned.columns:
        cloned[person_weight_col] = (
            cloned[person_weight_col].astype(float)
            * clone_weight_scale
            / max(clone_weight_divisor, 1)
        )

    head_idx = cloned.index[0]
    spouse_idx = cloned.index[1] if target_candidate.spouse_age is not None else None
    dependent_indices = (
        cloned.index[2 : 2 + len(target_candidate.dependent_ages)]
        if target_candidate.spouse_age is not None
        else cloned.index[1 : 1 + len(target_candidate.dependent_ages)]
    )

    cloned.loc[head_idx, age_col] = float(target_candidate.head_age)
    if spouse_idx is not None:
        cloned.loc[spouse_idx, age_col] = float(target_candidate.spouse_age)
    for dep_idx, dep_age in zip(dependent_indices, target_candidate.dependent_ages):
        cloned.loc[dep_idx, age_col] = float(dep_age)

    payroll_columns = tuple(_period_column(component, base_year) for component in PAYROLL_COMPONENTS)
    ss_columns = tuple(_period_column(component, base_year) for component in SS_COMPONENTS)
    qbi_col = _period_column("w2_wages_from_qualified_business", base_year)

    target_head_payroll = float(target_candidate.head_wages) / max(
        payroll_reference_factor,
        1e-12,
    )
    target_spouse_payroll = float(target_candidate.spouse_wages) / max(
        payroll_reference_factor,
        1e-12,
    )
    target_head_ss = float(target_candidate.head_ss) / max(ss_reference_factor, 1e-12)
    target_spouse_ss = float(target_candidate.spouse_ss) / max(
        ss_reference_factor,
        1e-12,
    )

    cloned.loc[head_idx] = _scale_person_components(
        cloned.loc[head_idx].copy(),
        payroll_columns,
        target_head_payroll,
    )
    cloned.loc[head_idx] = _scale_person_components(
        cloned.loc[head_idx].copy(),
        ss_columns,
        target_head_ss,
    )
    if spouse_idx is not None:
        cloned.loc[spouse_idx] = _scale_person_components(
            cloned.loc[spouse_idx].copy(),
            payroll_columns,
            target_spouse_payroll,
        )
        cloned.loc[spouse_idx] = _scale_person_components(
            cloned.loc[spouse_idx].copy(),
            ss_columns,
            target_spouse_ss,
        )
    for dep_idx in dependent_indices:
        cloned.loc[dep_idx] = _scale_person_components(
            cloned.loc[dep_idx].copy(),
            payroll_columns,
            0.0,
        )
        cloned.loc[dep_idx] = _scale_person_components(
            cloned.loc[dep_idx].copy(),
            ss_columns,
            0.0,
        )
        if qbi_col in cloned.columns:
            cloned.loc[dep_idx, qbi_col] = 0.0

    if qbi_col in cloned.columns and head_idx in cloned.index:
        cloned.loc[head_idx, qbi_col] = 0.0
        if spouse_idx is not None:
            cloned.loc[spouse_idx, qbi_col] = 0.0

    return cloned, id_counters


def build_donor_backed_augmented_input_dataframe(
    *,
    base_dataset: str,
    base_year: int,
    target_year: int,
    top_n_targets: int = 20,
    donors_per_target: int = 5,
    max_distance_for_clone: float = 3.0,
    clone_weight_scale: float = 0.1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sim = Microsimulation(dataset=base_dataset)
    input_df = attach_person_uprating_factors(
        sim.to_input_dataframe(),
        sim,
        base_year=base_year,
        target_year=target_year,
    )
    actual_summary = build_actual_tax_unit_summary(base_dataset)
    base_aggregates = load_base_aggregates(base_dataset)
    ss_scale = load_ssa_benefit_projections(target_year) / max(
        base_aggregates["weighted_ss_total"],
        1.0,
    )
    earnings_scale = load_taxable_payroll_projections(target_year) / max(
        base_aggregates["weighted_payroll_total"],
        1.0,
    )
    pools = build_quantile_pools(
        actual_summary,
        ss_scale=ss_scale,
        earnings_scale=earnings_scale,
    )
    candidates = generate_synthetic_candidates(
        pools,
        payroll_cap=load_policyengine_social_security_cap(target_year),
    )
    exact_weights, solve_info = solve_synthetic_support(candidates, year=target_year)
    exact_df = summarize_exact_candidates(candidates, exact_weights)
    target_df = exact_df[exact_df["synthetic_weight"] > 0].head(top_n_targets).copy()
    scaled_actual = build_scaled_actual_summary(
        actual_summary,
        ss_scale=ss_scale,
        earnings_scale=earnings_scale,
    )

    tax_unit_id_col = _period_column("person_tax_unit_id", base_year)
    id_counters = {
        entity_name: _next_entity_id(input_df[_period_column(columns[0], base_year)])
        for entity_name, columns in ENTITY_ID_COLUMNS.items()
    }
    id_counters["person"] = _next_entity_id(input_df[_period_column(PERSON_ID_COLUMN, base_year)])

    clone_frames = []
    clone_household_reports = []
    target_reports = []
    skipped_targets = []

    for _, target_row in target_df.iterrows():
        target_candidate = candidates[int(target_row["candidate_idx"])]
        donor_matches = match_real_donors_for_target(
            target_row,
            scaled_actual,
            donors_per_target=donors_per_target,
        )
        usable = donor_matches[donor_matches["distance"] <= max_distance_for_clone].copy()
        if usable.empty:
            skipped_targets.append(
                {
                    "candidate_idx": int(target_row["candidate_idx"]),
                    "archetype": target_candidate.archetype,
                    "weight_share_pct": float(target_row["weight_share_pct"]),
                    "best_distance": float(donor_matches["distance"].min()),
                }
            )
            continue

        successful_clone_count = 0
        for _, donor_row in usable.iterrows():
            donor_rows = input_df[input_df[tax_unit_id_col] == int(donor_row["tax_unit_id"])].copy()
            clone_df, id_counters = _clone_tax_unit_rows_to_target(
                donor_rows,
                base_year=base_year,
                target_candidate=target_candidate,
                ss_scale=ss_scale,
                earnings_scale=earnings_scale,
                id_counters=id_counters,
                clone_weight_scale=clone_weight_scale,
                clone_weight_divisor=len(usable),
            )
            if clone_df is None:
                continue
            clone_frames.append(clone_df)
            successful_clone_count += 1
        target_reports.append(
            {
                "candidate_idx": int(target_row["candidate_idx"]),
                "archetype": target_candidate.archetype,
                "weight_share_pct": float(target_row["weight_share_pct"]),
                "requested_donor_count": int(len(usable)),
                "successful_clone_count": int(successful_clone_count),
            }
        )

    augmented_df = (
        pd.concat([input_df, *clone_frames], ignore_index=True)
        if clone_frames
        else input_df.copy()
    )
    helper_columns = [PAYROLL_UPRATING_FACTOR_COLUMN, SS_UPRATING_FACTOR_COLUMN]
    augmented_df.drop(
        columns=[column for column in helper_columns if column in augmented_df.columns],
        inplace=True,
        errors="ignore",
    )
    report = {
        "base_dataset": base_dataset,
        "base_year": int(base_year),
        "target_year": int(target_year),
        "target_source": get_long_term_target_source(),
        "solve_info": solve_info,
        "top_n_targets": int(top_n_targets),
        "donors_per_target": int(donors_per_target),
        "max_distance_for_clone": float(max_distance_for_clone),
        "clone_weight_scale": float(clone_weight_scale),
        "base_household_count": int(input_df[_period_column("household_id", base_year)].nunique()),
        "augmented_household_count": int(augmented_df[_period_column("household_id", base_year)].nunique()),
        "base_person_count": int(len(input_df)),
        "augmented_person_count": int(len(augmented_df)),
        "target_reports": target_reports,
        "skipped_targets": skipped_targets,
    }
    return augmented_df, report


def build_role_composite_augmented_input_dataframe(
    *,
    base_dataset: str,
    base_year: int,
    target_year: int,
    top_n_targets: int = 20,
    donors_per_target: int = 5,
    max_older_distance: float = 3.0,
    max_worker_distance: float = 3.0,
    clone_weight_scale: float = 0.1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sim = Microsimulation(dataset=base_dataset)
    input_df = attach_person_uprating_factors(
        sim.to_input_dataframe(),
        sim,
        base_year=base_year,
        target_year=target_year,
    )
    actual_summary = build_actual_tax_unit_summary(base_dataset)
    base_aggregates = load_base_aggregates(base_dataset)
    ss_scale = load_ssa_benefit_projections(target_year) / max(
        base_aggregates["weighted_ss_total"],
        1.0,
    )
    earnings_scale = load_taxable_payroll_projections(target_year) / max(
        base_aggregates["weighted_payroll_total"],
        1.0,
    )
    pools = build_quantile_pools(
        actual_summary,
        ss_scale=ss_scale,
        earnings_scale=earnings_scale,
    )
    candidates = generate_synthetic_candidates(
        pools,
        payroll_cap=load_policyengine_social_security_cap(target_year),
    )
    exact_weights, solve_info = solve_synthetic_support(candidates, year=target_year)
    scaled_actual = build_scaled_actual_summary(
        actual_summary,
        ss_scale=ss_scale,
        earnings_scale=earnings_scale,
    )
    (
        role_composite_candidates,
        role_composite_prior,
        role_composite_probe,
    ) = build_role_donor_composites(
        candidates,
        exact_weights,
        actual_summary,
        ss_scale=ss_scale,
        earnings_scale=earnings_scale,
        top_n_targets=top_n_targets,
        older_donors_per_target=donors_per_target,
        worker_donors_per_target=donors_per_target,
        max_older_distance=max_older_distance,
        max_worker_distance=max_worker_distance,
    )
    role_composite_weights, role_composite_solve_info = solve_synthetic_support(
        role_composite_candidates,
        year=target_year,
        baseline_weights=role_composite_prior,
    )
    role_composite_df = summarize_exact_candidates(
        role_composite_candidates,
        role_composite_weights,
    )
    selected_composite_df = role_composite_df[
        role_composite_df["synthetic_weight"] > 0
    ].copy()
    composite_records_by_idx = {
        int(record["composite_idx"]): record
        for record in role_composite_probe["composite_records"]
    }

    tax_unit_id_col = _period_column("person_tax_unit_id", base_year)
    id_counters = {
        entity_name: _next_entity_id(input_df[_period_column(columns[0], base_year)])
        for entity_name, columns in ENTITY_ID_COLUMNS.items()
    }
    id_counters["person"] = _next_entity_id(
        input_df[_period_column(PERSON_ID_COLUMN, base_year)]
    )

    clone_frames = []
    clone_household_reports = []
    target_reports = []
    skipped_targets = []

    for _, target_row in selected_composite_df.iterrows():
        composite_idx = int(target_row["candidate_idx"])
        target_candidate = role_composite_candidates[composite_idx]
        composite_record = composite_records_by_idx.get(composite_idx)
        if composite_record is None:
            skipped_targets.append(
                {
                    "candidate_idx": composite_idx,
                    "archetype": target_candidate.archetype,
                    "weight_share_pct": float(target_row["weight_share_pct"]),
                    "reason": "missing_composite_record",
                }
            )
            continue

        older_tax_unit_id = composite_record.get("older_tax_unit_id")
        worker_tax_unit_id = composite_record.get("worker_tax_unit_id")
        older_row = None
        worker_row = None
        older_rows = None
        worker_rows = None
        if older_tax_unit_id is not None:
            older_row = scaled_actual[
                scaled_actual["tax_unit_id"].eq(int(older_tax_unit_id))
            ].iloc[0]
            older_rows = input_df[
                input_df[tax_unit_id_col] == int(older_tax_unit_id)
            ].copy()
        if worker_tax_unit_id is not None:
            worker_row = scaled_actual[
                scaled_actual["tax_unit_id"].eq(int(worker_tax_unit_id))
            ].iloc[0]
            worker_rows = input_df[
                input_df[tax_unit_id_col] == int(worker_tax_unit_id)
            ].copy()

        clone_df, id_counters = _compose_role_donor_rows_to_target(
            older_rows,
            worker_rows,
            base_year=base_year,
            target_candidate=target_candidate,
            ss_scale=ss_scale,
            earnings_scale=earnings_scale,
            id_counters=id_counters,
            clone_weight_scale=clone_weight_scale,
            clone_weight_divisor=1,
        )
        if clone_df is None:
            skipped_targets.append(
                {
                    "candidate_idx": composite_idx,
                    "archetype": target_candidate.archetype,
                    "weight_share_pct": float(target_row["weight_share_pct"]),
                    "reason": "clone_build_failed",
                }
            )
            continue
        clone_frames.append(clone_df)
        clone_household_reports.append(
            _clone_report_record(
                clone_df=clone_df,
                base_year=base_year,
                target_candidate=target_candidate,
                candidate_idx=composite_idx,
                target_weight_share_pct=float(target_row["weight_share_pct"]),
                clone_weight_scale=clone_weight_scale,
                combination_count=1,
                older_donor_row=older_row,
                worker_donor_row=worker_row,
            )
        )
        target_reports.append(
            {
                "candidate_idx": composite_idx,
                "archetype": target_candidate.archetype,
                "weight_share_pct": float(target_row["weight_share_pct"]),
                "older_match_count": int(older_tax_unit_id is not None),
                "worker_match_count": int(worker_tax_unit_id is not None),
                "successful_clone_count": 1,
            }
        )

    augmented_df = (
        pd.concat([input_df, *clone_frames], ignore_index=True)
        if clone_frames
        else input_df.copy()
    )
    helper_columns = [PAYROLL_UPRATING_FACTOR_COLUMN, SS_UPRATING_FACTOR_COLUMN]
    augmented_df.drop(
        columns=[column for column in helper_columns if column in augmented_df.columns],
        inplace=True,
        errors="ignore",
    )
    report = {
        "base_dataset": base_dataset,
        "base_year": int(base_year),
        "target_year": int(target_year),
        "target_source": get_long_term_target_source(),
        "solve_info": solve_info,
        "role_composite_solve_info": role_composite_solve_info,
        "selection_strategy": "role_composite_positive_support",
        "top_n_targets": int(top_n_targets),
        "donors_per_target": int(donors_per_target),
        "max_older_distance": float(max_older_distance),
        "max_worker_distance": float(max_worker_distance),
        "clone_weight_scale": float(clone_weight_scale),
        "base_household_count": int(
            input_df[_period_column("household_id", base_year)].nunique()
        ),
        "augmented_household_count": int(
            augmented_df[_period_column("household_id", base_year)].nunique()
        ),
        "base_person_count": int(len(input_df)),
        "augmented_person_count": int(len(augmented_df)),
        "role_composite_candidate_count": int(len(role_composite_candidates)),
        "selected_role_composite_count": int(len(selected_composite_df)),
        "clone_household_count": int(len(clone_household_reports)),
        "clone_household_reports": clone_household_reports,
        "target_reports": target_reports,
        "skipped_targets": skipped_targets,
    }
    return augmented_df, report


def build_role_composite_augmented_dataset(
    *,
    base_dataset: str,
    base_year: int,
    target_year: int,
    top_n_targets: int = 20,
    donors_per_target: int = 5,
    max_older_distance: float = 3.0,
    max_worker_distance: float = 3.0,
    clone_weight_scale: float = 0.1,
) -> tuple[Dataset, dict[str, object]]:
    augmented_df, report = build_role_composite_augmented_input_dataframe(
        base_dataset=base_dataset,
        base_year=base_year,
        target_year=target_year,
        top_n_targets=top_n_targets,
        donors_per_target=donors_per_target,
        max_older_distance=max_older_distance,
        max_worker_distance=max_worker_distance,
        clone_weight_scale=clone_weight_scale,
    )
    return Dataset.from_dataframe(augmented_df, base_year), report


def build_donor_backed_augmented_dataset(
    *,
    base_dataset: str,
    base_year: int,
    target_year: int,
    top_n_targets: int = 20,
    donors_per_target: int = 5,
    max_distance_for_clone: float = 3.0,
    clone_weight_scale: float = 0.1,
) -> tuple[Dataset, dict[str, object]]:
    augmented_df, report = build_donor_backed_augmented_input_dataframe(
        base_dataset=base_dataset,
        base_year=base_year,
        target_year=target_year,
        top_n_targets=top_n_targets,
        donors_per_target=donors_per_target,
        max_distance_for_clone=max_distance_for_clone,
        clone_weight_scale=clone_weight_scale,
    )
    return Dataset.from_dataframe(augmented_df, base_year), report


def _safe_split(numerator: float, denominator: float, fallback: float) -> float:
    if denominator <= 0:
        return fallback
    return float(numerator / denominator)


def _safe_series_split(
    numerator: pd.Series,
    denominator: pd.Series,
    fallback: float,
) -> pd.Series:
    numerator = numerator.astype(float)
    denominator = denominator.astype(float)
    result = pd.Series(fallback, index=numerator.index, dtype=float)
    positive = denominator > 0
    result.loc[positive] = numerator.loc[positive] / denominator.loc[positive]
    return result


def build_donor_backed_clones(
    candidates: list[SyntheticCandidate],
    weights: np.ndarray,
    actual_summary: pd.DataFrame,
    *,
    ss_scale: float,
    earnings_scale: float,
    top_n_targets: int,
    donors_per_target: int,
    max_distance_for_clone: float = 3.0,
) -> tuple[list[SyntheticCandidate], np.ndarray, dict[str, object]]:
    exact_df = summarize_exact_candidates(candidates, weights)
    target_df = exact_df[exact_df["synthetic_weight"] > 0].head(top_n_targets).copy()
    scaled_actual = build_scaled_actual_summary(
        actual_summary,
        ss_scale=ss_scale,
        earnings_scale=earnings_scale,
    )

    donor_backed_candidates: list[SyntheticCandidate] = []
    donor_backed_weights: list[float] = []
    clone_records: list[dict[str, object]] = []
    outlier_targets: list[dict[str, object]] = []

    for _, target_row in target_df.iterrows():
        target_candidate = candidates[int(target_row["candidate_idx"])]
        donor_matches = match_real_donors_for_target(
            target_row,
            scaled_actual,
            donors_per_target=donors_per_target,
        )
        usable = donor_matches[donor_matches["distance"] <= max_distance_for_clone].copy()
        if usable.empty:
            donor_backed_candidates.append(target_candidate)
            donor_backed_weights.append(float(target_row["synthetic_weight"]))
            outlier_targets.append(
                {
                    "candidate_idx": int(target_row["candidate_idx"]),
                    "archetype": target_row["archetype"],
                    "weight_share_pct": float(target_row["weight_share_pct"]),
                    "best_distance": float(donor_matches["distance"].min()),
                }
            )
            continue

        per_clone_weight = float(target_row["synthetic_weight"]) / len(usable)
        target_payroll_total = float(target_row["payroll_total"])
        target_ss_total = float(target_row["ss_total"])
        target_head_payroll_share = _safe_split(
            target_candidate.head_wages,
            target_candidate.payroll_total,
            1.0 if target_candidate.spouse_age is None else 0.5,
        )
        target_head_ss_share = _safe_split(
            target_candidate.head_ss,
            target_candidate.ss_total,
            1.0 if target_candidate.spouse_age is None else 0.5,
        )

        for _, donor_row in usable.iterrows():
            donor_head_payroll_share = _safe_split(
                float(donor_row["scaled_head_payroll"]),
                float(donor_row["scaled_payroll_total"]),
                target_head_payroll_share,
            )
            donor_head_ss_share = _safe_split(
                float(donor_row["scaled_head_ss"]),
                float(donor_row["scaled_ss_total"]),
                target_head_ss_share,
            )
            donor_backed_candidates.append(
                SyntheticCandidate(
                    archetype=target_candidate.archetype,
                    head_age=target_candidate.head_age,
                    spouse_age=target_candidate.spouse_age,
                    dependent_ages=target_candidate.dependent_ages,
                    head_wages=target_payroll_total * donor_head_payroll_share,
                    spouse_wages=target_payroll_total * (1.0 - donor_head_payroll_share),
                    head_ss=target_ss_total * donor_head_ss_share,
                    spouse_ss=target_ss_total * (1.0 - donor_head_ss_share),
                    pension_income=float(donor_row["pension_income"]) * earnings_scale,
                    dividend_income=float(donor_row["dividend_income"]) * earnings_scale,
                )
            )
            donor_backed_weights.append(per_clone_weight)
            clone_records.append(
                {
                    "candidate_idx": int(target_row["candidate_idx"]),
                    "archetype": target_row["archetype"],
                    "tax_unit_id": int(donor_row["tax_unit_id"]),
                    "distance": float(donor_row["distance"]),
                    "assigned_weight_share_pct": float(
                        per_clone_weight / max(float(weights.sum()), 1.0) * 100
                    ),
                }
            )

    clone_summary = summarize_solution(
        donor_backed_candidates,
        np.asarray(donor_backed_weights, dtype=float),
        actual_summary,
    )
    return (
        donor_backed_candidates,
        np.asarray(donor_backed_weights, dtype=float),
        {
            "top_n_targets": int(top_n_targets),
            "donors_per_target": int(donors_per_target),
            "max_distance_for_clone": float(max_distance_for_clone),
            "outlier_targets": outlier_targets,
            "clone_records": clone_records[:100],
            "clone_summary": clone_summary,
        },
    )


def solve_synthetic_support(
    candidates: list[SyntheticCandidate],
    *,
    year: int,
    max_constraint_error_pct: float = 0.0,
    warm_weights: np.ndarray | None = None,
    baseline_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    problem = build_synthetic_constraint_problem(
        candidates,
        year=year,
        baseline_weights=baseline_weights,
    )
    X = problem["X"]
    aggregated_age_targets = problem["aggregated_age_targets"]
    ss_values = problem["ss_values"]
    payroll_values = problem["payroll_values"]
    baseline_weights = problem["baseline_weights"]
    ss_target = problem["ss_target"]
    payroll_target = problem["payroll_target"]

    if max_constraint_error_pct > 0:
        try:
            weights, iterations, info = calibrate_entropy_bounded(
                X,
                aggregated_age_targets,
                baseline_weights,
                ss_values=ss_values,
                ss_target=ss_target,
                payroll_values=payroll_values,
                payroll_target=payroll_target,
                n_ages=X.shape[1],
                max_constraint_error_pct=max_constraint_error_pct,
                max_iters=500,
                tol=1e-9,
                warm_weights=([warm_weights] if warm_weights is not None else None),
            )
            return np.asarray(weights, dtype=float), {
                "method": "bounded_entropy",
                "iterations": int(iterations),
                "best_case_max_pct_error": float(info["best_case_max_pct_error"]),
                "requested_max_constraint_error_pct": float(max_constraint_error_pct),
                "age_bucket_size": 5,
                "status": int(info.get("status", 0)),
                "message": info.get("message"),
            }
        except RuntimeError as error:
            constraint_matrix, targets = build_constraint_matrix(problem)
            lp_weights = None
            if warm_weights is not None:
                lp_weights = np.asarray(warm_weights, dtype=float)
            else:
                feasibility = assess_nonnegative_feasibility(
                    constraint_matrix,
                    targets,
                    return_weights=True,
                )
                if feasibility["success"] and feasibility.get("weights") is not None:
                    lp_weights = np.asarray(feasibility["weights"], dtype=float)
            if lp_weights is None:
                raise RuntimeError(
                    f"Approximate synthetic support solve failed for {year}: {error}"
                ) from error
            dense_weights, dense_info = densify_lp_solution(
                constraint_matrix,
                targets,
                baseline_weights,
                lp_weights,
                max_constraint_error_pct,
            )
            return np.asarray(dense_weights, dtype=float), {
                "method": (
                    "lp_blend"
                    if dense_info["densification_effective"]
                    else "lp_minimax"
                ),
                "iterations": 1,
                "best_case_max_pct_error": float(
                    dense_info["best_case_max_pct_error"]
                ),
                "requested_max_constraint_error_pct": float(max_constraint_error_pct),
                "age_bucket_size": 5,
                "entropy_error": str(error),
                "lp_blend_lambda": float(dense_info["blend_lambda"]),
            }

    try:
        weights, iterations = calibrate_entropy(
            X,
            aggregated_age_targets,
            baseline_weights,
            ss_values=ss_values,
            ss_target=ss_target,
            payroll_values=payroll_values,
            payroll_target=payroll_target,
            n_ages=X.shape[1],
            max_iters=500,
            tol=1e-9,
        )
        return weights, {
            "method": "entropy",
            "iterations": int(iterations),
            "best_case_max_pct_error": 0.0,
            "age_bucket_size": 5,
        }
    except RuntimeError as error:
        constraint_matrix, targets = build_constraint_matrix(problem)
        feasibility = assess_nonnegative_feasibility(
            constraint_matrix,
            targets,
            return_weights=True,
        )
        if not feasibility["success"] or feasibility.get("weights") is None:
            raise RuntimeError(
                f"Synthetic support could not match {year} targets: {error}"
            ) from error
        return np.asarray(feasibility["weights"], dtype=float), {
            "method": "lp_minimax",
            "iterations": 1,
            "best_case_max_pct_error": feasibility["best_case_max_pct_error"],
            "age_bucket_size": 5,
            "entropy_error": str(error),
        }


def summarize_solution_diff(
    candidates: list[SyntheticCandidate],
    base_weights: np.ndarray,
    alt_weights: np.ndarray,
) -> dict[str, object]:
    base_weights = np.asarray(base_weights, dtype=float)
    alt_weights = np.asarray(alt_weights, dtype=float)
    base_total = max(float(base_weights.sum()), 1.0)
    alt_total = max(float(alt_weights.sum()), 1.0)
    rows: list[dict[str, object]] = []
    for candidate, base_weight, alt_weight in zip(candidates, base_weights, alt_weights):
        base_share_pct = float(base_weight / base_total * 100)
        alt_share_pct = float(alt_weight / alt_total * 100)
        rows.append(
            {
                "archetype": candidate.archetype,
                "head_age": candidate.head_age,
                "spouse_age": candidate.spouse_age,
                "dependent_count": len(candidate.dependent_ages),
                "payroll_total": candidate.payroll_total,
                "ss_total": candidate.ss_total,
                "base_weight_share_pct": base_share_pct,
                "alt_weight_share_pct": alt_share_pct,
                "weight_share_gain_pct_points": alt_share_pct - base_share_pct,
                "newly_entering": alt_weight > 0 and base_weight <= 1e-12,
            }
        )
    diff_df = pd.DataFrame(rows)
    entrants = diff_df[
        diff_df["newly_entering"] & diff_df["alt_weight_share_pct"].gt(0.01)
    ].copy()
    entrant_archetypes = (
        entrants.groupby("archetype", as_index=False)
        .agg(
            entrant_weight_share_pct=("alt_weight_share_pct", "sum"),
            entrant_candidate_count=("archetype", "count"),
        )
        .sort_values("entrant_weight_share_pct", ascending=False)
    )
    return {
        "top_weight_gainers": diff_df.sort_values(
            "weight_share_gain_pct_points",
            ascending=False,
        )
        .head(20)
        .to_dict("records"),
        "entrant_archetypes": entrant_archetypes.head(12).to_dict("records"),
        "entrant_candidate_count": int(len(entrants)),
    }


def summarize_solution(
    candidates: list[SyntheticCandidate],
    weights: np.ndarray,
    actual_summary: pd.DataFrame,
) -> dict[str, object]:
    weight_sum = float(weights.sum())
    candidate_df = pd.DataFrame(
        [
            {
                **asdict(candidate),
                "spouse_age": candidate.spouse_age,
                "dependent_count": len(candidate.dependent_ages),
                "payroll_total": candidate.payroll_total,
                "ss_total": candidate.ss_total,
                "taxable_benefits_proxy": candidate.taxable_benefits_proxy(),
                "synthetic_weight": float(weight),
            }
            for candidate, weight in zip(candidates, weights)
        ]
    )
    candidate_df["weight_share_pct"] = (
        candidate_df["synthetic_weight"] / weight_sum * 100 if weight_sum > 0 else 0.0
    )
    candidate_df = candidate_df.sort_values("synthetic_weight", ascending=False)
    positive_weights = candidate_df.loc[
        candidate_df["synthetic_weight"] > 0,
        "synthetic_weight",
    ].to_numpy(dtype=float)
    if positive_weights.size > 0:
        effective_sample_size = float(
            (positive_weights.sum() ** 2) / np.sum(positive_weights**2)
        )
        top_10_weight_share_pct = float(
            positive_weights[:10].sum() / positive_weights.sum() * 100
        )
        top_20_weight_share_pct = float(
            positive_weights[:20].sum() / positive_weights.sum() * 100
        )
    else:
        effective_sample_size = 0.0
        top_10_weight_share_pct = 0.0
        top_20_weight_share_pct = 0.0

    def _weighted_mean(group: pd.DataFrame, column: str) -> float:
        total = float(group["synthetic_weight"].sum())
        if total <= 0:
            return 0.0
        return float(np.average(group[column], weights=group["synthetic_weight"]))

    synthetic_rows = []
    for archetype, group in candidate_df.groupby("archetype", sort=False):
        synthetic_rows.append(
            {
                "archetype": archetype,
                "synthetic_weight": float(group["synthetic_weight"].sum()),
                "candidate_count": int(len(group)),
                "avg_head_age": _weighted_mean(group, "head_age"),
                "avg_payroll_total": _weighted_mean(group, "payroll_total"),
                "avg_ss_total": _weighted_mean(group, "ss_total"),
                "avg_pension_income": _weighted_mean(group, "pension_income"),
                "avg_dividend_income": _weighted_mean(group, "dividend_income"),
            }
        )
    synthetic_archetypes = pd.DataFrame(synthetic_rows).sort_values(
        "synthetic_weight",
        ascending=False,
    )
    synthetic_archetypes["synthetic_weight_share_pct"] = (
        synthetic_archetypes["synthetic_weight"] / weight_sum * 100
        if weight_sum > 0
        else 0.0
    )

    actual_archetypes = (
        actual_summary.groupby("archetype", as_index=False)
        .agg(
            actual_support_count=("archetype", "count"),
            avg_head_age=("head_age", "mean"),
            avg_payroll_total=("payroll_total", "mean"),
            avg_ss_total=("ss_total", "mean"),
            avg_pension_income=("pension_income", "mean"),
            avg_dividend_income=("dividend_income", "mean"),
        )
        .sort_values("actual_support_count", ascending=False)
    )
    actual_total = float(actual_archetypes["actual_support_count"].sum())
    actual_archetypes["actual_support_share_pct"] = (
        actual_archetypes["actual_support_count"] / actual_total * 100
        if actual_total > 0
        else 0.0
    )

    comparison = pd.merge(
        synthetic_archetypes[
            ["archetype", "synthetic_weight_share_pct", "candidate_count"]
        ],
        actual_archetypes[
            ["archetype", "actual_support_share_pct", "actual_support_count"]
        ],
        on="archetype",
        how="outer",
    ).fillna(0.0)
    comparison["share_gap_pct_points"] = (
        comparison["synthetic_weight_share_pct"]
        - comparison["actual_support_share_pct"]
    )
    comparison = comparison.sort_values(
        "synthetic_weight_share_pct",
        ascending=False,
    )

    weighted_metrics = {
        "synthetic_payroll_positive_85_plus_household_share_pct": float(
            candidate_df.loc[
                (candidate_df["head_age"] >= 85) & (candidate_df["payroll_total"] > 0),
                "synthetic_weight",
            ].sum()
            / weight_sum
            * 100
        )
        if weight_sum > 0
        else 0.0,
        "synthetic_mixed_retiree_worker_share_pct": float(
            candidate_df.loc[
                candidate_df["archetype"].eq("mixed_retiree_worker_couple"),
                "synthetic_weight",
            ].sum()
            / weight_sum
            * 100
        )
        if weight_sum > 0
        else 0.0,
        "synthetic_units_with_positive_pension_or_dividend_share_pct": float(
            candidate_df.loc[
                (candidate_df["pension_income"] > 0)
                | (candidate_df["dividend_income"] > 0),
                "synthetic_weight",
            ].sum()
            / weight_sum
            * 100
        )
        if weight_sum > 0
        else 0.0,
        "synthetic_avg_taxable_benefits_proxy_share_pct": float(
            (
                (candidate_df["taxable_benefits_proxy"] * candidate_df["synthetic_weight"]).sum()
                / max((candidate_df["ss_total"] * candidate_df["synthetic_weight"]).sum(), 1.0)
            )
            * 100
        ),
    }

    return {
        "synthetic_candidate_count": int(len(candidate_df)),
        "positive_weight_candidate_count": int((candidate_df["synthetic_weight"] > 0).sum()),
        "effective_sample_size": effective_sample_size,
        "top_10_weight_share_pct": top_10_weight_share_pct,
        "top_20_weight_share_pct": top_20_weight_share_pct,
        "top_candidates": candidate_df.head(20).to_dict("records"),
        "synthetic_archetypes": synthetic_archetypes.to_dict("records"),
        "actual_support_archetypes": actual_archetypes.to_dict("records"),
        "archetype_gap_table": comparison.to_dict("records"),
        "weighted_metrics": weighted_metrics,
    }


def main() -> int:
    args = parse_args()
    set_long_term_target_source(args.target_source)

    actual_summary = build_actual_tax_unit_summary(args.base_dataset)
    base_aggregates = load_base_aggregates(args.base_dataset)
    ss_scale = load_ssa_benefit_projections(args.year) / max(
        base_aggregates["weighted_ss_total"],
        1.0,
    )
    earnings_scale = load_taxable_payroll_projections(args.year) / max(
        base_aggregates["weighted_payroll_total"],
        1.0,
    )
    pools = build_quantile_pools(
        actual_summary,
        ss_scale=ss_scale,
        earnings_scale=earnings_scale,
    )
    candidates = generate_synthetic_candidates(
        pools,
        payroll_cap=load_policyengine_social_security_cap(args.year),
    )
    weights, solve_info = solve_synthetic_support(candidates, year=args.year)
    solution_summary = summarize_solution(candidates, weights, actual_summary)
    donor_probe = summarize_donor_probe(
        candidates,
        weights,
        actual_summary,
        ss_scale=ss_scale,
        earnings_scale=earnings_scale,
        top_n_targets=args.donor_probe_top_n,
        donors_per_target=args.donor_probe_k,
    )
    _, _, donor_backed_clone_probe = build_donor_backed_clones(
        candidates,
        weights,
        actual_summary,
        ss_scale=ss_scale,
        earnings_scale=earnings_scale,
        top_n_targets=args.donor_probe_top_n,
        donors_per_target=args.donor_probe_k,
    )
    (
        role_composite_candidates,
        role_composite_prior,
        role_donor_composite_probe,
    ) = build_role_donor_composites(
        candidates,
        weights,
        actual_summary,
        ss_scale=ss_scale,
        earnings_scale=earnings_scale,
        top_n_targets=args.donor_probe_top_n,
        older_donors_per_target=args.donor_probe_k,
        worker_donors_per_target=args.donor_probe_k,
    )
    role_donor_composite_result: dict[str, object] = {
        "candidate_count": int(len(role_composite_candidates)),
        "prior_summary": role_donor_composite_probe["prior_summary"],
        "skipped_targets": role_donor_composite_probe["skipped_targets"],
        "composite_records": role_donor_composite_probe["composite_records"],
    }
    if role_composite_candidates:
        role_weights, role_solve_info = solve_synthetic_support(
            role_composite_candidates,
            year=args.year,
            baseline_weights=role_composite_prior,
        )
        role_donor_composite_result["solve_info"] = role_solve_info
        role_donor_composite_result["solution_summary"] = summarize_solution(
            role_composite_candidates,
            role_weights,
            actual_summary,
        )
    else:
        role_donor_composite_result["solve_info"] = {"status": "no_candidates"}
    epsilon_path_results = []
    for epsilon in parse_epsilon_path(args.epsilon_path):
        epsilon_weights, epsilon_solve_info = solve_synthetic_support(
            candidates,
            year=args.year,
            max_constraint_error_pct=epsilon,
            warm_weights=weights,
        )
        epsilon_path_results.append(
            {
                "epsilon_pct": float(epsilon),
                "solve_info": epsilon_solve_info,
                "solution_summary": summarize_solution(
                    candidates,
                    epsilon_weights,
                    actual_summary,
                ),
                "vs_exact": summarize_solution_diff(
                    candidates,
                    weights,
                    epsilon_weights,
                ),
            }
        )

    report = {
        "year": args.year,
        "target_source": args.target_source,
        "base_dataset": args.base_dataset,
        "solve_info": solve_info,
        "targets": {
            "ss_total": float(load_ssa_benefit_projections(args.year)),
            "taxable_payroll": float(load_taxable_payroll_projections(args.year)),
        },
        "macro_scales": {
            "ss_scale": float(ss_scale),
            "earnings_scale": float(earnings_scale),
            **base_aggregates,
        },
        "actual_support_tax_unit_count": int(len(actual_summary)),
        "synthetic_solution": solution_summary,
        "donor_probe": donor_probe,
        "donor_backed_clone_probe": donor_backed_clone_probe,
        "role_donor_composite_probe": role_donor_composite_result,
        "epsilon_path": epsilon_path_results,
    }

    payload = json.dumps(report, indent=2)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
