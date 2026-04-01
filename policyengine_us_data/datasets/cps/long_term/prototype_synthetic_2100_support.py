from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from policyengine_us import Microsimulation

from calibration import assess_nonnegative_feasibility, calibrate_entropy
from projection_utils import aggregate_age_targets, build_age_bins
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
        head_ages=(67, 75, 85),
        spouse_age_offsets=(None,),
        dependent_age_sets=((),),
        ss_source="older_beneficiary",
        payroll_source="zero",
        pension_source="older_asset",
        dividend_source="older_asset",
        ss_split=(1.0, 0.0),
        payroll_split=(0.0, 0.0),
    ),
    SyntheticTemplate(
        name="older_beneficiary_couple",
        head_ages=(67, 75, 85),
        spouse_age_offsets=(-2, -5),
        dependent_age_sets=((),),
        ss_source="older_couple_beneficiary",
        payroll_source="zero",
        pension_source="older_asset",
        dividend_source="older_asset",
        ss_split=(0.55, 0.45),
        payroll_split=(0.0, 0.0),
    ),
    SyntheticTemplate(
        name="older_worker_single",
        head_ages=(67, 75, 80),
        spouse_age_offsets=(None,),
        dependent_age_sets=((),),
        ss_source="older_worker",
        payroll_source="older_worker",
        pension_source="older_asset",
        dividend_source="older_asset",
        ss_split=(1.0, 0.0),
        payroll_split=(1.0, 0.0),
    ),
    SyntheticTemplate(
        name="older_worker_couple",
        head_ages=(67, 75, 80),
        spouse_age_offsets=(-2,),
        dependent_age_sets=((),),
        ss_source="older_worker",
        payroll_source="older_worker",
        pension_source="older_asset",
        dividend_source="older_asset",
        ss_split=(0.55, 0.45),
        payroll_split=(0.55, 0.45),
    ),
    SyntheticTemplate(
        name="mixed_retiree_worker_couple",
        head_ages=(67, 75, 85),
        spouse_age_offsets=(-20, -25, -35),
        dependent_age_sets=((),),
        ss_source="older_beneficiary",
        payroll_source="prime_worker",
        pension_source="older_asset",
        dividend_source="older_asset",
        ss_split=(1.0, 0.0),
        payroll_split=(0.0, 1.0),
    ),
    SyntheticTemplate(
        name="prime_worker_single",
        head_ages=(22, 27, 35, 45, 60),
        spouse_age_offsets=(None,),
        dependent_age_sets=((),),
        ss_source="zero",
        payroll_source="prime_worker",
        pension_source="prime_asset",
        dividend_source="prime_asset",
        ss_split=(0.0, 0.0),
        payroll_split=(1.0, 0.0),
    ),
    SyntheticTemplate(
        name="prime_worker_couple",
        head_ages=(27, 40, 55),
        spouse_age_offsets=(-2,),
        dependent_age_sets=((),),
        ss_source="zero",
        payroll_source="prime_worker",
        pension_source="prime_asset",
        dividend_source="prime_asset",
        ss_split=(0.0, 0.0),
        payroll_split=(0.6, 0.4),
    ),
    SyntheticTemplate(
        name="prime_worker_family",
        head_ages=(27, 40, 55),
        spouse_age_offsets=(-2,),
        dependent_age_sets=((1, 3), (6, 11), (15, 17)),
        ss_source="zero",
        payroll_source="prime_worker_family",
        pension_source="prime_asset",
        dividend_source="prime_asset",
        ss_split=(0.0, 0.0),
        payroll_split=(0.6, 0.4),
    ),
    SyntheticTemplate(
        name="older_plus_prime_worker_family",
        head_ages=(67, 75, 85),
        spouse_age_offsets=(-25, -35),
        dependent_age_sets=((1, 3), (10, 15)),
        ss_source="older_beneficiary",
        payroll_source="prime_worker_family",
        pension_source="older_asset",
        dividend_source="older_asset",
        ss_split=(1.0, 0.0),
        payroll_split=(0.0, 1.0),
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
    return parser.parse_args()


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


def build_actual_tax_unit_summary(base_dataset: str) -> pd.DataFrame:
    sim = Microsimulation(dataset=base_dataset)
    input_df = sim.to_input_dataframe()

    person_df = pd.DataFrame(
        {
            "tax_unit_id": sim.calculate("person_tax_unit_id", period=BASE_YEAR).values,
            "age": sim.calculate("age", period=BASE_YEAR).values,
            "is_head": sim.calculate("is_tax_unit_head", period=BASE_YEAR).values,
            "is_spouse": sim.calculate("is_tax_unit_spouse", period=BASE_YEAR).values,
            "is_dependent": sim.calculate(
                "is_tax_unit_dependent", period=BASE_YEAR
            ).values,
            "social_security": sim.calculate(
                "social_security", period=BASE_YEAR
            ).values,
            "payroll": (
                sim.calculate("employment_income_before_lsr", period=BASE_YEAR).values
                + sim.calculate(
                    "self_employment_income_before_lsr", period=BASE_YEAR
                ).values
            ),
            "dividend_income": sim.calculate(
                "qualified_dividend_income", period=BASE_YEAR
            ).values,
            "pension_income": sim.calculate(
                "taxable_pension_income", period=BASE_YEAR
            ).values,
            "person_weight": input_df[f"person_weight__{BASE_YEAR}"].astype(float).values,
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
        row = {
            "tax_unit_id": int(tax_unit_id),
            "head_age": head_age,
            "spouse_age": spouse_age,
            "dependent_count": dependent_count,
            "payroll_total": float(group["payroll"].sum()),
            "ss_total": float(group["social_security"].sum()),
            "dividend_income": float(group["dividend_income"].sum()),
            "pension_income": float(group["pension_income"].sum()),
            "support_count_weight": 1.0,
            "person_weight_proxy": float(group["person_weight"].max()),
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
                    quantiles=(0.25, 0.5, 0.75),
                    include_zero=(name == "zero"),
                    positive_only=(name != "zero"),
                ),
                ss_scale,
            ),
            "payroll": _scale_levels(
                quantile_levels(
                    subset["payroll_total"],
                    quantiles=(0.25, 0.5, 0.75),
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
                    quantiles=(0.5, 0.9),
                    include_zero=True,
                    positive_only=True,
                ),
                earnings_scale,
            ),
            "dividend": _scale_levels(
                quantile_levels(
                    subset["dividend_income"],
                    quantiles=(0.5, 0.9),
                    include_zero=True,
                    positive_only=True,
                ),
                earnings_scale,
            ),
        }
    return pools


def generate_synthetic_candidates(
    pools: dict[str, dict[str, list[float]]],
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
                        for payroll_total in payroll_levels:
                            for pension_income in pension_levels:
                                for dividend_income in dividend_levels:
                                    head_ss = ss_total * template.ss_split[0]
                                    spouse_ss = ss_total * template.ss_split[1]
                                    head_wages = payroll_total * template.payroll_split[0]
                                    spouse_wages = payroll_total * template.payroll_split[1]
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
                                            pension_income=pension_income,
                                            dividend_income=dividend_income,
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


def solve_synthetic_support(
    candidates: list[SyntheticCandidate],
    *,
    year: int,
) -> tuple[np.ndarray, dict[str, object]]:
    age_targets = load_ssa_age_projections(start_year=year, end_year=year)
    age_bins = build_age_bins(n_ages=age_targets.shape[0], bucket_size=5)
    aggregated_age_targets = aggregate_age_targets(age_targets, age_bins)[:, 0]

    X = np.vstack([age_bucket_vector(candidate.ages(), age_bins) for candidate in candidates])
    ss_values = np.array([candidate.ss_total for candidate in candidates], dtype=float)
    payroll_values = np.array(
        [candidate.payroll_total for candidate in candidates], dtype=float
    )
    baseline_weights = np.ones(len(candidates), dtype=float)

    try:
        weights, iterations = calibrate_entropy(
            X,
            aggregated_age_targets,
            baseline_weights,
            ss_values=ss_values,
            ss_target=load_ssa_benefit_projections(year),
            payroll_values=payroll_values,
            payroll_target=load_taxable_payroll_projections(year),
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
        feasibility = assess_nonnegative_feasibility(
            np.column_stack([X, ss_values, payroll_values]),
            np.concatenate(
                [
                    aggregated_age_targets,
                    np.array(
                        [
                            load_ssa_benefit_projections(year),
                            load_taxable_payroll_projections(year),
                        ]
                    ),
                ]
            ),
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
    candidates = generate_synthetic_candidates(pools)
    weights, solve_info = solve_synthetic_support(candidates, year=args.year)
    solution_summary = summarize_solution(candidates, weights, actual_summary)

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
