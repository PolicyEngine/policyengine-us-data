"""Validate a national US.h5 file against reference values.

Loads the national H5, computes key variables, and compares to
known national totals. Also runs structural sanity checks.

Usage:
    python -m policyengine_us_data.calibration.validate_national_h5
    python -m policyengine_us_data.calibration.validate_national_h5 \
        --h5-path path/to/US.h5
    python -m policyengine_us_data.calibration.validate_national_h5 \
        --hf-path hf://policyengine/policyengine-us-data/national/US.h5
"""

import argparse
import os

import numpy as np
import pandas as pd

from policyengine_us_data.calibration.ctc_diagnostics import (
    create_ctc_diagnostic_tables,
    format_ctc_diagnostic_table,
)
from policyengine_us_data.db.etl_irs_soi import get_national_geography_soi_target

VARIABLES = [
    "adjusted_gross_income",
    "employment_income",
    "total_self_employment_income",
    "tax_unit_partnership_s_corp_income",
    "taxable_pension_income",
    "dividend_income",
    "net_capital_gains",
    "rental_income",
    "taxable_interest_income",
    "social_security",
    "snap",
    "ssi",
    "income_tax_before_credits",
    "ctc",
    "eitc",
    "non_refundable_ctc",
    "refundable_ctc",
    "real_estate_taxes",
    "rent",
    "is_pregnant",
    "ctc_qualifying_children",
    "person_count",
    "household_count",
]

REFERENCES = {
    "person_count": (335_000_000, "~335M"),
    "household_count": (130_000_000, "~130M"),
    "adjusted_gross_income": (15_000_000_000_000, "~$15T"),
    "employment_income": (10_000_000_000_000, "~$10T"),
    "social_security": (1_200_000_000_000, "~$1.2T"),
    "snap": (110_000_000_000, "~$110B"),
    "ssi": (60_000_000_000, "~$60B"),
    "eitc": (67_000_000_000, "~$67B"),
    "income_tax_before_credits": (4_000_000_000_000, "~$4T"),
}

DEFAULT_HF_PATH = "hf://policyengine/policyengine-us-data/national/US.h5"
PUB_4801_2022_CTC_QUALIFYING_CHILDREN = 63_622_000.0
ARTIFACT_CTC_SUMMARY_VARIABLES = [
    "ctc_qualifying_children",
    "ctc",
    "refundable_ctc",
    "non_refundable_ctc",
]
ADVANCE_CTC_2021_AGI_BANDS = [
    (-np.inf, 1.0, "No adjusted gross income [4]"),
    (1.0, 10_000.0, "$1 under $10,000"),
    (10_000.0, 20_000.0, "$10,000 under $20,000"),
    (20_000.0, 30_000.0, "$20,000 under $30,000"),
    (30_000.0, 40_000.0, "$30,000 under $40,000"),
    (40_000.0, 50_000.0, "$40,000 under $50,000"),
    (50_000.0, 60_000.0, "$50,000 under $60,000"),
    (60_000.0, 75_000.0, "$60,000 under $75,000"),
    (75_000.0, 100_000.0, "$75,000 under $100,000"),
    (100_000.0, 200_000.0, "$100,000 under $200,000"),
    (200_000.0, 400_000.0, "$200,000 under $400,000"),
    (400_000.0, np.inf, "$400,000 or more"),
]
ADVANCE_CTC_2021_AGI_REFERENCE_ROWS = [
    ("No adjusted gross income [4]", 349_933.0),
    ("$1 under $10,000", 3_604_619.0),
    ("$10,000 under $20,000", 6_379_391.0),
    ("$20,000 under $30,000", 7_419_453.0),
    ("$30,000 under $40,000", 6_689_905.0),
    ("$40,000 under $50,000", 5_093_918.0),
    ("$50,000 under $60,000", 4_008_971.0),
    ("$60,000 under $75,000", 4_943_145.0),
    ("$75,000 under $100,000", 6_522_488.0),
    ("$100,000 under $200,000", 12_149_615.0),
    ("$200,000 under $400,000", 4_174_996.0),
    ("$400,000 or more", 639_657.0),
]
ADVANCE_CTC_2021_FILING_STATUS_REFERENCE_ROWS = [
    ("Single [4]", 3_362_687.0),
    ("Married filing a joint return", 35_165_385.0),
    ("Married filing separate returns", 1_038_058.0),
    ("Head of household", 22_346_185.0),
    ("Qualifying Widow/Widower", 63_776.0),
]
ADVANCE_CTC_2021_FILING_STATUS_ORDER = [
    "Single [4]",
    "Head of household",
    "Married filing a joint return",
    "Married filing separate returns",
    "Qualifying Widow/Widower",
    "Other",
]
ADVANCE_CTC_2021_FILING_STATUS_MAP = {
    "SINGLE": "Single [4]",
    "HEAD_OF_HOUSEHOLD": "Head of household",
    "JOINT": "Married filing a joint return",
    "SEPARATE": "Married filing separate returns",
    "SURVIVING_SPOUSE": "Qualifying Widow/Widower",
}

COUNT_VARS = {
    "person_count",
    "household_count",
    "is_pregnant",
    "ctc_qualifying_children",
}

CANONICAL_CTC_REFORM_VARIABLES = [
    "ctc_value",
    "ctc",
    "refundable_ctc",
    "non_refundable_ctc",
    "eitc",
    "household_net_income",
]

CANONICAL_CTC_REFORM_DICT = {
    "gov.irs.credits.eitc.max[0].amount": {"2025-01-01.2100-12-31": 2_000},
    "gov.irs.credits.eitc.max[1].amount": {"2025-01-01.2100-12-31": 2_000},
    "gov.irs.credits.eitc.max[2].amount": {"2025-01-01.2100-12-31": 2_000},
    "gov.irs.credits.eitc.max[3].amount": {"2025-01-01.2100-12-31": 2_000},
    "gov.irs.credits.ctc.phase_out.amount": {"2025-01-01.2100-12-31": 25},
    "gov.irs.credits.ctc.amount.arpa[0].amount": {"2025-01-01.2100-12-31": 4_800},
    "gov.irs.credits.ctc.amount.arpa[1].amount": {"2025-01-01.2100-12-31": 4_800},
    "gov.irs.credits.ctc.phase_out.arpa.amount": {"2025-01-01.2100-12-31": 25},
    "gov.contrib.ctc.minimum_refundable.in_effect": {"2025-01-01.2100-12-31": True},
    "gov.contrib.ctc.per_child_phase_in.in_effect": {"2025-01-01.2100-12-31": True},
    "gov.irs.credits.ctc.phase_out.arpa.in_effect": {"2025-01-01.2100-12-31": True},
    "gov.irs.credits.ctc.refundable.phase_in.rate": {"2025-01-01.2100-12-31": 0.2},
    "gov.irs.credits.eitc.phase_in_rate[0].amount": {"2025-01-01.2100-12-31": 0.2},
    "gov.irs.credits.eitc.phase_in_rate[1].amount": {"2025-01-01.2100-12-31": 0.2},
    "gov.irs.credits.eitc.phase_in_rate[2].amount": {"2025-01-01.2100-12-31": 0.2},
    "gov.irs.credits.eitc.phase_in_rate[3].amount": {"2025-01-01.2100-12-31": 0.2},
    "gov.contrib.ctc.per_child_phase_out.in_effect": {"2025-01-01.2100-12-31": True},
    "gov.irs.credits.ctc.phase_out.threshold.JOINT": {"2025-01-01.2100-12-31": 200_000},
    "gov.irs.credits.ctc.refundable.individual_max": {"2025-01-01.2100-12-31": 4_800},
    "gov.irs.credits.eitc.phase_out.rate[0].amount": {"2025-01-01.2100-12-31": 0.1},
    "gov.irs.credits.eitc.phase_out.rate[1].amount": {"2025-01-01.2100-12-31": 0.1},
    "gov.irs.credits.eitc.phase_out.rate[2].amount": {"2025-01-01.2100-12-31": 0.1},
    "gov.irs.credits.eitc.phase_out.rate[3].amount": {"2025-01-01.2100-12-31": 0.1},
    "gov.irs.credits.ctc.phase_out.threshold.SINGLE": {
        "2025-01-01.2100-12-31": 100_000
    },
    "gov.irs.credits.eitc.phase_out.start[0].amount": {"2025-01-01.2100-12-31": 20_000},
    "gov.irs.credits.eitc.phase_out.start[1].amount": {"2025-01-01.2100-12-31": 20_000},
    "gov.irs.credits.eitc.phase_out.start[2].amount": {"2025-01-01.2100-12-31": 20_000},
    "gov.irs.credits.eitc.phase_out.start[3].amount": {"2025-01-01.2100-12-31": 20_000},
    "gov.irs.credits.ctc.phase_out.threshold.SEPARATE": {
        "2025-01-01.2100-12-31": 100_000
    },
    "gov.contrib.ctc.per_child_phase_out.avoid_overlap": {
        "2025-01-01.2100-12-31": True
    },
    "gov.irs.credits.ctc.refundable.phase_in.threshold": {"2025-01-01.2100-12-31": 0},
    "gov.irs.credits.ctc.phase_out.arpa.threshold.JOINT": {
        "2025-01-01.2100-12-31": 35_000
    },
    "gov.contrib.ctc.minimum_refundable.amount[0].amount": {
        "2025-01-01.2100-12-31": 2_400
    },
    "gov.contrib.ctc.minimum_refundable.amount[1].amount": {
        "2025-01-01.2100-12-31": 2_400
    },
    "gov.irs.credits.ctc.phase_out.arpa.threshold.SINGLE": {
        "2025-01-01.2100-12-31": 25_000
    },
    "gov.irs.credits.eitc.phase_out.joint_bonus[0].amount": {
        "2025-01-01.2100-12-31": 7_000
    },
    "gov.irs.credits.eitc.phase_out.joint_bonus[1].amount": {
        "2025-01-01.2100-12-31": 7_000
    },
    "gov.irs.credits.ctc.phase_out.arpa.threshold.SEPARATE": {
        "2025-01-01.2100-12-31": 25_000
    },
    "gov.irs.credits.ctc.phase_out.threshold.SURVIVING_SPOUSE": {
        "2025-01-01.2100-12-31": 100_000
    },
    "gov.irs.credits.ctc.phase_out.threshold.HEAD_OF_HOUSEHOLD": {
        "2025-01-01.2100-12-31": 100_000
    },
    "gov.irs.credits.ctc.phase_out.arpa.threshold.SURVIVING_SPOUSE": {
        "2025-01-01.2100-12-31": 25_000
    },
    "gov.irs.credits.ctc.phase_out.arpa.threshold.HEAD_OF_HOUSEHOLD": {
        "2025-01-01.2100-12-31": 25_000
    },
}


def get_reference_values(reference_year: int = 2024):
    """Return national validation references for the current production year."""
    references = dict(REFERENCES)
    references["ctc_qualifying_children"] = (
        PUB_4801_2022_CTC_QUALIFYING_CHILDREN,
        "IRS Pub. 4801 2022 63.6M",
    )
    for variable in ("refundable_ctc", "non_refundable_ctc"):
        target = get_national_geography_soi_target(
            variable,
            reference_year,
        )
        references[variable] = (
            target["amount"],
            f"IRS SOI {target['source_year']} ${target['amount'] / 1e9:.1f}B",
        )
    return references


def _build_reference_share_table(
    rows: list[tuple[str, float]],
) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=["group", "reference_children_2021"])
    table["reference_share_2021"] = (
        table["reference_children_2021"] / table["reference_children_2021"].sum()
    )
    return table


def _get_advance_ctc_agi_reference() -> pd.DataFrame:
    return _build_reference_share_table(ADVANCE_CTC_2021_AGI_REFERENCE_ROWS)


def _get_advance_ctc_filing_status_reference() -> pd.DataFrame:
    return _build_reference_share_table(ADVANCE_CTC_2021_FILING_STATUS_REFERENCE_ROWS)


def _assign_advance_ctc_agi_bands(
    adjusted_gross_income: np.ndarray,
) -> pd.Categorical:
    labels = [label for _, _, label in ADVANCE_CTC_2021_AGI_BANDS]
    agi_band = np.full(len(adjusted_gross_income), labels[-1], dtype=object)
    for lower, upper, label in ADVANCE_CTC_2021_AGI_BANDS:
        mask = (adjusted_gross_income >= lower) & (adjusted_gross_income < upper)
        agi_band[mask] = label
    return pd.Categorical(agi_band, categories=labels, ordered=True)


def _normalize_advance_ctc_filing_status(
    filing_status: pd.Series,
) -> pd.Categorical:
    labels = [
        ADVANCE_CTC_2021_FILING_STATUS_MAP.get(str(value), "Other")
        for value in filing_status.astype(str)
    ]
    return pd.Categorical(
        labels,
        categories=ADVANCE_CTC_2021_FILING_STATUS_ORDER,
        ordered=True,
    )


def _build_external_share_comparison(
    reference: pd.DataFrame,
    simulated: pd.DataFrame,
) -> pd.DataFrame:
    comparison = reference.merge(simulated, on="group", how="left")
    comparison["simulated_children"] = comparison["simulated_children"].fillna(0.0)
    comparison["simulated_share"] = comparison["simulated_share"].fillna(0.0)
    comparison["share_delta_pp"] = (
        comparison["simulated_share"] - comparison["reference_share_2021"]
    ) * 100
    return comparison


def build_advance_ctc_agi_share_comparison(
    sim,
    *,
    period: int = 2025,
) -> pd.DataFrame:
    adjusted_gross_income = sim.calculate(
        "adjusted_gross_income", period=period
    ).values.astype(float)
    ctc_qualifying_children = sim.calculate(
        "ctc_qualifying_children", period=period
    ).values.astype(float)
    tax_unit_weight = sim.calculate("tax_unit_weight", period=period).values.astype(
        float
    )

    frame = pd.DataFrame(
        {
            "group": _assign_advance_ctc_agi_bands(adjusted_gross_income),
            "simulated_children": ctc_qualifying_children * tax_unit_weight,
        }
    )
    simulated = (
        frame.groupby("group", observed=False)["simulated_children"].sum().reset_index()
    )
    total = simulated["simulated_children"].sum()
    simulated["simulated_share"] = (
        simulated["simulated_children"] / total if total else 0.0
    )
    return _build_external_share_comparison(
        _get_advance_ctc_agi_reference(),
        simulated,
    )


def build_advance_ctc_filing_status_share_comparison(
    sim,
    *,
    period: int = 2025,
) -> pd.DataFrame:
    filing_status = pd.Series(sim.calculate("filing_status", period=period).values)
    ctc_qualifying_children = sim.calculate(
        "ctc_qualifying_children", period=period
    ).values.astype(float)
    tax_unit_weight = sim.calculate("tax_unit_weight", period=period).values.astype(
        float
    )

    frame = pd.DataFrame(
        {
            "group": _normalize_advance_ctc_filing_status(filing_status),
            "simulated_children": ctc_qualifying_children * tax_unit_weight,
        }
    )
    simulated = (
        frame.groupby("group", observed=False)["simulated_children"].sum().reset_index()
    )
    total = simulated["simulated_children"].sum()
    simulated["simulated_share"] = (
        simulated["simulated_children"] / total if total else 0.0
    )
    return _build_external_share_comparison(
        _get_advance_ctc_filing_status_reference(),
        simulated,
    )


def _format_external_share_comparison(table: pd.DataFrame) -> str:
    display = table.copy()
    for column in ("simulated_children", "reference_children_2021"):
        display[column] = display[column].map(lambda value: f"{value / 1e6:,.2f}M")
    for column in ("simulated_share", "reference_share_2021"):
        display[column] = display[column].map(lambda value: f"{value * 100:,.1f}%")
    display["share_delta_pp"] = display["share_delta_pp"].map(
        lambda value: f"{value:+.1f}pp"
    )
    return display.to_string(index=False)


def get_external_ctc_benchmark_outputs(
    sim,
    *,
    period: int = 2025,
) -> dict[str, str]:
    """Return contextual external CTC child-mix benchmarks from public IRS data."""
    return {
        "CTC QUALIFYING-CHILD SHARE VS 2021 ADVANCE CTC ADMIN DATA BY AGI BAND": (
            _format_external_share_comparison(
                build_advance_ctc_agi_share_comparison(sim, period=period)
            )
        ),
        "CTC QUALIFYING-CHILD SHARE VS 2021 ADVANCE CTC ADMIN DATA BY FILING STATUS": (
            _format_external_share_comparison(
                build_advance_ctc_filing_status_share_comparison(
                    sim,
                    period=period,
                )
            )
        ),
    }


def get_ctc_diagnostic_outputs(sim) -> dict[str, str]:
    """Return formatted CTC diagnostics for human-readable validation output."""
    tables = create_ctc_diagnostic_tables(sim)
    outputs = {
        "CTC DIAGNOSTICS BY AGI BAND": format_ctc_diagnostic_table(
            tables["by_agi_band"]
        ),
        "CTC DIAGNOSTICS BY FILING STATUS": format_ctc_diagnostic_table(
            tables["by_filing_status"]
        ),
    }
    if "by_agi_band_and_filing_status" in tables:
        outputs["CTC DIAGNOSTICS BY AGI BAND AND FILING STATUS"] = (
            format_ctc_diagnostic_table(tables["by_agi_band_and_filing_status"])
        )
    if "by_child_count" in tables:
        outputs["CTC DIAGNOSTICS BY QUALIFYING-CHILD COUNT"] = (
            format_ctc_diagnostic_table(tables["by_child_count"])
        )
    if "by_child_age" in tables:
        outputs["CTC DIAGNOSTICS BY QUALIFYING-CHILD AGE"] = (
            format_ctc_diagnostic_table(tables["by_child_age"])
        )
    return outputs


def build_canonical_ctc_reform_summary(
    baseline_sim,
    reformed_sim,
    *,
    period: int = 2025,
) -> pd.DataFrame:
    rows = []
    for variable in CANONICAL_CTC_REFORM_VARIABLES:
        baseline = float(baseline_sim.calculate(variable, period=period).sum())
        reformed = float(reformed_sim.calculate(variable, period=period).sum())
        rows.append(
            {
                "variable": variable,
                "baseline": baseline,
                "reformed": reformed,
                "delta": reformed - baseline,
            }
        )
    return pd.DataFrame(rows)


def _format_canonical_ctc_reform_summary(table: pd.DataFrame) -> str:
    display = table.copy()
    numeric_columns = [
        column
        for column in display.columns
        if column != "variable" and pd.api.types.is_numeric_dtype(display[column])
    ]
    for column in numeric_columns:
        display[column] = display[column].map(lambda value: f"${value / 1e9:,.1f}B")
    return display.to_string(index=False)


def build_artifact_ctc_summary(
    reference_sim,
    candidate_sim,
    *,
    period: int = 2025,
) -> pd.DataFrame:
    rows = []
    for variable in ARTIFACT_CTC_SUMMARY_VARIABLES:
        reference = float(reference_sim.calculate(variable, period=period).sum())
        candidate = float(candidate_sim.calculate(variable, period=period).sum())
        rows.append(
            {
                "variable": variable,
                "reference": reference,
                "candidate": candidate,
                "delta": candidate - reference,
            }
        )
    return pd.DataFrame(rows)


def _format_artifact_ctc_summary(table: pd.DataFrame) -> str:
    display = table.copy()
    for column in ("reference", "candidate", "delta"):
        display[column] = display.apply(
            lambda row: (
                f"{row[column] / 1e6:,.2f}M"
                if row["variable"] in COUNT_VARS
                else f"${row[column] / 1e9:,.1f}B"
            ),
            axis=1,
        )
    return display.to_string(index=False)


def get_artifact_ctc_comparison_outputs(
    reference_sim,
    candidate_sim,
    *,
    period: int = 2025,
) -> dict[str, str]:
    outputs = {
        "CURRENT-LAW CTC TOTAL DELTAS VS COMPARISON DATASET": (
            _format_artifact_ctc_summary(
                build_artifact_ctc_summary(
                    reference_sim,
                    candidate_sim,
                    period=period,
                )
            )
        )
    }

    delta_tables = _subtract_diagnostic_tables(
        create_ctc_diagnostic_tables(reference_sim, period=period),
        create_ctc_diagnostic_tables(candidate_sim, period=period),
    )
    section_names = {
        "by_agi_band": "CURRENT-LAW CTC DIAGNOSTIC DELTAS BY AGI BAND",
        "by_filing_status": "CURRENT-LAW CTC DIAGNOSTIC DELTAS BY FILING STATUS",
        "by_agi_band_and_filing_status": (
            "CURRENT-LAW CTC DIAGNOSTIC DELTAS BY AGI BAND AND FILING STATUS"
        ),
        "by_child_count": "CURRENT-LAW CTC DIAGNOSTIC DELTAS BY QUALIFYING-CHILD COUNT",
        "by_child_age": "CURRENT-LAW CTC DIAGNOSTIC DELTAS BY QUALIFYING-CHILD AGE",
    }
    for name, table in delta_tables.items():
        if name in section_names:
            outputs[section_names[name]] = format_ctc_diagnostic_table(table)

    return outputs


def _build_canonical_ctc_reform_comparison_summary(
    reference_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
) -> pd.DataFrame:
    merged = reference_summary.merge(
        candidate_summary,
        on="variable",
        suffixes=("_reference", "_candidate"),
    )
    comparison = pd.DataFrame(
        {
            "variable": merged["variable"],
            "reference_baseline": merged["baseline_reference"],
            "candidate_baseline": merged["baseline_candidate"],
            "baseline_delta": (
                merged["baseline_candidate"] - merged["baseline_reference"]
            ),
            "reference_reformed": merged["reformed_reference"],
            "candidate_reformed": merged["reformed_candidate"],
            "reformed_delta": (
                merged["reformed_candidate"] - merged["reformed_reference"]
            ),
            "reference_delta": merged["delta_reference"],
            "candidate_delta": merged["delta_candidate"],
            "delta_drift": merged["delta_candidate"] - merged["delta_reference"],
        }
    )
    return comparison


def get_canonical_ctc_reform_comparison_outputs(
    reference_dataset_path: str | None = None,
    candidate_dataset_path: str | None = None,
    *,
    reference_baseline_sim=None,
    candidate_baseline_sim=None,
    reference_reformed_sim=None,
    candidate_reformed_sim=None,
    period: int = 2025,
) -> dict[str, str]:
    from policyengine_us import Microsimulation

    if reference_baseline_sim is None:
        if reference_dataset_path is None:
            raise ValueError(
                "reference_dataset_path is required when reference_baseline_sim is not provided"
            )
        reference_baseline_sim = Microsimulation(dataset=reference_dataset_path)
    if candidate_baseline_sim is None:
        if candidate_dataset_path is None:
            raise ValueError(
                "candidate_dataset_path is required when candidate_baseline_sim is not provided"
            )
        candidate_baseline_sim = Microsimulation(dataset=candidate_dataset_path)

    canonical_reform = _create_canonical_ctc_reform()
    if reference_reformed_sim is None:
        if reference_dataset_path is None:
            raise ValueError(
                "reference_dataset_path is required when reference_reformed_sim is not provided"
            )
        reference_reformed_sim = Microsimulation(
            dataset=reference_dataset_path,
            reform=canonical_reform,
        )
    if candidate_reformed_sim is None:
        if candidate_dataset_path is None:
            raise ValueError(
                "candidate_dataset_path is required when candidate_reformed_sim is not provided"
            )
        candidate_reformed_sim = Microsimulation(
            dataset=candidate_dataset_path,
            reform=canonical_reform,
        )

    comparison = _build_canonical_ctc_reform_comparison_summary(
        build_canonical_ctc_reform_summary(
            reference_baseline_sim,
            reference_reformed_sim,
            period=period,
        ),
        build_canonical_ctc_reform_summary(
            candidate_baseline_sim,
            candidate_reformed_sim,
            period=period,
        ),
    )
    return {
        "CANONICAL CTC REFORM DRIFT VS COMPARISON DATASET": (
            _format_canonical_ctc_reform_summary(comparison)
        )
    }


def _subtract_diagnostic_tables(
    baseline_tables: dict[str, pd.DataFrame],
    reformed_tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    delta_tables = {}
    for name, baseline in baseline_tables.items():
        if name not in reformed_tables:
            continue
        reformed = reformed_tables[name]
        numeric_columns = [
            column
            for column in baseline.columns
            if column in reformed.columns
            and pd.api.types.is_numeric_dtype(baseline[column])
            and pd.api.types.is_numeric_dtype(reformed[column])
        ]
        id_columns = [
            column
            for column in baseline.columns
            if column in reformed.columns and column not in numeric_columns
        ]
        merged = baseline.merge(
            reformed,
            on=id_columns,
            suffixes=("_baseline", "_reformed"),
        )
        delta = merged[id_columns].copy()
        for column in numeric_columns:
            delta[column] = merged[f"{column}_reformed"] - merged[f"{column}_baseline"]
        delta_tables[name] = delta
    return delta_tables


def _create_canonical_ctc_reform():
    from policyengine_core.reforms import Reform

    return Reform.from_dict(CANONICAL_CTC_REFORM_DICT, country_id="us")


def get_canonical_ctc_reform_outputs(
    dataset_path: str,
    *,
    baseline_sim=None,
    period: int = 2025,
) -> dict[str, str]:
    from policyengine_us import Microsimulation

    if baseline_sim is None:
        baseline_sim = Microsimulation(dataset=dataset_path)

    reformed_sim = Microsimulation(
        dataset=dataset_path,
        reform=_create_canonical_ctc_reform(),
    )

    outputs = {
        "CANONICAL CTC REFORM NATIONAL DELTAS": _format_canonical_ctc_reform_summary(
            build_canonical_ctc_reform_summary(
                baseline_sim,
                reformed_sim,
                period=period,
            )
        )
    }

    delta_tables = _subtract_diagnostic_tables(
        create_ctc_diagnostic_tables(baseline_sim, period=period),
        create_ctc_diagnostic_tables(reformed_sim, period=period),
    )
    section_names = {
        "by_agi_band": "CANONICAL CTC REFORM DELTAS BY AGI BAND",
        "by_filing_status": "CANONICAL CTC REFORM DELTAS BY FILING STATUS",
        "by_agi_band_and_filing_status": (
            "CANONICAL CTC REFORM DELTAS BY AGI BAND AND FILING STATUS"
        ),
        "by_child_count": "CANONICAL CTC REFORM DELTAS BY QUALIFYING-CHILD COUNT",
        "by_child_age": "CANONICAL CTC REFORM DELTAS BY QUALIFYING-CHILD AGE",
    }
    for name, table in delta_tables.items():
        if name in section_names:
            outputs[section_names[name]] = format_ctc_diagnostic_table(table)

    return outputs


def resolve_dataset_path(dataset_path: str) -> str:
    """Resolve Hugging Face dataset URIs to a local H5 path."""
    if not dataset_path.startswith("hf://"):
        return dataset_path

    from huggingface_hub import hf_hub_download

    parts = dataset_path[5:].split("/", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected hf:// dataset path: {dataset_path}")

    return hf_hub_download(
        repo_id=f"{parts[0]}/{parts[1]}",
        filename=parts[2],
        repo_type="model",
        token=os.environ.get("HUGGING_FACE_TOKEN"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate national US.h5")
    parser.add_argument(
        "--h5-path",
        default=None,
        help="Local path to US.h5",
    )
    parser.add_argument(
        "--hf-path",
        default=DEFAULT_HF_PATH,
        help=f"HF path to US.h5 (default: {DEFAULT_HF_PATH})",
    )
    parser.add_argument(
        "--compare-h5-path",
        default=None,
        help="Optional local path to comparison US.h5",
    )
    parser.add_argument(
        "--compare-hf-path",
        default=None,
        help="Optional HF path to comparison US.h5",
    )
    args = parser.parse_args(argv)

    dataset_path = args.h5_path or args.hf_path
    resolved_dataset_path = resolve_dataset_path(dataset_path)
    comparison_dataset_path = args.compare_h5_path or args.compare_hf_path
    resolved_comparison_dataset_path = (
        resolve_dataset_path(comparison_dataset_path)
        if comparison_dataset_path is not None
        else None
    )

    from policyengine_us import Microsimulation

    print(f"Loading {dataset_path}...")
    sim = Microsimulation(dataset=resolved_dataset_path)
    comparison_sim = None
    if resolved_comparison_dataset_path is not None:
        print(f"Loading comparison dataset {comparison_dataset_path}...")
        comparison_sim = Microsimulation(dataset=resolved_comparison_dataset_path)

    n_hh = sim.calculate("household_id", map_to="household").shape[0]
    print(f"Households in file: {n_hh:,}")

    print("\n" + "=" * 70)
    print("NATIONAL H5 VALUES")
    print("=" * 70)

    values = {}
    failures = []
    for var in VARIABLES:
        try:
            val = float(sim.calculate(var).sum())
            values[var] = val
            if var in COUNT_VARS:
                print(f"  {var:45s} {val:>15,.0f}")
            else:
                print(f"  {var:45s} ${val:>15,.0f}")
        except Exception as e:
            failures.append((var, str(e)))
            print(f"  {var:45s} FAILED: {e}")

    print("\n" + "=" * 70)
    print("COMPARISON TO REFERENCE VALUES")
    print("=" * 70)

    any_flag = False
    for var, (ref_val, ref_label) in get_reference_values().items():
        if var not in values:
            continue
        val = values[var]
        pct_diff = (val - ref_val) / ref_val * 100
        flag = " ***" if abs(pct_diff) > 30 else ""
        if flag:
            any_flag = True
        if var in COUNT_VARS:
            print(
                f"  {var:35s} {val:>15,.0f}  "
                f"ref {ref_label:>8s}  "
                f"({pct_diff:+.1f}%){flag}"
            )
        else:
            print(
                f"  {var:35s} ${val:>15,.0f}  "
                f"ref {ref_label:>8s}  "
                f"({pct_diff:+.1f}%){flag}"
            )

    if any_flag:
        print("\n*** = >30% deviation from reference. Investigate further.")

    if failures:
        print(f"\n{len(failures)} variables failed:")
        for var, err in failures:
            print(f"  {var}: {err}")

    for section_name, section_output in get_ctc_diagnostic_outputs(sim).items():
        print("\n" + "=" * 70)
        print(section_name)
        print("=" * 70)
        print(section_output)

    for section_name, section_output in get_external_ctc_benchmark_outputs(sim).items():
        print("\n" + "=" * 70)
        print(section_name)
        print("=" * 70)
        print(section_output)

    for section_name, section_output in get_canonical_ctc_reform_outputs(
        resolved_dataset_path,
        baseline_sim=sim,
    ).items():
        print("\n" + "=" * 70)
        print(section_name)
        print("=" * 70)
        print(section_output)

    if comparison_sim is not None:
        for section_name, section_output in get_artifact_ctc_comparison_outputs(
            comparison_sim,
            sim,
        ).items():
            print("\n" + "=" * 70)
            print(section_name)
            print("=" * 70)
            print(section_output)

        for section_name, section_output in get_canonical_ctc_reform_comparison_outputs(
            reference_dataset_path=resolved_comparison_dataset_path,
            candidate_dataset_path=resolved_dataset_path,
            reference_baseline_sim=comparison_sim,
            candidate_baseline_sim=sim,
        ).items():
            print("\n" + "=" * 70)
            print(section_name)
            print("=" * 70)
            print(section_output)

    print("\n" + "=" * 70)
    print("STRUCTURAL CHECKS")
    print("=" * 70)

    from policyengine_us_data.calibration.sanity_checks import (
        run_sanity_checks,
    )

    results = run_sanity_checks(resolved_dataset_path)
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    for r in results:
        icon = (
            "PASS"
            if r["status"] == "PASS"
            else "FAIL"
            if r["status"] == "FAIL"
            else "WARN"
        )
        print(f"  [{icon}] {r['check']}: {r['detail']}")

    print(f"\n{n_pass}/{len(results)} passed, {n_fail} failed")

    return 0 if n_fail == 0 and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
