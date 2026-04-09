import numpy as np
import pandas as pd

IRS_AGI_BANDS = [
    (-np.inf, 1.0, "<$1"),
    (1.0, 10_000.0, "$1-$10k"),
    (10_000.0, 25_000.0, "$10k-$25k"),
    (25_000.0, 50_000.0, "$25k-$50k"),
    (50_000.0, 75_000.0, "$50k-$75k"),
    (75_000.0, 100_000.0, "$75k-$100k"),
    (100_000.0, 200_000.0, "$100k-$200k"),
    (200_000.0, 500_000.0, "$200k-$500k"),
    (500_000.0, np.inf, "$500k+"),
]

FILING_STATUS_LABELS = {
    "SINGLE": "Single",
    "HEAD_OF_HOUSEHOLD": "Head of household",
    "JOINT": "Joint / surviving spouse",
    "SURVIVING_SPOUSE": "Joint / surviving spouse",
    "SEPARATE": "Separate",
}

FILING_STATUS_ORDER = [
    "Single",
    "Head of household",
    "Joint / surviving spouse",
    "Separate",
    "Other",
]

CTC_GROUP_COLUMNS = [
    "tax_unit_count",
    "ctc_qualifying_children",
    "ctc_recipient_count",
    "refundable_ctc_recipient_count",
    "non_refundable_ctc_recipient_count",
    "ctc",
    "refundable_ctc",
    "non_refundable_ctc",
]


def _assign_agi_bands(adjusted_gross_income: np.ndarray) -> pd.Categorical:
    labels = [label for _, _, label in IRS_AGI_BANDS]
    agi_band = np.full(len(adjusted_gross_income), labels[-1], dtype=object)
    for lower, upper, label in IRS_AGI_BANDS:
        mask = (adjusted_gross_income >= lower) & (adjusted_gross_income < upper)
        agi_band[mask] = label
    return pd.Categorical(agi_band, categories=labels, ordered=True)


def _normalize_filing_status(filing_status: pd.Series) -> pd.Categorical:
    labels = [
        FILING_STATUS_LABELS.get(str(value), "Other")
        for value in filing_status.astype(str)
    ]
    return pd.Categorical(labels, categories=FILING_STATUS_ORDER, ordered=True)


def build_ctc_diagnostic_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate weighted CTC diagnostics by AGI band and filing status."""
    work = frame.copy()
    weights = work["tax_unit_weight"].astype(float).to_numpy()

    work["agi_band"] = _assign_agi_bands(
        work["adjusted_gross_income"].astype(float).to_numpy()
    )
    work["filing_status_group"] = _normalize_filing_status(work["filing_status"])

    work["tax_unit_count"] = weights
    work["ctc_qualifying_children"] = (
        work["ctc_qualifying_children"].astype(float).to_numpy() * weights
    )
    work["ctc_recipient_count"] = (
        (work["ctc"].astype(float).to_numpy() > 0).astype(float) * weights
    )
    work["refundable_ctc_recipient_count"] = (
        (work["refundable_ctc"].astype(float).to_numpy() > 0).astype(float) * weights
    )
    work["non_refundable_ctc_recipient_count"] = (
        (work["non_refundable_ctc"].astype(float).to_numpy() > 0).astype(float)
        * weights
    )
    work["ctc"] = work["ctc"].astype(float).to_numpy() * weights
    work["refundable_ctc"] = work["refundable_ctc"].astype(float).to_numpy() * weights
    work["non_refundable_ctc"] = (
        work["non_refundable_ctc"].astype(float).to_numpy() * weights
    )

    by_agi = (
        work.groupby("agi_band", observed=False)[CTC_GROUP_COLUMNS]
        .sum()
        .reset_index()
        .rename(columns={"agi_band": "group"})
    )
    by_filing_status = (
        work.groupby("filing_status_group", observed=False)[CTC_GROUP_COLUMNS]
        .sum()
        .reset_index()
        .rename(columns={"filing_status_group": "group"})
    )

    return {
        "by_agi_band": by_agi,
        "by_filing_status": by_filing_status,
    }


def create_ctc_diagnostic_tables(sim) -> dict[str, pd.DataFrame]:
    """Calculate weighted CTC diagnostic tables from a microsimulation."""
    frame = pd.DataFrame(
        {
            "adjusted_gross_income": sim.calculate("adjusted_gross_income").values,
            "filing_status": sim.calculate("filing_status").values,
            "tax_unit_weight": sim.calculate("tax_unit_weight").values,
            "ctc_qualifying_children": sim.calculate("ctc_qualifying_children").values,
            "ctc": sim.calculate("ctc").values,
            "refundable_ctc": sim.calculate("refundable_ctc").values,
            "non_refundable_ctc": sim.calculate("non_refundable_ctc").values,
        }
    )
    return build_ctc_diagnostic_tables(frame)


def _format_count(value: float) -> str:
    return f"{value / 1e6:,.2f}M"


def _format_amount(value: float) -> str:
    return f"${value / 1e9:,.1f}B"


def format_ctc_diagnostic_table(table: pd.DataFrame) -> str:
    display = table.copy()
    for column in [
        "tax_unit_count",
        "ctc_qualifying_children",
        "ctc_recipient_count",
        "refundable_ctc_recipient_count",
        "non_refundable_ctc_recipient_count",
    ]:
        display[column] = display[column].map(_format_count)
    for column in ["ctc", "refundable_ctc", "non_refundable_ctc"]:
        display[column] = display[column].map(_format_amount)
    return display.to_string(index=False)
