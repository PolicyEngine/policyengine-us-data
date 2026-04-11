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

CHILD_AGE_GROUP_COLUMNS = [
    "tax_unit_count",
    "ctc_qualifying_children",
    "ctc_recipient_count",
    "refundable_ctc_recipient_count",
    "non_refundable_ctc_recipient_count",
]

COUNT_FORMAT_COLUMNS = {
    "tax_unit_count",
    "ctc_qualifying_children",
    "ctc_recipient_count",
    "refundable_ctc_recipient_count",
    "non_refundable_ctc_recipient_count",
}

AMOUNT_FORMAT_COLUMNS = {
    "ctc",
    "refundable_ctc",
    "non_refundable_ctc",
}


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


def _assign_ctc_child_count_buckets(
    ctc_qualifying_children: np.ndarray,
) -> pd.Categorical:
    labels = ["0", "1", "2", "3+"]
    bucket = np.full(len(ctc_qualifying_children), labels[-1], dtype=object)
    bucket[ctc_qualifying_children <= 0] = "0"
    bucket[ctc_qualifying_children == 1] = "1"
    bucket[ctc_qualifying_children == 2] = "2"
    return pd.Categorical(bucket, categories=labels, ordered=True)


def _add_weighted_ctc_columns(work: pd.DataFrame) -> pd.DataFrame:
    weights = work["tax_unit_weight"].astype(float).to_numpy()

    work["tax_unit_count"] = weights
    work["ctc_qualifying_children"] = (
        work["ctc_qualifying_children"].astype(float).to_numpy() * weights
    )
    work["ctc_recipient_count"] = (work["ctc"].astype(float).to_numpy() > 0).astype(
        float
    ) * weights
    work["refundable_ctc_recipient_count"] = (
        work["refundable_ctc"].astype(float).to_numpy() > 0
    ).astype(float) * weights
    work["non_refundable_ctc_recipient_count"] = (
        work["non_refundable_ctc"].astype(float).to_numpy() > 0
    ).astype(float) * weights
    work["ctc"] = work["ctc"].astype(float).to_numpy() * weights
    work["refundable_ctc"] = work["refundable_ctc"].astype(float).to_numpy() * weights
    work["non_refundable_ctc"] = (
        work["non_refundable_ctc"].astype(float).to_numpy() * weights
    )

    return work


def _build_child_age_table(work: pd.DataFrame) -> pd.DataFrame | None:
    if (
        "ctc_qualifying_children_under_6" not in work
        or "ctc_qualifying_children_6_to_17" not in work
    ):
        return None

    weights = work["tax_unit_weight"].astype(float).to_numpy()
    ctc_positive = work["ctc"].astype(float).to_numpy() > 0
    refundable_positive = work["refundable_ctc"].astype(float).to_numpy() > 0
    non_refundable_positive = work["non_refundable_ctc"].astype(float).to_numpy() > 0

    rows = []
    for label, child_counts in (
        (
            "Under 6",
            work["ctc_qualifying_children_under_6"].astype(float).to_numpy(),
        ),
        (
            "Age 6-17",
            work["ctc_qualifying_children_6_to_17"].astype(float).to_numpy(),
        ),
    ):
        has_children = child_counts > 0
        rows.append(
            {
                "group": label,
                "tax_unit_count": float((has_children.astype(float) * weights).sum()),
                "ctc_qualifying_children": float((child_counts * weights).sum()),
                "ctc_recipient_count": float(
                    ((ctc_positive & has_children).astype(float) * weights).sum()
                ),
                "refundable_ctc_recipient_count": float(
                    ((refundable_positive & has_children).astype(float) * weights).sum()
                ),
                "non_refundable_ctc_recipient_count": float(
                    (
                        (non_refundable_positive & has_children).astype(float) * weights
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(rows, columns=["group"] + CHILD_AGE_GROUP_COLUMNS)


def build_ctc_diagnostic_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate weighted CTC diagnostics by AGI band and filing status."""
    work = frame.copy()
    child_counts = work["ctc_qualifying_children"].astype(float).to_numpy()

    work["agi_band"] = _assign_agi_bands(
        work["adjusted_gross_income"].astype(float).to_numpy()
    )
    work["filing_status_group"] = _normalize_filing_status(work["filing_status"])
    work["child_count_group"] = _assign_ctc_child_count_buckets(child_counts)
    work = _add_weighted_ctc_columns(work)

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
    by_agi_band_and_filing_status = (
        work.groupby(["agi_band", "filing_status_group"], observed=False)[
            CTC_GROUP_COLUMNS
        ]
        .sum()
        .reset_index()
        .rename(columns={"filing_status_group": "filing_status"})
    )
    by_child_count = (
        work.groupby("child_count_group", observed=False)[CTC_GROUP_COLUMNS]
        .sum()
        .reset_index()
        .rename(columns={"child_count_group": "group"})
    )
    by_child_age = _build_child_age_table(frame)

    tables = {
        "by_agi_band": by_agi,
        "by_filing_status": by_filing_status,
        "by_agi_band_and_filing_status": by_agi_band_and_filing_status,
        "by_child_count": by_child_count,
    }
    if by_child_age is not None:
        tables["by_child_age"] = by_child_age
    return tables


def create_ctc_diagnostic_tables(sim, period=None) -> dict[str, pd.DataFrame]:
    """Calculate weighted CTC diagnostic tables from a microsimulation."""
    frame = pd.DataFrame(
        {
            "adjusted_gross_income": sim.calculate(
                "adjusted_gross_income", period=period
            ).values,
            "filing_status": sim.calculate("filing_status", period=period).values,
            "tax_unit_weight": sim.calculate("tax_unit_weight", period=period).values,
            "ctc_qualifying_children": sim.calculate(
                "ctc_qualifying_children", period=period
            ).values,
            "ctc": sim.calculate("ctc", period=period).values,
            "refundable_ctc": sim.calculate("refundable_ctc", period=period).values,
            "non_refundable_ctc": sim.calculate(
                "non_refundable_ctc", period=period
            ).values,
        }
    )

    try:
        ctc_qualifying_child = sim.calculate(
            "ctc_qualifying_child",
            map_to="person",
            period=period,
        ).values.astype(bool)
        age = sim.calculate("age", map_to="person", period=period).values.astype(float)
        frame["ctc_qualifying_children_under_6"] = sim.map_result(
            (ctc_qualifying_child & (age < 6)).astype(float),
            "person",
            "tax_unit",
        )
        frame["ctc_qualifying_children_6_to_17"] = sim.map_result(
            (ctc_qualifying_child & (age >= 6) & (age < 18)).astype(float),
            "person",
            "tax_unit",
        )
    except Exception:
        pass

    return build_ctc_diagnostic_tables(frame)


def _format_count(value: float) -> str:
    return f"{value / 1e6:,.2f}M"


def _format_amount(value: float) -> str:
    return f"${value / 1e9:,.1f}B"


def format_ctc_diagnostic_table(table: pd.DataFrame) -> str:
    display = table.copy()
    for column in display.columns:
        if column in COUNT_FORMAT_COLUMNS:
            display[column] = display[column].map(_format_count)
        elif column in AMOUNT_FORMAT_COLUMNS:
            display[column] = display[column].map(_format_amount)
    return display.to_string(index=False)
