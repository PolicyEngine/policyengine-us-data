"""Shared SSI calibration targets."""

from datetime import date, timedelta


SSI_CBO_TARGET_SOURCE = (
    "https://www.cbo.gov/system/files/2026-02/51313-2026-02-ssi.xlsx"
)
SSI_PAYMENT_TIMING_SOURCE = (
    "https://www.ssa.gov/budget/assets/materials/2026/2026BST.pdf"
)
SSI_PAYMENT_TARGET_SOURCE = f"{SSI_CBO_TARGET_SOURCE}; {SSI_PAYMENT_TIMING_SOURCE}"


def _as_fiscal_year(year) -> int:
    return int(str(year)[:4])


def _is_first_day_federal_holiday(day: date) -> bool:
    is_new_years_day = day.month == 1 and day.day == 1
    is_labor_day = day.month == 9 and day.weekday() == 0 and 1 <= day.day <= 7
    return is_new_years_day or is_labor_day


def _ssi_payment_date(year: int, month: int) -> date:
    day = date(year, month, 1)
    while day.weekday() >= 5 or _is_first_day_federal_holiday(day):
        day -= timedelta(days=1)
    return day


def get_ssi_fiscal_year_payment_count(year) -> int:
    """Return SSI monthly benefit payments counted in the federal fiscal year."""
    fiscal_year = _as_fiscal_year(year)
    start = date(fiscal_year - 1, 10, 1)
    end = date(fiscal_year, 9, 30)
    payment_count = 0

    for calendar_year in (fiscal_year - 1, fiscal_year):
        for month in range(1, 13):
            payment_day = _ssi_payment_date(calendar_year, month)
            if start <= payment_day <= end:
                payment_count += 1

    return payment_count


def normalize_ssi_payment_target(value, year) -> float:
    """Convert fiscal-year SSI outlays to a 12-payment-equivalent target."""
    payment_count = get_ssi_fiscal_year_payment_count(year)
    return float(value) * 12 / payment_count


def get_ssi_payment_target_notes(year) -> str:
    payment_count = get_ssi_fiscal_year_payment_count(year)
    return (
        "CBO SSI total outlays normalized to a 12-payment-equivalent "
        "annual target for PolicyEngine's annual SSI variable; "
        f"FY{_as_fiscal_year(year)} has {payment_count} monthly SSI "
        "payments under federal budget timing"
    )


SSI_RECIPIENT_TARGET_YEAR = 2024
SSI_RECIPIENT_TARGET_SOURCE = (
    "https://www.ssa.gov/policy/docs/statcomps/ssi_monthly/2024-12/table01.html"
)
SSI_RECIPIENT_TARGET_NOTES = (
    "SSI recipients with a federal payment, December 2024, SSA SSI Monthly "
    "Statistics Table 1"
)

SSI_RECIPIENT_TARGETS_2024 = {
    "all": {
        "label": "all",
        "stratum_notes": "National SSI Federal Payment Recipients",
        "person_count": 7_289_843,
        "age_constraints": (),
    },
    "under_18": {
        "label": "under_18",
        "stratum_notes": "National SSI Federal Payment Recipients - Under 18",
        "person_count": 1_001_922,
        "age_constraints": (("<", "18"),),
    },
    "18_64": {
        "label": "18_64",
        "stratum_notes": "National SSI Federal Payment Recipients - Ages 18-64",
        "person_count": 3_905_779,
        "age_constraints": ((">=", "18"), ("<", "65")),
    },
    "65_plus": {
        "label": "65_plus",
        "stratum_notes": "National SSI Federal Payment Recipients - Ages 65+",
        "person_count": 2_382_142,
        "age_constraints": ((">=", "65"),),
    },
}
