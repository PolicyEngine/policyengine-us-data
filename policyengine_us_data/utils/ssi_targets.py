"""Shared SSI calibration targets."""

from datetime import date, timedelta

SSI_CBO_TARGET_SOURCE = (
    "https://www.cbo.gov/system/files/2026-02/51313-2026-02-ssi.xlsx"
)
SSI_PAYMENT_TIMING_SOURCE = "https://www.ssa.gov/oact/ssir/SSI24/IV_C_Payments.html"
SSI_PAYMENT_RULE_SOURCE = "https://www.ssa.gov/OP_Home/cfr20/416/416-0502.htm"
SSI_PAYMENT_TARGET_SOURCE = (
    f"{SSI_CBO_TARGET_SOURCE}; {SSI_PAYMENT_TIMING_SOURCE}; {SSI_PAYMENT_RULE_SOURCE}"
)


def _as_fiscal_year(year) -> int:
    return int(str(year)[:4])


def _is_new_years_day_observed(day: date) -> bool:
    new_years_day = date(day.year, 1, 1)
    next_new_years_day = date(day.year + 1, 1, 1)
    return (
        day == new_years_day
        or (new_years_day.weekday() == 6 and day == date(day.year, 1, 2))
        or (next_new_years_day.weekday() == 5 and day == date(day.year, 12, 31))
    )


def _is_labor_day(day: date) -> bool:
    return day.month == 9 and day.weekday() == 0 and day.day <= 7


def _is_federal_holiday_affecting_ssi_payment(day: date) -> bool:
    return _is_new_years_day_observed(day) or _is_labor_day(day)


def _ssi_payment_date(year: int, month: int) -> date:
    payment_date = date(year, month, 1)
    while payment_date.weekday() >= 5 or _is_federal_holiday_affecting_ssi_payment(
        payment_date
    ):
        payment_date -= timedelta(days=1)
    return payment_date


def _ssi_fiscal_year_benefit_months(year) -> list[date]:
    fiscal_year = _as_fiscal_year(year)
    fiscal_year_start = date(fiscal_year - 1, 10, 1)
    fiscal_year_end = date(fiscal_year, 9, 30)

    benefit_months = []
    for calendar_year in (fiscal_year - 1, fiscal_year):
        for month in range(1, 13):
            payment_day = _ssi_payment_date(calendar_year, month)
            if fiscal_year_start <= payment_day <= fiscal_year_end:
                benefit_months.append(date(calendar_year, month, 1))
    return benefit_months


def get_ssi_fiscal_year_payment_count(year) -> int:
    """Return SSI benefit months with payment dates in the federal fiscal year."""
    return len(_ssi_fiscal_year_benefit_months(year))


def get_ssi_single_year_available_payment_count(year) -> int:
    """Return fiscal-year SSI benefit months available from a single-year H5."""
    fiscal_year = _as_fiscal_year(year)
    return sum(
        benefit_month.year == fiscal_year
        for benefit_month in _ssi_fiscal_year_benefit_months(year)
    )


def scale_ssi_fiscal_year_target_for_single_year_data(value, year) -> float:
    """Scale full fiscal-year SSI outlays to months computable from one H5 year."""
    return (
        float(value)
        * get_ssi_single_year_available_payment_count(year)
        / get_ssi_fiscal_year_payment_count(year)
    )


def get_ssi_payment_target_notes(year) -> str:
    fiscal_year = _as_fiscal_year(year)
    available_count = get_ssi_single_year_available_payment_count(year)
    payment_count = get_ssi_fiscal_year_payment_count(year)
    return (
        "CBO SSI federal fiscal-year outlays scaled to the benefit months "
        "computable from a single-year PolicyEngine-US-data H5 using "
        "policyengine-us ssi_federal_fiscal_year_outlays; "
        f"FY{fiscal_year} has {payment_count} SSI benefit months paid in the "
        f"federal fiscal year, of which {available_count} are benefit months "
        f"in calendar year {fiscal_year}"
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
