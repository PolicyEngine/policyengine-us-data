MEDICARE_PART_B_GROSS_PREMIUM_INCOME = {
    2024: 139.837e9,
}


MEDICARE_STATE_BUY_IN_MINIMUM_BENEFICIARIES = {
    2024: 10_000_000,
}


BENEFICIARY_PAID_MEDICARE_PART_B_PREMIUM_TARGETS = {
    2024: 112e9,
}


def get_beneficiary_paid_medicare_part_b_premiums_target(year: int) -> float:
    try:
        return BENEFICIARY_PAID_MEDICARE_PART_B_PREMIUM_TARGETS[year]
    except KeyError as exc:
        raise ValueError(
            f"No beneficiary-paid Medicare Part B premium target sourced for {year}."
        ) from exc


def get_beneficiary_paid_medicare_part_b_premiums_source(year: int) -> str:
    gross_income = MEDICARE_PART_B_GROSS_PREMIUM_INCOME[year] / 1e9
    minimum_buy_in = MEDICARE_STATE_BUY_IN_MINIMUM_BENEFICIARIES[year]
    return (
        "CMS 2025 Medicare Trustees Report Table III.C3 actual 2024 Part B "
        f"premium income (${gross_income:.3f}B), plus CMS State Buy-In FAQ "
        f"noting states paid Part B premiums for over {minimum_buy_in:,} people"
    )


def get_beneficiary_paid_medicare_part_b_premiums_notes(year: int) -> str:
    return (
        "Approximate beneficiary-paid Medicare Part B out-of-pocket premiums "
        "for SPM/MOOP calibration. This intentionally does not target gross "
        "trust-fund premium income because Medicaid and other MSP pathways pay "
        "premiums on behalf of some enrollees."
    )
