"""Shared SSI calibration targets."""

SSI_CBO_TARGET_SOURCE = (
    "https://www.cbo.gov/system/files/2026-02/51313-2026-02-ssi.xlsx"
)
SSI_ANNUAL_PAYMENT_TARGET_SOURCE = (
    "https://www.ssa.gov/policy/docs/statcomps/ssi_asr/2024/sect01.html"
)
SSI_OACT_PAYMENT_DATE_TARGET_SOURCE = (
    "https://www.ssa.gov/oact/ssir/SSI25/IV_C_Payments.html"
)
SSI_OACT_CY2024_PAYMENT_DATE_ALL = 63_080_000_000
SSI_OACT_FY2024_PAYMENT_DATE_ALL = 57_600_000_000
SSI_ANNUAL_PAYMENT_TARGET_NOTES = (
    "SSA SSI Annual Statistical Report, 2024, Table 2; Federal SSI total "
    "annual payments for all recipients, excluding federally administered "
    "state supplementation. ASR allocates payments to the month due, so this "
    "target aligns with annual `ssi` over January-December benefit months. "
    "Do not replace it with OACT payment-date accounting; OACT Table IV.C2 "
    "reports FY2024 all Federal SSI payments of $57.600B because fiscal-year "
    "payment-date totals can contain 11, 12, or 13 monthly payments. The "
    "smaller gap between ASR CY2024 and OACT FY2024 is not a pure 12-vs-11 "
    "month comparison: OACT Table IV.C1 reports CY2024 all Federal SSI "
    "payments of $63.080B on a payment-date basis, $5.480B above OACT "
    "FY2024, and OACT obligations are not reduced for certain recovered "
    "overpayments remitted directly to Treasury that ASR nets out."
)
SSI_ANNUAL_PAYMENT_TARGETS = {
    2024: {
        "value": 59_665_127_000,
        "source": SSI_ANNUAL_PAYMENT_TARGET_SOURCE,
        "notes": SSI_ANNUAL_PAYMENT_TARGET_NOTES,
    },
}


def get_ssi_annual_payment_target(year) -> dict | None:
    return SSI_ANNUAL_PAYMENT_TARGETS.get(int(str(year)[:4]))


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
