"""Shared SSI calibration targets."""

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
