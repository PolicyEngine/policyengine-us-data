from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ApproximateCalibrationWindow:
    start_year: int
    end_year: int | None
    max_constraint_error_pct: float
    max_age_error_pct: float
    max_negative_weight_pct: float | None = 0.0
    age_bucket_size: int | None = None
    min_positive_household_count: int | None = None
    min_effective_sample_size: float | None = None
    max_top_10_weight_share_pct: float | None = None
    max_top_100_weight_share_pct: float | None = None

    def applies(self, year: int) -> bool:
        if year < self.start_year:
            return False
        if self.end_year is not None and year > self.end_year:
            return False
        return True


@dataclass(frozen=True)
class CalibrationProfile:
    name: str
    description: str
    calibration_method: str
    use_greg: bool
    use_ss: bool
    use_payroll: bool
    use_h6_reform: bool
    use_tob: bool
    benchmark_tob: bool = False
    allow_greg_fallback: bool = False
    max_constraint_error_pct: float = 0.1
    max_age_error_pct: float = 0.1
    max_negative_weight_pct: float | None = None
    min_positive_household_count: int | None = None
    min_effective_sample_size: float | None = None
    max_top_10_weight_share_pct: float | None = None
    max_top_100_weight_share_pct: float | None = None
    approximate_windows: tuple[ApproximateCalibrationWindow, ...] = field(
        default_factory=tuple
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_LONG_RUN_APPROXIMATE_WINDOWS = (
    ApproximateCalibrationWindow(
        start_year=2075,
        end_year=2078,
        max_constraint_error_pct=0.5,
        max_age_error_pct=0.5,
        max_negative_weight_pct=0.0,
        age_bucket_size=5,
        min_positive_household_count=1000,
        min_effective_sample_size=75.0,
        max_top_10_weight_share_pct=25.0,
        max_top_100_weight_share_pct=95.0,
    ),
    ApproximateCalibrationWindow(
        start_year=2079,
        end_year=2085,
        max_constraint_error_pct=10.0,
        max_age_error_pct=10.0,
        max_negative_weight_pct=0.0,
        age_bucket_size=5,
        min_positive_household_count=1000,
        min_effective_sample_size=75.0,
        max_top_10_weight_share_pct=25.0,
        max_top_100_weight_share_pct=95.0,
    ),
    ApproximateCalibrationWindow(
        start_year=2086,
        end_year=2095,
        max_constraint_error_pct=20.0,
        max_age_error_pct=20.0,
        max_negative_weight_pct=0.0,
        age_bucket_size=5,
        min_positive_household_count=1000,
        min_effective_sample_size=75.0,
        max_top_10_weight_share_pct=25.0,
        max_top_100_weight_share_pct=95.0,
    ),
    ApproximateCalibrationWindow(
        start_year=2096,
        end_year=None,
        max_constraint_error_pct=35.0,
        max_age_error_pct=35.0,
        max_negative_weight_pct=0.0,
        age_bucket_size=5,
        min_positive_household_count=1000,
        min_effective_sample_size=75.0,
        max_top_10_weight_share_pct=25.0,
        max_top_100_weight_share_pct=95.0,
    ),
)


NAMED_PROFILES: dict[str, CalibrationProfile] = {
    "age-only": CalibrationProfile(
        name="age-only",
        description="Age-only calibration using IPF.",
        calibration_method="ipf",
        use_greg=False,
        use_ss=False,
        use_payroll=False,
        use_h6_reform=False,
        use_tob=False,
        benchmark_tob=False,
        allow_greg_fallback=False,
        min_positive_household_count=1000,
        min_effective_sample_size=75.0,
        max_top_10_weight_share_pct=25.0,
        max_top_100_weight_share_pct=95.0,
    ),
    "ss": CalibrationProfile(
        name="ss",
        description="Age plus Social Security benefits using positive entropy calibration.",
        calibration_method="entropy",
        use_greg=False,
        use_ss=True,
        use_payroll=False,
        use_h6_reform=False,
        use_tob=False,
        benchmark_tob=False,
        max_negative_weight_pct=0.0,
        min_positive_household_count=1000,
        min_effective_sample_size=75.0,
        max_top_10_weight_share_pct=25.0,
        max_top_100_weight_share_pct=95.0,
        approximate_windows=DEFAULT_LONG_RUN_APPROXIMATE_WINDOWS,
    ),
    "ss-payroll": CalibrationProfile(
        name="ss-payroll",
        description="Age, Social Security, and taxable payroll using positive entropy calibration.",
        calibration_method="entropy",
        use_greg=False,
        use_ss=True,
        use_payroll=True,
        use_h6_reform=False,
        use_tob=False,
        benchmark_tob=False,
        max_negative_weight_pct=0.0,
        min_positive_household_count=1000,
        min_effective_sample_size=75.0,
        max_top_10_weight_share_pct=25.0,
        max_top_100_weight_share_pct=95.0,
        approximate_windows=DEFAULT_LONG_RUN_APPROXIMATE_WINDOWS,
    ),
    "ss-payroll-tob": CalibrationProfile(
        name="ss-payroll-tob",
        description="Age, Social Security, and taxable payroll using positive entropy calibration, with TOB benchmarked post-calibration against the selected long-term target source.",
        calibration_method="entropy",
        use_greg=False,
        use_ss=True,
        use_payroll=True,
        use_h6_reform=False,
        use_tob=False,
        benchmark_tob=True,
        max_negative_weight_pct=0.0,
        min_positive_household_count=1000,
        min_effective_sample_size=75.0,
        max_top_10_weight_share_pct=25.0,
        max_top_100_weight_share_pct=95.0,
        approximate_windows=DEFAULT_LONG_RUN_APPROXIMATE_WINDOWS,
    ),
    "ss-payroll-tob-h6": CalibrationProfile(
        name="ss-payroll-tob-h6",
        description="Age, Social Security, taxable payroll, and H6 using positive entropy calibration, with TOB benchmarked post-calibration against the selected long-term target source.",
        calibration_method="entropy",
        use_greg=False,
        use_ss=True,
        use_payroll=True,
        use_h6_reform=True,
        use_tob=False,
        benchmark_tob=True,
        max_negative_weight_pct=0.0,
        min_positive_household_count=1000,
        min_effective_sample_size=75.0,
        max_top_10_weight_share_pct=25.0,
        max_top_100_weight_share_pct=95.0,
        approximate_windows=DEFAULT_LONG_RUN_APPROXIMATE_WINDOWS,
    ),
}

QUALITY_RANK = {
    "aggregate": 0,
    "approximate": 1,
    "exact": 2,
}


def get_profile(name: str) -> CalibrationProfile:
    try:
        return NAMED_PROFILES[name]
    except KeyError as error:
        valid = ", ".join(sorted(NAMED_PROFILES))
        raise ValueError(f"Unknown calibration profile '{name}'. Valid profiles: {valid}") from error


def approximate_window_for_year(
    profile: CalibrationProfile,
    year: int | None,
) -> ApproximateCalibrationWindow | None:
    if not profile.approximate_windows:
        return None

    if year is None:
        return max(
            profile.approximate_windows,
            key=lambda window: (
                float("inf") if window.end_year is None else window.end_year,
                window.max_constraint_error_pct,
                window.max_age_error_pct,
            ),
        )

    for window in profile.approximate_windows:
        if window.applies(year):
            return window
    return None


def build_profile_from_flags(
    *,
    use_greg: bool,
    use_ss: bool,
    use_payroll: bool,
    use_h6_reform: bool,
    use_tob: bool,
) -> CalibrationProfile:
    if use_tob and not use_greg:
        use_greg = True

    if not use_greg:
        for profile in NAMED_PROFILES.values():
            if (
                profile.use_greg is False
                and profile.use_ss == use_ss
                and profile.use_payroll == use_payroll
                and profile.use_h6_reform == use_h6_reform
                and profile.use_tob == use_tob
            ):
                return profile

    for profile in NAMED_PROFILES.values():
        if (
            profile.calibration_method == ("greg" if use_greg else "ipf")
            and profile.use_greg == use_greg
            and profile.use_ss == use_ss
            and profile.use_payroll == use_payroll
            and profile.use_h6_reform == use_h6_reform
            and profile.use_tob == use_tob
        ):
            return profile

    flag_names = []
    if use_greg:
        flag_names.append("greg")
    if use_ss:
        flag_names.append("ss")
    if use_payroll:
        flag_names.append("payroll")
    if use_h6_reform:
        flag_names.append("h6")
    if use_tob:
        flag_names.append("tob")

    suffix = "-".join(flag_names) if flag_names else "age-only"
    return CalibrationProfile(
        name=f"custom-{suffix}",
        description="Legacy flag-derived calibration profile.",
        calibration_method="greg" if use_greg else "ipf",
        use_greg=use_greg,
        use_ss=use_ss,
        use_payroll=use_payroll,
        use_h6_reform=use_h6_reform,
        use_tob=use_tob,
        benchmark_tob=False,
    )


def validate_calibration_audit(
    audit: dict[str, Any],
    profile: CalibrationProfile,
    *,
    year: int | None = None,
    quality: str | None = None,
) -> list[str]:
    if quality is None:
        quality = audit.get("calibration_quality") or classify_calibration_quality(
            audit,
            profile,
            year=year,
        )

    if quality == "exact":
        window = approximate_window_for_year(profile, year)
        return _collect_threshold_issues(
            audit,
            profile,
            max_constraint_error_pct=profile.max_constraint_error_pct,
            max_age_error_pct=profile.max_age_error_pct,
            max_negative_weight_pct=profile.max_negative_weight_pct,
            min_positive_household_count=(
                window.min_positive_household_count
                if window is not None
                else profile.min_positive_household_count
            ),
            min_effective_sample_size=(
                window.min_effective_sample_size
                if window is not None
                else profile.min_effective_sample_size
            ),
            max_top_10_weight_share_pct=(
                window.max_top_10_weight_share_pct
                if window is not None
                else profile.max_top_10_weight_share_pct
            ),
            max_top_100_weight_share_pct=(
                window.max_top_100_weight_share_pct
                if window is not None
                else profile.max_top_100_weight_share_pct
            ),
        )

    if quality == "approximate":
        window = approximate_window_for_year(profile, year)
        if window is None:
            issues = _collect_threshold_issues(
                audit,
                profile,
                max_constraint_error_pct=profile.max_constraint_error_pct,
                max_age_error_pct=profile.max_age_error_pct,
                max_negative_weight_pct=profile.max_negative_weight_pct,
                min_positive_household_count=profile.min_positive_household_count,
                min_effective_sample_size=profile.min_effective_sample_size,
                max_top_10_weight_share_pct=profile.max_top_10_weight_share_pct,
                max_top_100_weight_share_pct=profile.max_top_100_weight_share_pct,
            )
            issues.append(
                "Approximate calibration is not permitted for this profile/year"
            )
            return issues
        return _collect_threshold_issues(
            audit,
            profile,
            max_constraint_error_pct=window.max_constraint_error_pct,
            max_age_error_pct=window.max_age_error_pct,
            max_negative_weight_pct=window.max_negative_weight_pct,
            min_positive_household_count=window.min_positive_household_count,
            min_effective_sample_size=window.min_effective_sample_size,
            max_top_10_weight_share_pct=window.max_top_10_weight_share_pct,
            max_top_100_weight_share_pct=window.max_top_100_weight_share_pct,
        )

    exact_issues = _collect_threshold_issues(
        audit,
        profile,
        max_constraint_error_pct=profile.max_constraint_error_pct,
        max_age_error_pct=profile.max_age_error_pct,
        max_negative_weight_pct=profile.max_negative_weight_pct,
        min_positive_household_count=profile.min_positive_household_count,
        min_effective_sample_size=profile.min_effective_sample_size,
        max_top_10_weight_share_pct=profile.max_top_10_weight_share_pct,
        max_top_100_weight_share_pct=profile.max_top_100_weight_share_pct,
    )
    window = approximate_window_for_year(profile, year)
    if window is None:
        return exact_issues + [
            "Calibration quality aggregate exceeds approximate thresholds"
        ]
    approximate_issues = _collect_threshold_issues(
        audit,
        profile,
        max_constraint_error_pct=window.max_constraint_error_pct,
        max_age_error_pct=window.max_age_error_pct,
        max_negative_weight_pct=window.max_negative_weight_pct,
        min_positive_household_count=window.min_positive_household_count,
        min_effective_sample_size=window.min_effective_sample_size,
        max_top_10_weight_share_pct=window.max_top_10_weight_share_pct,
        max_top_100_weight_share_pct=window.max_top_100_weight_share_pct,
    )
    return approximate_issues + [
        "Calibration quality aggregate exceeds approximate thresholds"
    ]


def _collect_threshold_issues(
    audit: dict[str, Any],
    profile: CalibrationProfile,
    *,
    max_constraint_error_pct: float | None,
    max_age_error_pct: float | None,
    max_negative_weight_pct: float | None,
    min_positive_household_count: int | None,
    min_effective_sample_size: float | None,
    max_top_10_weight_share_pct: float | None,
    max_top_100_weight_share_pct: float | None,
) -> list[str]:
    issues: list[str] = []

    if profile.calibration_method == "greg" and audit.get("fell_back_to_ipf"):
        issues.append("GREG calibration fell back to IPF")

    age_error = audit.get("age_max_pct_error")
    if (
        max_age_error_pct is not None
        and age_error is not None
        and age_error > max_age_error_pct
    ):
        issues.append(
            f"Age max error {age_error:.3f}% exceeds {max_age_error_pct:.3f}%"
        )

    for constraint_name, stats in audit.get("constraints", {}).items():
        pct_error = stats.get("pct_error")
        if (
            max_constraint_error_pct is not None
            and pct_error is not None
            and abs(pct_error) > max_constraint_error_pct
        ):
            issues.append(
                f"{constraint_name} error {pct_error:.3f}% exceeds "
                f"{max_constraint_error_pct:.3f}%"
            )

    if max_negative_weight_pct is not None:
        pct = audit.get("negative_weight_pct")
        if pct is not None and pct > max_negative_weight_pct:
            issues.append(
                f"Negative weight share {pct:.3f}% exceeds "
                f"{max_negative_weight_pct:.3f}%"
            )

    positive_count = audit.get("positive_weight_count")
    if (
        min_positive_household_count is not None
        and positive_count is not None
        and positive_count < min_positive_household_count
    ):
        issues.append(
            f"Positive household count {positive_count} is below "
            f"{min_positive_household_count}"
        )

    ess = audit.get("effective_sample_size")
    if (
        min_effective_sample_size is not None
        and ess is not None
        and ess < min_effective_sample_size
    ):
        issues.append(
            f"Effective sample size {ess:.3f} is below "
            f"{min_effective_sample_size:.3f}"
        )

    top_10_share = audit.get("top_10_weight_share_pct")
    if (
        max_top_10_weight_share_pct is not None
        and top_10_share is not None
        and top_10_share > max_top_10_weight_share_pct
    ):
        issues.append(
            f"Top-10 weight share {top_10_share:.3f}% exceeds "
            f"{max_top_10_weight_share_pct:.3f}%"
        )

    top_100_share = audit.get("top_100_weight_share_pct")
    if (
        max_top_100_weight_share_pct is not None
        and top_100_share is not None
        and top_100_share > max_top_100_weight_share_pct
    ):
        issues.append(
            f"Top-100 weight share {top_100_share:.3f}% exceeds "
            f"{max_top_100_weight_share_pct:.3f}%"
        )

    return issues


def classify_calibration_quality(
    audit: dict[str, Any],
    profile: CalibrationProfile,
    *,
    year: int | None = None,
) -> str:
    exact_issues = _collect_threshold_issues(
        audit,
        profile,
        max_constraint_error_pct=profile.max_constraint_error_pct,
        max_age_error_pct=profile.max_age_error_pct,
        max_negative_weight_pct=profile.max_negative_weight_pct,
        min_positive_household_count=profile.min_positive_household_count,
        min_effective_sample_size=profile.min_effective_sample_size,
        max_top_10_weight_share_pct=profile.max_top_10_weight_share_pct,
        max_top_100_weight_share_pct=profile.max_top_100_weight_share_pct,
    )
    if not exact_issues:
        return "exact"

    window = approximate_window_for_year(profile, year)
    if window is None:
        if year is not None:
            return "aggregate"
        return "approximate"

    approximate_issues = _collect_threshold_issues(
        audit,
        profile,
        max_constraint_error_pct=window.max_constraint_error_pct,
        max_age_error_pct=window.max_age_error_pct,
        max_negative_weight_pct=window.max_negative_weight_pct,
        min_positive_household_count=window.min_positive_household_count,
        min_effective_sample_size=window.min_effective_sample_size,
        max_top_10_weight_share_pct=window.max_top_10_weight_share_pct,
        max_top_100_weight_share_pct=window.max_top_100_weight_share_pct,
    )
    if not approximate_issues:
        return "approximate"

    return "aggregate"
