from __future__ import annotations

import json
import subprocess
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
UV_LOCK_PATH = REPO_ROOT / "uv.lock"


@dataclass(frozen=True)
class PolicyEngineUSBuildInfo:
    version: str
    locked_version: str | None = None
    git_commit: str | None = None
    source_path: str | None = None

    def to_dict(self) -> dict[str, str]:
        result = {"version": self.version}
        if self.locked_version is not None:
            result["locked_version"] = self.locked_version
        if self.git_commit is not None:
            result["git_commit"] = self.git_commit
        if self.source_path is not None:
            result["source_path"] = self.source_path
        return result

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "PolicyEngineUSBuildInfo":
        return cls(
            version=data["version"],
            locked_version=data.get("locked_version"),
            git_commit=data.get("git_commit"),
            source_path=data.get("source_path"),
        )


def _find_git_root(start_path: Path | None) -> Path | None:
    current = start_path
    while current is not None:
        if (current / ".git").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent
    return None


def _get_git_commit(path: Path | None) -> str | None:
    if path is None:
        return None
    git_root = _find_git_root(path)
    if git_root is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(git_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@lru_cache(maxsize=1)
def ensure_policyengine_us_compat_variables() -> None:
    """Backfill SSTB/QBI variables when running against older policyengine-us.

    The SSTB split landed across `policyengine-us` and `policyengine-us-data`
    in separate PRs. Until the model package release catches up, keep the data
    package usable by registering the missing inputs/formulas on import.
    """

    try:
        from policyengine_us.model_api import (
            Person,
            TaxUnit,
            USD,
            Variable,
            YEAR,
            add,
            max_,
            min_,
            np,
            where,
        )
        from policyengine_us.system import CountryTaxBenefitSystem, system
    except Exception:
        return

    class sstb_self_employment_income(Variable):
        value_type = float
        entity = Person
        label = "SSTB self-employment income"
        unit = USD
        documentation = (
            "Self-employment non-farm income from a specified service trade or "
            "business (SSTB) under IRC Section 199A(d)(2)."
        )
        definition_period = YEAR
        reference = (
            "https://www.law.cornell.edu/uscode/text/26/1402#a",
            "https://www.law.cornell.edu/uscode/text/26/199A#d_2",
        )
        uprating = "calibration.gov.irs.soi.self_employment_income"
        default_value = 0

    class sstb_w2_wages_from_qualified_business(Variable):
        value_type = float
        entity = Person
        label = "SSTB allocable W-2 wages"
        unit = USD
        documentation = (
            "Portion of w2_wages_from_qualified_business allocable to "
            "specified service trades or businesses for section 199A."
        )
        definition_period = YEAR
        reference = (
            "https://www.law.cornell.edu/uscode/text/26/199A#b_2",
            "https://www.law.cornell.edu/uscode/text/26/199A#d_3",
        )
        uprating = "calibration.gov.cbo.income_by_source.adjusted_gross_income"
        default_value = 0

    class sstb_unadjusted_basis_qualified_property(Variable):
        value_type = float
        entity = Person
        label = "SSTB allocable UBIA of qualified property"
        unit = USD
        documentation = (
            "Portion of unadjusted_basis_qualified_property allocable to "
            "specified service trades or businesses for section 199A."
        )
        definition_period = YEAR
        reference = (
            "https://www.law.cornell.edu/uscode/text/26/199A#b_2",
            "https://www.law.cornell.edu/uscode/text/26/199A#d_3",
        )
        default_value = 0

    class sstb_self_employment_income_would_be_qualified(Variable):
        value_type = bool
        entity = Person
        label = "SSTB self-employment income would be qualified"
        documentation = (
            "Whether SSTB self-employment income would count toward qualified "
            "business income before the section 199A(d)(3) phaseout."
        )
        definition_period = YEAR
        reference = "https://www.law.cornell.edu/uscode/text/26/199A#c_3_A"
        default_value = True

    def _split_qbi_components(person, period, parameters):
        p = parameters(period).gov.irs.deductions.qbi
        non_sstb_gross = 0
        for var in p.income_definition:
            non_sstb_gross += person(var, period) * person(
                var + "_would_be_qualified", period
            )
        sstb_gross = person("sstb_self_employment_income", period) * person(
            "sstb_self_employment_income_would_be_qualified", period
        )
        positive_non_sstb_gross = max_(0, non_sstb_gross)
        positive_sstb_gross = max_(0, sstb_gross)
        positive_gross_total = positive_non_sstb_gross + positive_sstb_gross
        qbi_deductions = add(person, period, p.deduction_definition)
        non_sstb_share = where(
            positive_gross_total > 0,
            positive_non_sstb_gross / positive_gross_total,
            0,
        )
        sstb_share = where(
            positive_gross_total > 0,
            positive_sstb_gross / positive_gross_total,
            0,
        )
        return (
            max_(0, non_sstb_gross - qbi_deductions * non_sstb_share),
            max_(0, sstb_gross - qbi_deductions * sstb_share),
        )

    class sstb_qualified_business_income(Variable):
        value_type = float
        entity = Person
        label = "SSTB qualified business income"
        documentation = (
            "Qualified business income from a specified service trade or "
            "business under section 199A(d)(2)."
        )
        unit = USD
        definition_period = YEAR
        reference = (
            "https://www.law.cornell.edu/uscode/text/26/199A#c",
            "https://www.law.cornell.edu/uscode/text/26/199A#d_2",
        )

        def formula(person, period, parameters):
            return _split_qbi_components(person, period, parameters)[1]

    class total_self_employment_income(Variable):
        value_type = float
        entity = Person
        label = "total self-employment income"
        unit = USD
        documentation = (
            "Total non-farm self-employment income, including both SSTB and "
            "non-SSTB Schedule C income."
        )
        definition_period = YEAR
        adds = ["self_employment_income", "sstb_self_employment_income"]
        reference = "https://www.law.cornell.edu/uscode/text/26/1402#a"
        uprating = "calibration.gov.irs.soi.self_employment_income"

    class qualified_business_income(Variable):
        value_type = float
        entity = Person
        label = "Qualified business income"
        documentation = (
            "Business income that qualifies for the qualified business income "
            "deduction."
        )
        unit = USD
        definition_period = YEAR
        reference = "https://www.law.cornell.edu/uscode/text/26/199A#c"
        defined_for = "business_is_qualified"

        def formula(person, period, parameters):
            p = parameters(period).gov.irs.deductions.qbi
            gross_qbi = 0
            for var in p.income_definition:
                gross_qbi += person(var, period) * person(
                    var + "_would_be_qualified", period
                )
            gross_qbi += person("sstb_self_employment_income", period) * person(
                "sstb_self_employment_income_would_be_qualified", period
            )
            qbi_deductions = add(person, period, p.deduction_definition)
            return max_(0, gross_qbi - qbi_deductions)

    class qbid_amount(Variable):
        value_type = float
        entity = Person
        label = "Per-person qualified business income deduction amount"
        unit = USD
        definition_period = YEAR
        reference = (
            "https://www.law.cornell.edu/uscode/text/26/199A#b_1",
            "https://www.law.cornell.edu/uscode/text/26/199A#d_3",
            "https://www.irs.gov/pub/irs-prior/p535--2018.pdf",
            "https://www.irs.gov/pub/irs-pdf/f8995.pdf",
            "https://www.irs.gov/pub/irs-pdf/f8995a.pdf",
        )

        def formula(person, period, parameters):
            p = parameters(period).gov.irs.deductions.qbi
            taxinc_less_qbid = person.tax_unit("taxable_income_less_qbid", period)
            filing_status = person.tax_unit("filing_status", period)
            po_start = p.phase_out.start[filing_status]
            po_length = p.phase_out.length[filing_status]
            reduction_rate = min_(1, (max_(0, taxinc_less_qbid - po_start)) / po_length)
            applicable_rate = 1 - reduction_rate
            total_w2_wages = person("w2_wages_from_qualified_business", period)
            total_b_property = person("unadjusted_basis_qualified_property", period)

            def qbi_component(qbi, full_cap, sstb_multiplier):
                qbid_max = p.max.rate * qbi
                adj_qbid_max = qbid_max * sstb_multiplier
                adj_cap = full_cap * sstb_multiplier
                line11 = min_(adj_qbid_max, adj_cap)
                reduction = reduction_rate * max_(0, adj_qbid_max - adj_cap)
                line26 = max_(0, adj_qbid_max - reduction)
                line12 = where(adj_cap < adj_qbid_max, line26, 0)
                return max_(line11, line12)

            split_non_sstb_qbi = _split_qbi_components(person, period, parameters)[0]
            legacy_total_qbi = person("qualified_business_income", period)
            sstb_qbi_from_se = person("sstb_qualified_business_income", period)
            is_sstb_legacy = person("business_is_sstb", period)
            sstb_qbi = where(is_sstb_legacy, legacy_total_qbi, sstb_qbi_from_se)
            non_sstb_qbi_final = where(
                is_sstb_legacy,
                0,
                split_non_sstb_qbi,
            )

            has_non_sstb = non_sstb_qbi_final > 0
            has_sstb = sstb_qbi > 0
            has_mixed_categories = has_non_sstb & has_sstb

            sstb_w2_wages = where(
                is_sstb_legacy,
                total_w2_wages,
                where(
                    has_mixed_categories,
                    person("sstb_w2_wages_from_qualified_business", period),
                    where(has_sstb, total_w2_wages, 0),
                ),
            )
            non_sstb_w2_wages = where(
                is_sstb_legacy,
                0,
                where(
                    has_mixed_categories,
                    max_(0, total_w2_wages - sstb_w2_wages),
                    where(has_non_sstb, total_w2_wages, 0),
                ),
            )

            sstb_b_property = where(
                is_sstb_legacy,
                total_b_property,
                where(
                    has_mixed_categories,
                    person("sstb_unadjusted_basis_qualified_property", period),
                    where(has_sstb, total_b_property, 0),
                ),
            )
            non_sstb_b_property = where(
                is_sstb_legacy,
                0,
                where(
                    has_mixed_categories,
                    max_(0, total_b_property - sstb_b_property),
                    where(has_non_sstb, total_b_property, 0),
                ),
            )

            def full_cap(w2_wages, b_property):
                wage_cap = w2_wages * p.max.w2_wages.rate
                alt_cap = (
                    w2_wages * p.max.w2_wages.alt_rate
                    + b_property * p.max.business_property.rate
                )
                return max_(wage_cap, alt_cap)

            non_sstb_component = qbi_component(
                non_sstb_qbi_final,
                full_cap(non_sstb_w2_wages, non_sstb_b_property),
                1,
            )
            sstb_component = qbi_component(
                sstb_qbi,
                full_cap(sstb_w2_wages, sstb_b_property),
                applicable_rate,
            )

            reit_ptp_income = person("qualified_reit_and_ptp_income", period)
            reit_ptp_component = p.max.reit_ptp_rate * max_(0, reit_ptp_income)
            return non_sstb_component + sstb_component + reit_ptp_component

    class qualified_business_income_deduction(Variable):
        value_type = float
        entity = TaxUnit
        label = "Qualified business income deduction for tax unit"
        unit = USD
        definition_period = YEAR
        reference = (
            "https://www.law.cornell.edu/uscode/text/26/199A#b_1"
            "https://www.irs.gov/pub/irs-prior/p535--2018.pdf"
        )

        def formula(tax_unit, period, parameters):
            person = tax_unit.members
            qbid_amt = person("qbid_amount", period)
            split_non_sstb_qbi = _split_qbi_components(person, period, parameters)[0]
            legacy_total_qbi = person("qualified_business_income", period)
            sstb_qbi = person("sstb_qualified_business_income", period)
            is_sstb_legacy = person("business_is_sstb", period)
            total_qbi = tax_unit.sum(
                where(
                    is_sstb_legacy,
                    legacy_total_qbi,
                    split_non_sstb_qbi + sstb_qbi,
                )
            )
            uncapped_qbid = tax_unit.sum(qbid_amt)
            taxinc_less_qbid = tax_unit("taxable_income_less_qbid", period)
            netcg_qdiv = tax_unit("adjusted_net_capital_gain", period)
            p = parameters(period).gov.irs.deductions.qbi
            taxinc_cap = p.max.rate * max_(0, taxinc_less_qbid - netcg_qdiv)
            pre_floor_qbid = min_(uncapped_qbid, taxinc_cap)
            if p.deduction_floor.in_effect:
                floor = p.deduction_floor.amount.calc(total_qbi)
                return max_(pre_floor_qbid, floor)
            return pre_floor_qbid

    compat_variables = [
        sstb_self_employment_income,
        sstb_w2_wages_from_qualified_business,
        sstb_unadjusted_basis_qualified_property,
        sstb_self_employment_income_would_be_qualified,
        sstb_qualified_business_income,
        total_self_employment_income,
    ]
    compat_replacements = [
        qualified_business_income,
        qbid_amount,
        qualified_business_income_deduction,
    ]

    def install_compat_variables(tbs) -> None:
        needs_sstb_qbi_compat = "sstb_qualified_business_income" not in tbs.variables
        for variable in compat_variables:
            if variable.__name__ not in tbs.variables:
                tbs.add_variable(variable)
        if needs_sstb_qbi_compat:
            for variable in compat_replacements:
                tbs.replace_variable(variable)

    if not getattr(CountryTaxBenefitSystem, "_policyengine_us_data_compat", False):
        original_init = CountryTaxBenefitSystem.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            install_compat_variables(self)

        CountryTaxBenefitSystem.__init__ = patched_init
        CountryTaxBenefitSystem._policyengine_us_data_compat = True

    install_compat_variables(system)


@lru_cache(maxsize=None)
def get_locked_dependency_version(package_name: str) -> str | None:
    if not UV_LOCK_PATH.exists():
        return None
    lock_data = tomllib.loads(UV_LOCK_PATH.read_text())
    for package in lock_data.get("package", []):
        if package.get("name") == package_name:
            return package.get("version")
    return None


@lru_cache(maxsize=1)
def get_policyengine_us_build_info() -> PolicyEngineUSBuildInfo:
    version = metadata.version("policyengine-us")
    distribution = metadata.distribution("policyengine-us")

    source_path = None
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        source_path = direct_url.get("url")
        if source_path and source_path.startswith("file://"):
            source_path = source_path.removeprefix("file://")
    if source_path is None:
        try:
            import policyengine_us

            source_path = str(Path(policyengine_us.__file__).resolve().parent)
        except Exception:
            source_path = None

    git_commit = _get_git_commit(Path(source_path)) if source_path else None
    return PolicyEngineUSBuildInfo(
        version=version,
        locked_version=get_locked_dependency_version("policyengine-us"),
        git_commit=git_commit,
        source_path=source_path,
    )


def assert_locked_policyengine_us_version() -> PolicyEngineUSBuildInfo:
    build_info = get_policyengine_us_build_info()
    if (
        build_info.locked_version is not None
        and build_info.version != build_info.locked_version
    ):
        raise RuntimeError(
            "Installed policyengine-us version does not match uv.lock: "
            f"found {build_info.version}, expected {build_info.locked_version}."
        )
    return build_info


@lru_cache(maxsize=1)
def _policyengine_us_variable_names() -> frozenset[str]:
    from policyengine_us import CountryTaxBenefitSystem

    ensure_policyengine_us_compat_variables()
    return frozenset(CountryTaxBenefitSystem().variables)


def has_policyengine_us_variables(*variables: str) -> bool:
    try:
        available_variables = _policyengine_us_variable_names()
    except Exception:
        return False

    return set(variables).issubset(available_variables)


def supports_medicare_enrollment_input() -> bool:
    return has_policyengine_us_variables("medicare_enrolled")


def supports_modeled_medicare_part_b_inputs() -> bool:
    return has_policyengine_us_variables(
        "medicare_part_b_premiums_reported",
    )


ensure_policyengine_us_compat_variables()
