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
        from policyengine_us.model_api import Person, USD, Variable, YEAR
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

    compat_variables = [
        sstb_self_employment_income,
        sstb_w2_wages_from_qualified_business,
        sstb_unadjusted_basis_qualified_property,
        total_self_employment_income,
    ]

    def install_compat_variables(tbs) -> None:
        for variable in compat_variables:
            if variable.__name__ not in tbs.variables:
                tbs.add_variable(variable)

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
