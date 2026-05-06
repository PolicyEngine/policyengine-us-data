"""Map legacy national ``build_loss_matrix`` labels to target DB rows."""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from policyengine_us_data.storage import STORAGE_FOLDER

SCHEMA_VERSION = 1

EITC_AGI_CHILD_DOMAIN = "adjusted_gross_income,eitc,eitc_child_count"

_EITC_AGI_CHILD_LABEL = re.compile(
    r"^nation/irs/eitc/(?P<metric>returns|amount)/"
    r"c(?P<count_children>\d+)_(?P<agi_lower>[^_]+)_(?P<agi_upper>[^/]+)$"
)
_CTC_LABEL = re.compile(
    r"^nation/irs/(?P<variable>refundable_ctc|non_refundable_ctc)(?P<count>_count)?$"
)
_REAL_ESTATE_TAX_LABEL = re.compile(r"^nation/irs/real_estate_taxes(?P<count>_count)?$")
_SOI_TAXABLE_DETAIL_LABEL = re.compile(
    r"^nation/irs/.+/(?:count|total)/AGI in .+/taxable/.+$"
)
_AGI_RANGE_LABEL = re.compile(
    r"^(?P<lower>-inf|inf|[0-9.]+(?:bn|m|k)?)-"
    r"(?P<upper>-inf|inf|[0-9.]+(?:bn|m|k)?)$"
)
_EITC_STATE_LABEL = re.compile(r"^nation/irs/eitc/(?:returns|amount)/state_[0-9]+$")
_NATIONAL_AGE_LABEL = re.compile(r"^nation/census/population_by_age/[0-9]+$")
_MEDICARE_PART_B_AGE_LABEL = re.compile(
    r"^nation/census/medicare_part_b_premiums/age_.+$"
)
_SPM_THRESHOLD_LABEL = re.compile(
    r"^nation/census/(?:agi|count)_in_spm_threshold_decile_[0-9]+$"
)
_SOI_FILER_AGI_LABEL = re.compile(r"^nation/soi/filer_count/agi_.+$")
_DEPRECATED_SPM_SURVEY_LABEL = re.compile(
    r"^nation/census/(?:spm_unit_|(?:agi|count)_in_spm_threshold_decile_).+$"
)

_DIRECT_NATIONAL_CENSUS_TARGET_VARIABLES = {
    "medicaid",
    "medicare_part_b_premiums",
    "rent",
    "real_estate_taxes",
    "tip_income",
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
    "traditional_ira_contributions",
    "traditional_401k_contributions",
    "roth_401k_contributions",
    "self_employed_pension_contribution_ald",
    "roth_ira_contributions",
}

_SOI_TAXABLE_DETAIL_TARGET_VARIABLES = {
    ("adjusted gross income", "total"): "adjusted_gross_income",
    ("count", "count"): "tax_unit_count",
}
_SOI_FILING_STATUS_CONSTRAINTS = {
    "Single": ("==", "SINGLE"),
    "Head of Household": ("==", "HEAD_OF_HOUSEHOLD"),
    "Married Filing Separately": ("==", "SEPARATE"),
    "Married Filing Jointly/Surviving Spouse": (
        "in",
        "JOINT|SURVIVING_SPOUSE",
    ),
}

_DB_CONSTRAINTS_QUERY = """
    SELECT
        v.target_id,
        v.stratum_id,
        v.variable,
        v.reform_id,
        v.value,
        v.period,
        v.active,
        v.geo_level,
        v.geographic_id,
        v.domain_variable,
        t.source,
        t.notes,
        sc.constraint_variable,
        sc.operation,
        sc.value AS constraint_value
    FROM target_overview v
    JOIN targets t
        ON t.target_id = v.target_id
    LEFT JOIN stratum_constraints sc
        ON sc.stratum_id = v.stratum_id
    WHERE
        v.active = 1
        AND v.geo_level = 'national'
    ORDER BY v.target_id, sc.constraint_variable, sc.operation, sc.value
"""


@dataclass(frozen=True)
class Constraint:
    variable: str
    operation: str
    value: str

    def normalized(self) -> tuple[str, str, str]:
        return (self.variable, self.operation, _normalize_constraint_value(self.value))


@dataclass(frozen=True)
class TargetRecord:
    target_id: int
    stratum_id: int
    variable: str
    reform_id: int
    value: float | None
    period: int
    source: str | None
    notes: str | None
    geo_level: str
    geographic_id: str
    domain_variable: str | None
    constraints: tuple[Constraint, ...]

    @property
    def constraints_set(self) -> frozenset[tuple[str, str, str]]:
        return frozenset(constraint.normalized() for constraint in self.constraints)

    def to_manifest_match(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "stratum_id": self.stratum_id,
            "variable": self.variable,
            "reform_id": self.reform_id,
            "period": self.period,
            "value": self.value,
            "source": self.source,
            "domain_variable": self.domain_variable,
            "constraints": [
                {
                    "variable": constraint.variable,
                    "operation": constraint.operation,
                    "value": constraint.value,
                }
                for constraint in self.constraints
            ],
        }


def _normalize_constraint_value(value: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return repr(number)


def _parse_numeric_token(token: str) -> float:
    if token == "-inf":
        return float("-inf")
    if token == "inf":
        return float("inf")
    multipliers = {
        "bn": 1_000_000_000.0,
        "m": 1_000_000.0,
        "k": 1_000.0,
    }
    for suffix, multiplier in multipliers.items():
        if token.endswith(suffix):
            return float(token[: -len(suffix)]) * multiplier
    return float(token)


def _constraint(variable: str, operation: str, value: Any) -> Constraint:
    return Constraint(variable=variable, operation=operation, value=str(value))


def load_national_target_records(db_path: str | Path) -> list[TargetRecord]:
    """Load active national target DB rows with their constraints."""

    path = Path(db_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    grouped: dict[int, dict[str, Any]] = {}
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(_DB_CONSTRAINTS_QUERY):
            target_id = int(row["target_id"])
            target = grouped.setdefault(
                target_id,
                {
                    "target_id": target_id,
                    "stratum_id": int(row["stratum_id"]),
                    "variable": row["variable"],
                    "reform_id": int(row["reform_id"]),
                    "value": row["value"],
                    "period": int(row["period"]),
                    "source": row["source"],
                    "notes": row["notes"],
                    "geo_level": row["geo_level"],
                    "geographic_id": row["geographic_id"],
                    "domain_variable": row["domain_variable"],
                    "constraints": [],
                },
            )
            if row["constraint_variable"] is not None:
                target["constraints"].append(
                    Constraint(
                        variable=row["constraint_variable"],
                        operation=row["operation"],
                        value=row["constraint_value"],
                    )
                )

    return [
        TargetRecord(
            target_id=row["target_id"],
            stratum_id=row["stratum_id"],
            variable=row["variable"],
            reform_id=row["reform_id"],
            value=row["value"],
            period=row["period"],
            source=row["source"],
            notes=row["notes"],
            geo_level=row["geo_level"],
            geographic_id=row["geographic_id"],
            domain_variable=row["domain_variable"],
            constraints=tuple(row["constraints"]),
        )
        for row in grouped.values()
    ]


class NationalTargetIndex:
    def __init__(self, records: Sequence[TargetRecord]):
        self.records = list(records)

    def match(
        self,
        *,
        variable: str,
        period: int,
        domain_variable: str | None = None,
        reform_id: int = 0,
        constraints: Sequence[Constraint] = (),
    ) -> list[TargetRecord]:
        required_constraints = {constraint.normalized() for constraint in constraints}
        candidates = [
            record
            for record in self.records
            if record.variable == variable
            and record.reform_id == reform_id
            and record.domain_variable == domain_variable
            and record.period <= period
            and required_constraints <= record.constraints_set
        ]
        if not candidates:
            return []
        latest_period = max(record.period for record in candidates)
        return [record for record in candidates if record.period == latest_period]


def _match_result(
    target_name: str,
    matches: Sequence[TargetRecord],
    *,
    reason: str,
) -> dict[str, Any]:
    if len(matches) == 1:
        match = matches[0]
        return {
            "target_name": target_name,
            "scope": "national",
            "status": "matched",
            "reason": reason,
            "target_id": match.target_id,
            "target": match.to_manifest_match(),
        }
    if len(matches) > 1:
        return {
            "target_name": target_name,
            "scope": "national",
            "status": "ambiguous",
            "reason": f"{reason}_ambiguous",
            "matches": [match.to_manifest_match() for match in matches],
        }
    return {
        "target_name": target_name,
        "scope": "national",
        "status": "db_missing",
        "reason": f"{reason}_missing_from_target_db",
    }


def classify_national_target(
    target_name: str,
    index: NationalTargetIndex,
    *,
    period: int,
    target_value: float | None = None,
) -> dict[str, Any]:
    """Classify a legacy national loss label against the structured target DB."""

    if not target_name.startswith("nation/"):
        return {
            "target_name": target_name,
            "scope": "non_national",
            "status": "out_of_scope",
            "reason": "non_national_loss_target",
        }

    match = _EITC_AGI_CHILD_LABEL.match(target_name)
    if match:
        count_children = int(match.group("count_children"))
        child_constraint = (
            _constraint("eitc_child_count", "==", count_children)
            if count_children < 3
            else _constraint("eitc_child_count", ">", 2)
        )
        variable = "tax_unit_count" if match.group("metric") == "returns" else "eitc"
        matches = index.match(
            variable=variable,
            domain_variable=EITC_AGI_CHILD_DOMAIN,
            period=period,
            constraints=[
                _constraint("tax_unit_is_filer", "==", 1),
                _constraint("eitc", ">", 0),
                child_constraint,
                _constraint(
                    "adjusted_gross_income",
                    ">=",
                    _parse_numeric_token(match.group("agi_lower")),
                ),
                _constraint(
                    "adjusted_gross_income",
                    "<",
                    _parse_numeric_token(match.group("agi_upper")),
                ),
            ],
        )
        if not matches and target_value is not None and math.isclose(target_value, 0.0):
            return {
                "target_name": target_name,
                "scope": "national",
                "status": "legacy_only",
                "reason": "zero_eitc_agi_child_target_omitted_from_target_db",
            }
        return _match_result(
            target_name,
            matches,
            reason="structured_eitc_agi_child_target",
        )

    match = _CTC_LABEL.match(target_name)
    if match:
        variable = match.group("variable")
        matches = index.match(
            variable="tax_unit_count" if match.group("count") else variable,
            domain_variable=variable,
            period=period,
            constraints=[
                _constraint("tax_unit_is_filer", "==", 1),
                _constraint(variable, ">", 0),
            ],
        )
        return _match_result(target_name, matches, reason="structured_ctc_target")

    match = _REAL_ESTATE_TAX_LABEL.match(target_name)
    if match:
        matches = index.match(
            variable="tax_unit_count" if match.group("count") else "real_estate_taxes",
            domain_variable="real_estate_taxes,tax_unit_itemizes",
            period=period,
            constraints=[
                _constraint("tax_unit_is_filer", "==", 1),
                _constraint("tax_unit_itemizes", "==", 1),
                _constraint("real_estate_taxes", ">", 0),
            ],
        )
        return _match_result(
            target_name,
            matches,
            reason="structured_real_estate_tax_itemizer_target",
        )

    if target_name.startswith("nation/cbo/"):
        variable = target_name.removeprefix("nation/cbo/")
        matches = index.match(variable=variable, period=period)
        return _match_result(target_name, matches, reason="structured_cbo_target")

    direct_census_variable = _direct_census_variable(target_name)
    if direct_census_variable is not None:
        matches = index.match(variable=direct_census_variable, period=period)
        return _match_result(
            target_name,
            matches,
            reason="structured_direct_national_target",
        )

    if target_name == "nation/hhs/medicaid_spending":
        return _match_result(
            target_name,
            index.match(variable="medicaid", period=period),
            reason="structured_medicaid_spending_target",
        )
    if target_name == "nation/hhs/medicaid_enrollment":
        return _match_result(
            target_name,
            index.match(
                variable="person_count",
                domain_variable="medicaid",
                period=period,
                constraints=[_constraint("medicaid", ">", 0)],
            ),
            reason="structured_medicaid_enrollment_target",
        )
    if target_name == "nation/gov/aca_enrollment":
        return _match_result(
            target_name,
            index.match(
                variable="person_count",
                domain_variable="aca_ptc",
                period=period,
                constraints=[_constraint("aca_ptc", ">", 0)],
            ),
            reason="structured_aca_enrollment_target",
        )
    if target_name == "nation/census/tanf":
        return _match_result(
            target_name,
            index.match(
                variable="tanf",
                domain_variable="tanf",
                period=period,
                constraints=[_constraint("tanf", ">", 0)],
            ),
            reason="structured_tanf_cash_assistance_target",
        )
    if target_name.startswith("nation/census/acs/"):
        variable = target_name.removeprefix("nation/census/acs/")
        return _match_result(
            target_name,
            index.match(variable=variable, period=period),
            reason="structured_acs_national_target",
        )
    if target_name == "nation/bls/ce/childcare_expenses":
        return _match_result(
            target_name,
            index.match(variable="childcare_expenses", period=period),
            reason="structured_bls_ce_target",
        )
    if target_name == "nation/net_worth/total":
        return _match_result(
            target_name,
            index.match(variable="net_worth", period=period),
            reason="structured_net_worth_target",
        )
    if target_name == "nation/ssa/ssn_card_type_none_count":
        return _match_result(
            target_name,
            index.match(
                variable="person_count",
                domain_variable="ssn_card_type",
                period=period,
                constraints=[_constraint("ssn_card_type", "==", "NONE")],
            ),
            reason="structured_ssn_card_type_target",
        )
    if target_name == "nation/db/liheap/household_count":
        return _match_result(
            target_name,
            index.match(
                variable="household_count",
                domain_variable="spm_unit_energy_subsidy_reported",
                period=period,
                constraints=[_constraint("spm_unit_energy_subsidy_reported", ">", 0)],
            ),
            reason="structured_liheap_target",
        )

    tax_expenditure = _tax_expenditure_variable(target_name)
    if tax_expenditure is not None:
        variable, reform_id = tax_expenditure
        return _match_result(
            target_name,
            index.match(variable=variable, reform_id=reform_id, period=period),
            reason="structured_tax_expenditure_target",
        )

    soi_taxable_detail = _parse_soi_taxable_detail_target(target_name)
    if soi_taxable_detail is not None:
        variable, domain_variable, constraints = soi_taxable_detail
        matches = index.match(
            variable=variable,
            domain_variable=domain_variable,
            period=period,
            constraints=constraints,
        )
        if not matches and _soi_taxable_detail_label_has_lossy_agi_range(target_name):
            return {
                "target_name": target_name,
                "scope": "national",
                "status": "legacy_only",
                "reason": "legacy_soi_taxable_agi_label_has_lossy_bucket_encoding",
            }
        return _match_result(
            target_name,
            matches,
            reason="structured_soi_taxable_agi_filing_status_target",
        )

    return {
        "target_name": target_name,
        "scope": "national",
        "status": "legacy_only",
        "reason": _legacy_reason(target_name),
    }


def _direct_census_variable(target_name: str) -> str | None:
    prefix = "nation/census/"
    if not target_name.startswith(prefix):
        return None
    variable = target_name.removeprefix(prefix)
    if "/" in variable:
        return None
    if variable not in _DIRECT_NATIONAL_CENSUS_TARGET_VARIABLES:
        return None
    return variable


def _tax_expenditure_variable(target_name: str) -> tuple[str, int] | None:
    mapping = {
        "nation/jct/salt_deduction_expenditure": ("salt_deduction", 1),
        "nation/jct/medical_expense_deduction_expenditure": (
            "medical_expense_deduction",
            2,
        ),
        "nation/jct/charitable_deduction_expenditure": (
            "charitable_deduction",
            3,
        ),
        "nation/jct/interest_deduction_expenditure": (
            "deductible_mortgage_interest",
            4,
        ),
        "nation/jct/qualified_business_income_deduction_expenditure": (
            "qualified_business_income_deduction",
            5,
        ),
    }
    return mapping.get(target_name)


def _parse_soi_taxable_detail_target(
    target_name: str,
) -> tuple[str, str, list[Constraint]] | None:
    if not _SOI_TAXABLE_DETAIL_LABEL.match(target_name):
        return None

    body = target_name.removeprefix("nation/irs/")
    try:
        variable_and_metric, rest = body.split("/AGI in ", 1)
        variable_label, metric = variable_and_metric.rsplit("/", 1)
        agi_range, taxable_label, filing_status = rest.split("/", 2)
    except ValueError:
        return None
    if taxable_label != "taxable":
        return None

    variable = _SOI_TAXABLE_DETAIL_TARGET_VARIABLES.get((variable_label, metric))
    range_match = _AGI_RANGE_LABEL.match(agi_range)
    if variable is None or range_match is None:
        return None

    constraints = [
        _constraint("tax_unit_is_filer", "==", 1),
        _constraint("income_tax_before_credits", ">", 0),
        _constraint(
            "adjusted_gross_income",
            ">=",
            _parse_numeric_token(range_match.group("lower")),
        ),
        _constraint(
            "adjusted_gross_income",
            "<",
            _parse_numeric_token(range_match.group("upper")),
        ),
    ]
    domain_variables = ["adjusted_gross_income", "income_tax_before_credits"]
    filing_constraint = _SOI_FILING_STATUS_CONSTRAINTS.get(filing_status)
    if filing_constraint is not None:
        operation, value = filing_constraint
        constraints.append(_constraint("filing_status", operation, value))
        domain_variables.append("filing_status")
    elif filing_status != "All":
        return None

    return variable, ",".join(sorted(domain_variables)), constraints


def _soi_taxable_detail_label_has_lossy_agi_range(target_name: str) -> bool:
    try:
        agi_range = target_name.split("/AGI in ", 1)[1].split("/", 1)[0]
    except IndexError:
        return False

    return agi_range in {
        "1m-2m",
        "2m-2m",
        "676k-3m",
        "3m-16m",
        "16m-79m",
        "79m-inf",
    }


def _legacy_reason(target_name: str) -> str:
    if _SOI_TAXABLE_DETAIL_LABEL.match(target_name):
        return "legacy_soi_taxable_agi_filing_status_detail_not_in_target_db"
    if _EITC_STATE_LABEL.match(target_name):
        return "legacy_eitc_state_targets_not_structured_in_target_db"
    if _NATIONAL_AGE_LABEL.match(target_name):
        return "legacy_single_year_age_targets_replaced_by_db_age_bins"
    if _MEDICARE_PART_B_AGE_LABEL.match(target_name):
        return "legacy_age_bucketed_medicare_part_b_premiums_not_in_target_db"
    if _SPM_THRESHOLD_LABEL.match(target_name):
        return "deprecated_survey_spm_threshold_target_removed_from_national_pipeline"
    if _DEPRECATED_SPM_SURVEY_LABEL.match(target_name):
        return "deprecated_survey_spm_target_removed_from_national_pipeline"
    if target_name in {
        "nation/census/alimony_expense",
        "nation/census/alimony_income",
        "nation/census/child_support_expense",
        "nation/census/child_support_received",
    }:
        return "deprecated_survey_transfer_flow_target_removed_from_national_pipeline"
    if target_name in {
        "nation/census/health_insurance_premiums_without_medicare_part_b",
        "nation/census/other_medical_expenses",
        "nation/census/over_the_counter_health_expenses",
    }:
        return "deprecated_survey_health_expense_target_removed_from_national_pipeline"
    if _SOI_FILER_AGI_LABEL.match(target_name):
        return "legacy_soi_filer_agi_label_has_lossy_bucket_encoding"
    if target_name == "nation/gov/aca_spending":
        return "legacy_cms_aca_spending_target_not_in_target_db"
    if target_name.startswith("nation/accounting/"):
        return "legacy_accounting_balance_target_not_in_target_db"
    if target_name.startswith("nation/irs/negative_household_market_income_"):
        return "legacy_negative_market_income_target_not_in_target_db"
    if target_name == "nation/census/infants":
        return "legacy_single_year_infant_target_not_in_target_db"
    return "legacy_national_target_not_structured_in_target_db"


def build_national_target_parity_manifest(
    target_names: Iterable[str | Mapping[str, Any]],
    *,
    db_path: str | Path,
    period: int,
) -> dict[str, Any]:
    records = load_national_target_records(db_path)
    index = NationalTargetIndex(records)
    national_targets = []
    for target in target_names:
        normalized = _normalize_target_input(target)
        if normalized["target_name"].startswith("nation/"):
            national_targets.append(normalized)
    entries = []
    for target in national_targets:
        entry = classify_national_target(
            target["target_name"],
            index,
            period=period,
            target_value=target.get("target_value"),
        )
        if target.get("target_value") is not None:
            entry["target_value"] = target["target_value"]
        entries.append(entry)
    summary: dict[str, Any] = {
        "total": len(entries),
        "statuses": _count_by(entries, "status"),
        "reasons": _count_by(entries, "reason"),
    }
    matched = summary["statuses"].get("matched", 0)
    summary["match_rate"] = matched / len(entries) if entries else None
    return {
        "schema_version": SCHEMA_VERSION,
        "period": period,
        "target_db_path": str(Path(db_path).expanduser()),
        "summary": summary,
        "targets": entries,
    }


def _count_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _normalize_target_input(target: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(target, Mapping):
        target_name = str(target["target_name"])
        target_value = target.get("target_value")
        return {
            "target_name": target_name,
            "target_value": (float(target_value) if target_value is not None else None),
        }
    return {"target_name": str(target), "target_value": None}


def extract_target_names_from_json(path: str | Path) -> list[str]:
    return [target["target_name"] for target in extract_targets_from_json(path)]


def extract_targets_from_json(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        return [_normalize_target_input(value) for value in payload]
    if isinstance(payload, dict):
        if isinstance(payload.get("target_names"), list):
            return [
                {"target_name": str(value), "target_value": None}
                for value in payload["target_names"]
            ]
        if isinstance(payload.get("targets"), list):
            targets = []
            for row in payload["targets"]:
                if isinstance(row, dict) and "target_name" in row:
                    targets.append(_normalize_target_input(row))
                else:
                    targets.append({"target_name": str(row), "target_value": None})
            return targets
    raise ValueError(
        "Expected JSON list, {'target_names': [...]}, or {'targets': [{'target_name': ...}]}"
    )


def extract_target_names_from_dataset(
    dataset_path: str | Path,
    *,
    period: int,
) -> list[str]:
    return [
        target["target_name"]
        for target in extract_targets_from_dataset(dataset_path, period=period)
    ]


def extract_targets_from_dataset(
    dataset_path: str | Path,
    *,
    period: int,
) -> list[dict[str, Any]]:
    from policyengine_core.data import Dataset
    from policyengine_us_data.utils.loss import build_loss_matrix

    class LocalDataset(Dataset):
        name = "national_target_parity_dataset"
        label = name
        file_path = str(dataset_path)
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = period

    loss_matrix, targets_array = build_loss_matrix(LocalDataset, period)
    return [
        {"target_name": str(column), "target_value": float(target_value)}
        for column, target_value in zip(loss_matrix.columns, targets_array)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a parity manifest from legacy national build_loss_matrix "
            "labels to structured policy_data.db targets."
        )
    )
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument(
        "--target-db",
        default=str(STORAGE_FOLDER / "calibration" / "policy_data.db"),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--target-names-json")
    source.add_argument("--dataset-path")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    if args.target_names_json:
        target_names = extract_targets_from_json(args.target_names_json)
    else:
        target_names = extract_targets_from_dataset(
            args.dataset_path,
            period=args.period,
        )

    manifest = build_national_target_parity_manifest(
        target_names,
        db_path=args.target_db,
        period=args.period,
    )
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
