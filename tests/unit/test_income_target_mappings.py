import policyengine_us_data.db.etl_national_targets as etl_national_targets
import policyengine_us_data.utils.loss as loss
from policyengine_us_data.calibration.unified_calibration import load_target_config


TARGET_CONFIG_PATH = "policyengine_us_data/calibration/target_config.yaml"


def _target_config_include_entries():
    config = load_target_config(TARGET_CONFIG_PATH)
    return {
        (
            rule["variable"],
            rule["geo_level"],
            rule.get("domain_variable"),
        )
        for rule in config["include"]
    }


def test_cbo_taxable_interest_and_ordinary_dividends_excludes_qualified_dividends():
    expected = "taxable_interest_income+non_qualified_dividend_income"

    assert loss.TAXABLE_INTEREST_AND_ORDINARY_DIVIDENDS_VARIABLE == expected
    assert etl_national_targets.TAXABLE_INTEREST_AND_ORDINARY_DIVIDENDS_VARIABLE == (
        expected
    )

    target = next(
        target
        for target in etl_national_targets.CBO_INCOME_BY_SOURCE_TARGETS
        if target["parameter"] == "taxable_interest_and_ordinary_dividends"
    )
    assert target["variable"] == expected
    assert "explicitly excluding qualified dividends" in target["notes"]


def test_cbo_income_by_source_targets_match_between_legacy_and_target_db():
    legacy_targets = {
        parameter: variable for variable, parameter in loss.CBO_INCOME_BY_SOURCE_TARGETS
    }
    db_targets = {
        target["parameter"]: target["variable"]
        for target in etl_national_targets.CBO_INCOME_BY_SOURCE_TARGETS
    }

    assert legacy_targets == db_targets


def test_bea_nipa_direct_sum_targets_match_between_legacy_and_target_db():
    assert (
        loss.BEA_NIPA_WAGES_AND_SALARIES_2024
        == etl_national_targets.BEA_NIPA_WAGES_AND_SALARIES_2024
    )
    assert (
        loss.BEA_NIPA_PROPRIETORS_INCOME_2024
        == etl_national_targets.BEA_NIPA_PROPRIETORS_INCOME_2024
    )
    assert (
        loss.NIPA_PROPRIETORS_INCOME_VARIABLE
        == etl_national_targets.NIPA_PROPRIETORS_INCOME_VARIABLE
    )

    legacy_targets = {
        variable: target for _, variable, target in loss.BEA_NIPA_DIRECT_SUM_TARGETS
    }
    assert legacy_targets == {
        "employment_income_before_lsr": (
            etl_national_targets.BEA_NIPA_WAGES_AND_SALARIES_2024
        ),
        etl_national_targets.NIPA_PROPRIETORS_INCOME_VARIABLE: (
            etl_national_targets.BEA_NIPA_PROPRIETORS_INCOME_2024
        ),
    }


def test_bea_nipa_direct_sum_targets_are_in_default_target_config():
    include_entries = _target_config_include_entries()

    expected_entries = {
        ("employment_income_before_lsr", "national", None),
        (etl_national_targets.NIPA_PROPRIETORS_INCOME_VARIABLE, "national", None),
    }

    assert expected_entries <= include_entries
