import policyengine_us_data.db.etl_national_targets as etl_national_targets
import policyengine_us_data.utils.loss as loss


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
