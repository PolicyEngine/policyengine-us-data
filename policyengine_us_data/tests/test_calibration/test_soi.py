from policyengine_us_data.utils.soi import get_soi


def test_get_soi_includes_mortgage_interest_deduction_targets():
    soi = get_soi(2024)
    mortgage_interest = soi[soi.Variable == "mortgage_interest_deductions"]

    assert not mortgage_interest.empty
    assert mortgage_interest["Value"].gt(0).all()
