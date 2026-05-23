from policyengine_us_data.parameters import load_take_up_rate


def test_housing_assistance_takeup_rate_reflects_ecps_benchmark_adjustment():
    assert load_take_up_rate("housing_assistance", 2024) == 0.50
