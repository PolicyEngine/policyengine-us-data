import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.utils.loss import (
    _get_enhanced_cps_national_targets_with_inflation_correction,
    HARD_CODED_TOTALS,
)

def test_enhanced_cps_2024_targets():
    targets, data_year = _get_enhanced_cps_national_targets_with_inflation_correction(2024)

    assert data_year == 2024
    assert len(targets) == 51

    # Check that capital gains, dividends, and interest income are corrected
    assert targets['capital_gains'].sum() < 5 * HARD_CODED_TOTALS['capital_gains']
    assert targets['dividends'].sum() < 5 * HARD_CODED_TOTALS['dividends']
    assert targets['taxable_interest_income'].sum() < 5 * HARD_CODED_TOTALS['taxable_interest_income']
    assert targets['partnership_s_corp_income'].sum() < 5 * HARD_CODED_TOTALS['partnership_s_corp_income']
    assert targets['household_net_income'].sum() < 5 * HARD_CODED_TOTALS['household_net_income']
    assert targets['household_market_income'].sum() < 5 * HARD_CODED_TOTALS['household_market_income']