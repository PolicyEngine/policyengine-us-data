import numpy as np
import pandas as pd
from typing import Tuple

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.storage.calibration_targets.make_county_cd_distributions import build_county_fips_to_enum_mapping

def _get_enhanced_cps_national_targets(data_year: int) -> Tuple[pd.DataFrame, int]:
    """
    Get national targets for Enhanced CPS data.

    Args:
    data_year (int): The year for which to get targets.

    Returns:
    Tuple[pd.DataFrame, int]: A tuple containing the targets DataFrame and the data year.
    """
    # Load national targets from CSV file
    targets_file = f"{STORAGE_FOLDER}/calibration_targets/enhanced_cps_national_targets_{data_year}.csv"
    targets = pd.read_csv(targets_file)

    # Validate input types
    if not isinstance(data_year, int):
        raise ValueError("data_year must be an integer")

    # Handle edge case: data_year is not in the targets file
    if data_year not in targets['year'].values:
        raise ValueError(f"No targets found for data_year {data_year}")

    # Get county FIPS to enum mapping
    county_fips_to_enum = build_county_fips_to_enum_mapping()

    # Validate county FIPS to enum mapping
    if not isinstance(county_fips_to_enum, dict):
        raise ValueError("county_fips_to_enum must be a dictionary")

    # Get national targets for the specified data year
    national_targets = targets[targets['year'] == data_year]

    # Validate national targets
    if national_targets.empty:
        raise ValueError(f"No national targets found for data_year {data_year}")

    # Return national targets and data year
    return national_targets, data_year

def _get_enhanced_cps_national_targets_with_inflation_correction(data_year: int) -> Tuple[pd.DataFrame, int]:
    """
    Get national targets for Enhanced CPS data with inflation correction.

    Args:
    data_year (int): The year for which to get targets.

    Returns:
    Tuple[pd.DataFrame, int]: A tuple containing the targets DataFrame and the data year.
    """
    # Get national targets without inflation correction
    targets, data_year = _get_enhanced_cps_national_targets(data_year)

    # Correct inflation in capital-gains/dividend/interest aggregates
    targets['capital_gains'] = targets['capital_gains'] / 12.2
    targets['dividends'] = targets['dividends'] / 5.6
    targets['taxable_interest_income'] = targets['taxable_interest_income'] / 6.2
    targets['partnership_s_corp_income'] = targets['partnership_s_corp_income'] / 2.3
    targets['household_net_income'] = targets['household_net_income'] / 3.7
    targets['household_market_income'] = targets['household_market_income'] / 3.9

    # Return corrected national targets and data year
    return targets, data_year