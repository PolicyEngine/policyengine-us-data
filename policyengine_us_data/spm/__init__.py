"""
SPM (Supplemental Poverty Measure) threshold calculation module.

This module provides tools to calculate SPM thresholds from source data:
- Base thresholds from BLS Consumer Expenditure Survey (by tenure type)
- Geographic adjustments (GEOADJ) from ACS median rents
- Local thresholds for congressional districts

The SPM threshold formula is:
    threshold = base_threshold[tenure] × equivalence_scale × geoadj

Where:
- base_threshold varies by tenure: renter, owner with mortgage, owner without
- equivalence_scale adjusts for family composition
- geoadj adjusts for local housing costs (0.84 in WV to 1.27 in HI)

Usage:
    from policyengine_us_data.spm import (
        calculate_base_thresholds,
        create_district_geoadj_lookup,
        calculate_local_spm_thresholds,
        update_spm_thresholds_for_districts,
    )

    # Get base thresholds for 2024
    base = calculate_base_thresholds(2024)

    # Get district GEOADJ lookup table
    geoadj = create_district_geoadj_lookup(2022)

    # Calculate local thresholds for SPM units
    thresholds = calculate_local_spm_thresholds(
        district_codes=["0612", "0611"],
        tenure_types=["renter", "owner_with_mortgage"],
        num_adults=[2, 1],
        num_children=[2, 0],
        year=2024,
    )
"""

from .ce_threshold import calculate_base_thresholds
from .district_geoadj import (
    create_district_geoadj_lookup,
    get_district_geoadj,
    calculate_geoadj_from_rent,
)
from .local_threshold import (
    calculate_local_spm_thresholds,
    update_spm_thresholds_for_districts,
    spm_equivalence_scale,
)

__all__ = [
    "calculate_base_thresholds",
    "create_district_geoadj_lookup",
    "get_district_geoadj",
    "calculate_geoadj_from_rent",
    "calculate_local_spm_thresholds",
    "update_spm_thresholds_for_districts",
    "spm_equivalence_scale",
]
