from __future__ import annotations

import numpy as np
import pandas as pd

# Derived by joining:
# 1. Treasury Tipped Occupation Codes (TTOCs) and related 2018 SOC codes from
#    the IRS "Occupations that customarily and regularly received tips on or
#    before December 31, 2024" list / IRB 2025-42.
# 2. The Census Bureau 2018 occupation code list crosswalk from 2018 Census
#    occupation code to 2018 SOC code.
#
# A few IRS SOC entries correspond to multiple TTOCs. For those collisions we
# pick one representative TTOC because the current policyengine-us logic only
# needs to distinguish listed occupations (TTOC > 0) from unlisted ones. The
# more detailed approximation work belongs here in policyengine-us-data, not in
# policyengine-us.
CENSUS_OCCUPATION_CODE_TO_TTOC = {
    725: 502,
    2350: 507,
    2633: 502,
    2752: 206,
    2755: 207,
    2770: 208,
    2910: 503,
    3602: 501,
    3630: 602,
    4000: 105,
    4010: 106,
    4030: 106,
    4040: 101,
    4055: 107,
    4110: 102,
    4120: 103,
    4130: 104,
    4140: 108,
    4150: 109,
    4160: 106,
    4230: 304,
    4251: 402,
    4350: 506,
    4420: 210,
    4500: 603,
    4510: 603,
    4521: 605,
    4522: 601,
    4600: 508,
    4621: 607,
    4655: 501,
    5130: 203,
    5300: 303,
    6355: 403,
    6442: 404,
    7120: 401,
    7200: 409,
    7315: 405,
    7320: 406,
    7340: 401,
    7540: 408,
    7610: 401,
    7800: 110,
    8510: 401,
    9122: 806,
    9141: 803,
    9142: 802,
    9350: 801,
    9610: 805,
    9620: 809,
}


def derive_treasury_tipped_occupation_code(
    census_occupation_codes: pd.Series | np.ndarray,
) -> np.ndarray:
    """Map CPS PEIOOCC detailed occupation codes to Treasury tipped codes."""

    values = pd.Series(census_occupation_codes, copy=False)
    values = pd.to_numeric(values, errors="coerce").fillna(-1).astype(int)
    return (
        values.map(CENSUS_OCCUPATION_CODE_TO_TTOC).fillna(0).astype(np.int16).to_numpy()
    )
