"""Retirement contribution limits from policyengine-us parameters.

Reads IRS contribution limits from the policyengine-us parameter tree
instead of hard-coding them.
"""

from functools import lru_cache


@lru_cache(maxsize=16)
def get_retirement_limits(year: int) -> dict:
    """Return contribution limits for the given tax year.

    Reads from policyengine-us parameters at:
      gov.irs.gross_income.retirement_contributions.limit.{401k, ira}
      gov.irs.gross_income.retirement_contributions.catch_up.limit.{k401, ira}

    The k401 catch-up parameter is a SingleAmountTaxScale with age
    brackets (SECURE 2.0); we use the age-50 bracket for the standard
    catch-up amount.

    Returns:
        Dict with keys: 401k, 401k_catch_up, ira, ira_catch_up.
    """
    from policyengine_us import CountryTaxBenefitSystem

    tbs = CountryTaxBenefitSystem()
    p = tbs.parameters.gov.irs.gross_income.retirement_contributions
    d = f"{year}-01-01"

    return {
        "401k": int(p.limit.children["401k"](d)),
        "401k_catch_up": int(p.catch_up.limit.children["k401"](d).calc(50)),
        "ira": int(p.limit.ira(d)),
        "ira_catch_up": int(p.catch_up.limit.ira(d)),
    }
