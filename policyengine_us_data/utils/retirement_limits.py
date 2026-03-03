"""Retirement contribution limits from policyengine-us parameters.

Reads IRS contribution limits from the policyengine-us parameter tree
instead of hard-coding them.
"""

from functools import lru_cache


def _catch_up_401k(catch_up_limit, instant: str) -> int:
    """Extract the standard 401(k) catch-up amount.

    Handles two policyengine-us parameter layouts:
      - Older releases: children["401k"] returns an int directly.
      - Newer releases (SECURE 2.0): children["k401"] returns a
        SingleAmountTaxScale with age brackets; we use the age-50
        bracket (the standard catch-up).
    """
    children = catch_up_limit.children
    if "401k" in children:
        return int(children["401k"](instant))
    scale = children["k401"](instant)
    return int(scale.calc(50))


@lru_cache(maxsize=16)
def get_retirement_limits(year: int) -> dict:
    """Return contribution limits for the given tax year.

    Reads from policyengine-us parameters at:
      gov.irs.gross_income.retirement_contributions.limit.{401k, ira}
      gov.irs.gross_income.retirement_contributions.catch_up.limit

    Returns:
        Dict with keys: 401k, 401k_catch_up, ira, ira_catch_up.
    """
    from policyengine_us import CountryTaxBenefitSystem

    tbs = CountryTaxBenefitSystem()
    p = tbs.parameters.gov.irs.gross_income.retirement_contributions
    d = f"{year}-01-01"

    return {
        "401k": int(p.limit.children["401k"](d)),
        "401k_catch_up": _catch_up_401k(p.catch_up.limit, d),
        "ira": int(p.limit.ira(d)),
        "ira_catch_up": int(p.catch_up.limit.ira(d)),
    }
