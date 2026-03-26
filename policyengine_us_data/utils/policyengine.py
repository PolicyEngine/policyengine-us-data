from functools import lru_cache


@lru_cache(maxsize=1)
def _policyengine_us_variable_names() -> frozenset[str]:
    from policyengine_us import CountryTaxBenefitSystem

    return frozenset(CountryTaxBenefitSystem().variables)


def has_policyengine_us_variables(*variables: str) -> bool:
    try:
        available_variables = _policyengine_us_variable_names()
    except Exception:
        return False

    return set(variables).issubset(available_variables)
