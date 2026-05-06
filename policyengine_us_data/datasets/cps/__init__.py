from importlib import import_module

_LAZY_EXPORTS = {
    "CPS_CLONE_FEATURE_PREDICTORS": (
        "policyengine_us_data.datasets.cps.extended_cps",
        "CPS_CLONE_FEATURE_PREDICTORS",
    ),
    "CPS_CLONE_FEATURE_VARIABLES": (
        "policyengine_us_data.datasets.cps.extended_cps",
        "CPS_CLONE_FEATURE_VARIABLES",
    ),
    "CPS_ONLY_IMPUTED_VARIABLES": (
        "policyengine_us_data.datasets.cps.extended_cps",
        "CPS_ONLY_IMPUTED_VARIABLES",
    ),
    "CPS_STAGE2_DEMOGRAPHIC_PREDICTORS": (
        "policyengine_us_data.datasets.cps.extended_cps",
        "CPS_STAGE2_DEMOGRAPHIC_PREDICTORS",
    ),
    "CPS_STAGE2_INCOME_PREDICTORS": (
        "policyengine_us_data.datasets.cps.extended_cps",
        "CPS_STAGE2_INCOME_PREDICTORS",
    ),
    "CPS": ("policyengine_us_data.datasets.cps.cps", "CPS"),
    "CPS_2019": ("policyengine_us_data.datasets.cps.cps", "CPS_2019"),
    "CPS_2020": ("policyengine_us_data.datasets.cps.cps", "CPS_2020"),
    "CPS_2021": ("policyengine_us_data.datasets.cps.cps", "CPS_2021"),
    "CPS_2022": ("policyengine_us_data.datasets.cps.cps", "CPS_2022"),
    "CPS_2023": ("policyengine_us_data.datasets.cps.cps", "CPS_2023"),
    "CPS_2024": ("policyengine_us_data.datasets.cps.cps", "CPS_2024"),
    "CPS_2024_Full": (
        "policyengine_us_data.datasets.cps.cps",
        "CPS_2024_Full",
    ),
    "EnhancedCPS": (
        "policyengine_us_data.datasets.cps.enhanced_cps",
        "EnhancedCPS",
    ),
    "EnhancedCPS_2024": (
        "policyengine_us_data.datasets.cps.enhanced_cps",
        "EnhancedCPS_2024",
    ),
    "CURRENT_HEALTH_COVERAGE_REPORTED_VAR_MAP": (
        "policyengine_us_data.datasets.cps.cps",
        "CURRENT_HEALTH_COVERAGE_REPORTED_VAR_MAP",
    ),
    "CURRENT_HEALTH_COVERAGE_RULE_INPUT_ALIAS_MAP": (
        "policyengine_us_data.datasets.cps.cps",
        "CURRENT_HEALTH_COVERAGE_RULE_INPUT_ALIAS_MAP",
    ),
    "ESI_POLICYHOLDER_VARIABLE": (
        "policyengine_us_data.datasets.cps.cps",
        "ESI_POLICYHOLDER_VARIABLE",
    ),
    "ESI_SOURCE_COLUMNS": (
        "policyengine_us_data.datasets.cps.cps",
        "ESI_SOURCE_COLUMNS",
    ),
    "ExtendedCPS": (
        "policyengine_us_data.datasets.cps.extended_cps",
        "ExtendedCPS",
    ),
    "ExtendedCPS_2024": (
        "policyengine_us_data.datasets.cps.extended_cps",
        "ExtendedCPS_2024",
    ),
    "ExtendedCPS_2024_Half": (
        "policyengine_us_data.datasets.cps.extended_cps",
        "ExtendedCPS_2024_Half",
    ),
    "MARKETPLACE_PLAN_BENCHMARK_RATIO_MAX": (
        "policyengine_us_data.datasets.cps.cps",
        "MARKETPLACE_PLAN_BENCHMARK_RATIO_MAX",
    ),
    "MARKETPLACE_PLAN_BENCHMARK_RATIO_MIN": (
        "policyengine_us_data.datasets.cps.cps",
        "MARKETPLACE_PLAN_BENCHMARK_RATIO_MIN",
    ),
    "OTHER_HEALTH_INSURANCE_PREMIUM_TARGETS": (
        "policyengine_us_data.datasets.cps.cps",
        "OTHER_HEALTH_INSURANCE_PREMIUM_TARGETS",
    ),
    "ReweightedCPS_2024": (
        "policyengine_us_data.datasets.cps.enhanced_cps",
        "ReweightedCPS_2024",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attribute_name)
        globals()[name] = value
        return value

    try:
        value = import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        if exc.name == f"{__name__}.{name}":
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from exc
        raise
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
