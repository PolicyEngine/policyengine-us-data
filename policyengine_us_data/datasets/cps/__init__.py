from importlib import import_module

_LAZY_EXPORTS = {
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
