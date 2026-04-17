from importlib import import_module

from .geography import ZIP_CODE_DATASET

_LAZY_EXPORTS = {
    "CPS_2024": (
        "policyengine_us_data.datasets.cps.cps",
        "CPS_2024",
    ),
    "EnhancedCPS_2024": (
        "policyengine_us_data.datasets.cps.enhanced_cps",
        "EnhancedCPS_2024",
    ),
    "ExtendedCPS_2024": (
        "policyengine_us_data.datasets.cps.extended_cps",
        "ExtendedCPS_2024",
    ),
    "PUF_2024": (
        "policyengine_us_data.datasets.puf.puf",
        "PUF_2024",
    ),
}

__all__ = ["ZIP_CODE_DATASET", *_LAZY_EXPORTS]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
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

    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
