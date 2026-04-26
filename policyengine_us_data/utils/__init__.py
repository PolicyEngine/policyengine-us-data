from importlib import import_module

_LAZY_MODULES = (
    "policyengine_us_data.utils.soi",
    "policyengine_us_data.utils.uprating",
    "policyengine_us_data.utils.loss",
    "policyengine_us_data.utils.l0",
    "policyengine_us_data.utils.seed",
)

__all__ = [
    "ABSOLUTE_ERROR_SCALE_TARGETS",
    "HardConcrete",
    "build_loss_matrix",
    "get_target_error_normalisation",
    "print_reweighting_diagnostics",
    "set_seeds",
]


def __getattr__(name: str):
    for module_name in _LAZY_MODULES:
        module = import_module(module_name)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
