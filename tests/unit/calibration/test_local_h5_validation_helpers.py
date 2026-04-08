import importlib.util
from pathlib import Path
import sys


def _load_validation_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "policyengine_us_data"
        / "calibration"
        / "local_h5"
        / "validation.py"
    )
    spec = importlib.util.spec_from_file_location("local_h5_validation", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validation = _load_validation_module()
make_validation_error = validation.make_validation_error
summarize_validation_rows = validation.summarize_validation_rows
validation_geo_level_for_area_type = validation.validation_geo_level_for_area_type


def test_validation_geo_level_maps_current_area_types_explicitly():
    assert validation_geo_level_for_area_type("states") == "state"
    assert validation_geo_level_for_area_type("districts") == "district"
    assert validation_geo_level_for_area_type("cities") == "district"
    assert validation_geo_level_for_area_type("national") == "national"


def test_summarize_validation_rows_counts_failures_and_ignores_infinite_error():
    summary = summarize_validation_rows(
        (
            {"sanity_check": "PASS", "rel_abs_error": 0.10},
            {"sanity_check": "FAIL", "rel_abs_error": 0.30},
            {"sanity_check": "FAIL", "rel_abs_error": float("inf")},
        )
    )

    assert summary == {
        "n_targets": 3,
        "n_sanity_fail": 2,
        "mean_rel_abs_error": 0.2,
    }


def test_make_validation_error_returns_structured_payload():
    payload = make_validation_error(
        item_key="district:CA-12",
        error=RuntimeError("validator crashed"),
        traceback_text="traceback lines",
    )

    assert payload == {
        "item": "district:CA-12",
        "error": "validator crashed",
        "traceback": "traceback lines",
    }
