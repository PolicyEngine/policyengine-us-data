import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


def _module_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[3].joinpath(*parts)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_package_hierarchy(monkeypatch):
    package = types.ModuleType("policyengine_us_data")
    package.__path__ = []
    calibration_package = types.ModuleType("policyengine_us_data.calibration")
    calibration_package.__path__ = []
    local_h5_package = types.ModuleType("policyengine_us_data.calibration.local_h5")
    local_h5_package.__path__ = []

    monkeypatch.setitem(sys.modules, "policyengine_us_data", package)
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration",
        calibration_package,
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5",
        local_h5_package,
    )

    _load_module(
        "policyengine_us_data.calibration.local_h5.entity_graph",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "entity_graph.py",
        ),
    )
    _load_module(
        "policyengine_us_data.calibration.local_h5.source_dataset",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "source_dataset.py",
        ),
    )
    _load_module(
        "policyengine_us_data.calibration.local_h5.reindexing",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "reindexing.py",
        ),
    )
    variables = _load_module(
        "policyengine_us_data.calibration.local_h5.variables",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "variables.py",
        ),
    )
    writer = _load_module(
        "policyengine_us_data.calibration.local_h5.writer",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "writer.py",
        ),
    )
    return variables, writer


def test_h5_writer_writes_payload_and_verifies_output(monkeypatch, tmp_path):
    variables, writer_module = _install_fake_package_hierarchy(monkeypatch)
    H5Payload = variables.H5Payload
    H5Writer = writer_module.H5Writer

    payload = H5Payload(
        variables={
            "household_id": {2024: np.asarray([1, 2], dtype=np.int32)},
            "person_id": {2024: np.asarray([10, 11, 12], dtype=np.int32)},
            "household_weight": {2024: np.asarray([1.5, 2.0], dtype=np.float32)},
            "person_weight": {
                2024: np.asarray([1.0, 1.25, 1.25], dtype=np.float32)
            },
        }
    )
    output_path = tmp_path / "nested" / "local.h5"

    writer = H5Writer()
    written_path = writer.write_payload(payload, output_path)
    summary = writer.verify_output(written_path, time_period=2024)

    assert written_path == output_path
    assert output_path.exists()
    assert summary == {
        "household_count": 2,
        "person_count": 3,
        "household_weight_sum": 3.5,
        "person_weight_sum": 3.5,
    }
