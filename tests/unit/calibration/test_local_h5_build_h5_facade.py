import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np
import pytest


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


def _load_publish_local_area_module(monkeypatch):
    fake_policyengine_us = types.ModuleType("policyengine_us")
    fake_policyengine_us.Microsimulation = object
    monkeypatch.setitem(sys.modules, "policyengine_us", fake_policyengine_us)

    package = types.ModuleType("policyengine_us_data")
    package.__path__ = []
    calibration_package = types.ModuleType("policyengine_us_data.calibration")
    calibration_package.__path__ = []
    local_h5_package = types.ModuleType("policyengine_us_data.calibration.local_h5")
    local_h5_package.__path__ = []
    utils_package = types.ModuleType("policyengine_us_data.utils")
    utils_package.__path__ = []

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
    monkeypatch.setitem(sys.modules, "policyengine_us_data.utils", utils_package)

    @dataclass(frozen=True)
    class FakeAreaFilter:
        geography_field: str
        op: str
        value: tuple[str, ...]

    contracts = types.ModuleType("policyengine_us_data.calibration.local_h5.contracts")
    contracts.AreaFilter = FakeAreaFilter
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.contracts",
        contracts,
    )

    source_dataset = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.source_dataset"
    )

    @dataclass(frozen=True)
    class FakeSourceDatasetSnapshot:
        dataset_path: Path
        time_period: int
        n_households: int

    class FakeReader:
        instances = []
        snapshot = FakeSourceDatasetSnapshot(
            dataset_path=Path("/tmp/source.h5"),
            time_period=2024,
            n_households=2,
        )

        def __init__(self, sub_entities):
            self.sub_entities = tuple(sub_entities)
            self.load_calls = []
            FakeReader.instances.append(self)

        def load(self, dataset_path):
            self.load_calls.append(Path(dataset_path))
            return self.snapshot

    source_dataset.PolicyEngineDatasetReader = FakeReader
    source_dataset.SourceDatasetSnapshot = FakeSourceDatasetSnapshot
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.source_dataset",
        source_dataset,
    )

    builder_module = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.builder"
    )
    writer_module = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.writer"
    )

    class FakeBuilder:
        instances = []

        def __init__(self):
            self.calls = []
            FakeBuilder.instances.append(self)

        def build(
            self,
            *,
            weights,
            geography,
            source,
            filters=(),
            takeup_filter=None,
        ):
            self.calls.append(
                {
                    "weights": np.asarray(weights),
                    "geography": geography,
                    "source": source,
                    "filters": filters,
                    "takeup_filter": takeup_filter,
                }
            )
            selection = types.SimpleNamespace(
                n_household_clones=2,
                active_weights=np.asarray([1.0, 2.0], dtype=float),
            )
            reindexed = types.SimpleNamespace(
                person_source_indices=np.asarray([0, 1, 2], dtype=np.int64),
                entity_source_indices={
                    "tax_unit": np.asarray([0], dtype=np.int64),
                    "spm_unit": np.asarray([0], dtype=np.int64),
                    "family": np.asarray([0], dtype=np.int64),
                    "marital_unit": np.asarray([0], dtype=np.int64),
                },
            )
            payload = types.SimpleNamespace(dataset_count=7)
            return types.SimpleNamespace(
                payload=payload,
                selection=selection,
                reindexed=reindexed,
                time_period=2024,
            )

    class FakeWriter:
        instances = []

        def __init__(self):
            self.write_calls = []
            self.verify_calls = []
            FakeWriter.instances.append(self)

        def write_payload(self, payload, output_path):
            self.write_calls.append((payload, Path(output_path)))
            return Path(output_path)

        def verify_output(self, output_path, *, time_period):
            self.verify_calls.append((Path(output_path), time_period))
            return {
                "household_count": 2,
                "person_count": 3,
                "household_weight_sum": 3.0,
            }

    builder_module.LocalAreaDatasetBuilder = FakeBuilder
    writer_module.H5Writer = FakeWriter
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.builder",
        builder_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.writer",
        writer_module,
    )

    us_augmentations_module = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.us_augmentations"
    )
    us_augmentations_module.build_reported_takeup_anchors = (
        lambda data, time_period: {}
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.us_augmentations",
        us_augmentations_module,
    )

    weights_module = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.weights"
    )
    weights_module.infer_clone_count_from_weight_length = (
        lambda length, n_households: length // max(n_households, 1)
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.weights",
        weights_module,
    )

    calibration_utils = types.ModuleType(
        "policyengine_us_data.calibration.calibration_utils"
    )
    calibration_utils.STATE_CODES = {1: "AL"}
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.calibration_utils",
        calibration_utils,
    )

    clone_and_assign = types.ModuleType(
        "policyengine_us_data.calibration.clone_and_assign"
    )
    clone_and_assign.GeographyAssignment = object
    clone_and_assign.assign_random_geography = lambda *_a, **_k: object()
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.clone_and_assign",
        clone_and_assign,
    )

    hf_module = types.ModuleType("policyengine_us_data.utils.huggingface")
    hf_module.download_calibration_inputs = lambda *_a, **_k: None
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.utils.huggingface",
        hf_module,
    )

    upload_module = types.ModuleType("policyengine_us_data.utils.data_upload")
    upload_module.upload_local_area_file = lambda *_a, **_k: None
    upload_module.upload_local_area_batch_to_hf = lambda *_a, **_k: None
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.utils.data_upload",
        upload_module,
    )

    takeup_module = types.ModuleType("policyengine_us_data.utils.takeup")
    takeup_module.SIMPLE_TAKEUP_VARS = []
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.utils.takeup",
        takeup_module,
    )

    module = _load_module(
        "publish_local_area_under_test",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "publish_local_area.py",
        ),
    )
    return module, FakeReader, FakeBuilder, FakeWriter


def test_build_h5_delegates_to_builder_and_writer(monkeypatch, tmp_path):
    publish_local_area, FakeReader, FakeBuilder, FakeWriter = (
        _load_publish_local_area_module(monkeypatch)
    )

    output_path = tmp_path / "states" / "AL.h5"
    dataset_path = Path("/tmp/source.h5")
    geography = types.SimpleNamespace(name="geography")

    result = publish_local_area.build_h5(
        weights=np.asarray([1.0, 0.0, 0.0, 2.0], dtype=float),
        geography=geography,
        dataset_path=dataset_path,
        output_path=output_path,
        cd_subset=["0101"],
        takeup_filter=["snap"],
    )

    assert result == output_path
    assert FakeReader.instances[0].load_calls == [dataset_path]
    builder_call = FakeBuilder.instances[0].calls[0]
    assert builder_call["geography"] is geography
    assert builder_call["source"] == FakeReader.snapshot
    assert builder_call["takeup_filter"] == ["snap"]
    assert len(builder_call["filters"]) == 1
    assert builder_call["filters"][0].geography_field == "cd_geoid"
    assert builder_call["filters"][0].value == ("0101",)
    assert FakeWriter.instances[0].write_calls[0][1] == output_path
    assert FakeWriter.instances[0].verify_calls[0] == (output_path, 2024)


def test_build_h5_rejects_mismatched_source_snapshot(monkeypatch, tmp_path):
    publish_local_area, _, _, _ = _load_publish_local_area_module(monkeypatch)

    with pytest.raises(ValueError, match="source_snapshot.dataset_path does not match"):
        publish_local_area.build_h5(
            weights=np.asarray([1.0], dtype=float),
            geography=types.SimpleNamespace(),
            dataset_path=tmp_path / "expected.h5",
            output_path=tmp_path / "out.h5",
            source_snapshot=types.SimpleNamespace(
                dataset_path=tmp_path / "other.h5",
                time_period=2024,
                n_households=1,
            ),
        )
