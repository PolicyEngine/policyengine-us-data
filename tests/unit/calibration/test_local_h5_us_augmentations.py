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

    block_assignment = types.ModuleType(
        "policyengine_us_data.calibration.block_assignment"
    )
    block_assignment.derive_geography_from_blocks = lambda blocks: {}
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.block_assignment",
        block_assignment,
    )

    calibration_utils = types.ModuleType(
        "policyengine_us_data.calibration.calibration_utils"
    )
    calibration_utils.calculate_spm_thresholds_vectorized = (
        lambda **kwargs: np.asarray([], dtype=np.float64)
    )
    calibration_utils.load_cd_geoadj_values = lambda cds: {}
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.calibration_utils",
        calibration_utils,
    )

    takeup = types.ModuleType("policyengine_us_data.utils.takeup")
    takeup.apply_block_takeup_to_arrays = lambda **kwargs: {}
    takeup.reported_subsidized_marketplace_by_tax_unit = (
        lambda person_tax_unit_ids, tax_unit_ids, reported_mask: np.asarray(
            [
                bool(
                    reported_mask[person_tax_unit_ids == tax_unit_id].any()
                )
                for tax_unit_id in tax_unit_ids
            ],
            dtype=bool,
        )
    )
    monkeypatch.setitem(sys.modules, "policyengine_us_data.utils.takeup", takeup)

    reindexing = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.reindexing"
    )
    reindexing.ReindexedEntities = type("ReindexedEntities", (), {})
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.reindexing",
        reindexing,
    )

    selection = types.ModuleType("policyengine_us_data.calibration.local_h5.selection")
    selection.CloneSelection = type("CloneSelection", (), {})
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.selection",
        selection,
    )

    source_dataset = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.source_dataset"
    )
    source_dataset.SourceDatasetSnapshot = type("SourceDatasetSnapshot", (), {})
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.source_dataset",
        source_dataset,
    )

    return _load_module(
        "policyengine_us_data.calibration.local_h5.us_augmentations",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "us_augmentations.py",
        ),
    )


def test_us_augmentation_service_applies_geography_outputs(monkeypatch):
    module = _install_fake_package_hierarchy(monkeypatch)
    USAugmentationService = module.USAugmentationService

    def fake_geography_lookup(unique_blocks):
        np.testing.assert_array_equal(unique_blocks, np.asarray(["b1", "b2"]))
        return {
            "state_fips": np.asarray([2, 1]),
            "county_index": np.asarray([1, 0]),
            "county_fips": np.asarray(["02001", "01001"]),
            "tract_geoid": np.asarray(["t2", "t1"]),
        }

    service = USAugmentationService(
        geography_lookup=fake_geography_lookup,
        county_name_lookup=lambda idx: np.asarray(
            [f"COUNTY_{value}" for value in idx], dtype="S"
        ),
    )

    data = {}
    clone_geo = service.apply_geography(
        data,
        time_period=2024,
        active_blocks=np.asarray(["b2", "b1", "b2"]),
        active_clone_cds=np.asarray(["0101", "0201", "0101"]),
    )

    np.testing.assert_array_equal(
        clone_geo["county_fips"],
        np.asarray(["01001", "02001", "01001"]),
    )
    np.testing.assert_array_equal(
        data["state_fips"][2024],
        np.asarray([1, 2, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        data["county"][2024],
        np.asarray([b"COUNTY_0", b"COUNTY_1", b"COUNTY_0"]),
    )
    np.testing.assert_array_equal(
        data["tract_geoid"][2024],
        np.asarray([b"t1", b"t2", b"t1"]),
    )
    np.testing.assert_array_equal(
        data["congressional_district_geoid"][2024],
        np.asarray([101, 201, 101], dtype=np.int32),
    )


def test_us_augmentation_service_applies_la_zip_patch(monkeypatch):
    module = _install_fake_package_hierarchy(monkeypatch)
    service = module.USAugmentationService()
    data = {}

    service.apply_zip_code_patch(
        data,
        time_period=2024,
        county_fips=np.asarray(["06037", "06059", "06037"]),
    )

    np.testing.assert_array_equal(
        data["zip_code"][2024],
        np.asarray([b"90001", b"UNKNOWN", b"90001"]),
    )


def test_us_augmentation_service_recalculates_spm_thresholds(monkeypatch):
    module = _install_fake_package_hierarchy(monkeypatch)
    USAugmentationService = module.USAugmentationService

    captured = {}

    def fake_threshold_calculator(**kwargs):
        captured.update(kwargs)
        return np.asarray([11.0, 22.0])

    service = USAugmentationService(
        cd_geoadj_loader=lambda cds: {"0101": 1.1, "0201": 2.2},
        threshold_calculator=fake_threshold_calculator,
    )

    class FakeProvider:
        def calculate(self, variable, *, map_to=None):
            assert variable == "age"
            assert map_to == "person"
            return types.SimpleNamespace(values=np.asarray([30, 40, 50]))

        def get_known_periods(self, variable):
            assert variable == "spm_unit_tenure_type"
            return (2024,)

        def get_array(self, variable, period):
            assert variable == "spm_unit_tenure_type"
            assert period == 2024
            return np.asarray([b"OWNER", b"RENTER"])

    source = types.SimpleNamespace(variable_provider=FakeProvider())
    reindexed = types.SimpleNamespace(
        person_source_indices=np.asarray([2, 0, 1], dtype=np.int64),
        entity_source_indices={"spm_unit": np.asarray([1, 0], dtype=np.int64)},
        entities_per_clone={"spm_unit": np.asarray([1, 1], dtype=np.int64)},
        new_person_entity_ids={"spm_unit": np.asarray([0, 1, 1], dtype=np.int32)},
    )
    data = {}

    service.apply_spm_thresholds(
        data,
        time_period=2024,
        active_clone_cds=np.asarray(["0101", "0201"]),
        source=source,
        reindexed=reindexed,
    )

    np.testing.assert_array_equal(
        data["spm_unit_spm_threshold"][2024],
        np.asarray([11.0, 22.0]),
    )
    np.testing.assert_array_equal(
        captured["person_ages"],
        np.asarray([50, 30, 40]),
    )
    np.testing.assert_array_equal(
        captured["person_spm_unit_ids"],
        np.asarray([0, 1, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        captured["spm_unit_tenure_types"],
        np.asarray([b"RENTER", b"OWNER"]),
    )
    np.testing.assert_array_equal(
        captured["spm_unit_geoadj"],
        np.asarray([1.1, 2.2]),
    )
    assert captured["year"] == 2024


def test_us_augmentation_service_applies_takeup_with_clone_indices(monkeypatch):
    module = _install_fake_package_hierarchy(monkeypatch)
    USAugmentationService = module.USAugmentationService

    captured = {}

    def fake_takeup_fn(**kwargs):
        captured.update(kwargs)
        return {"snap": np.asarray([True, False, True])}

    service = USAugmentationService(takeup_fn=fake_takeup_fn)
    selection = types.SimpleNamespace(
        active_block_geoids=np.asarray(["b1", "b2"]),
        active_clone_indices=np.asarray([3, 4], dtype=np.int64),
        active_household_indices=np.asarray([1, 0], dtype=np.int64),
        n_household_clones=2,
    )
    source = types.SimpleNamespace(
        household_ids=np.asarray([10, 20], dtype=np.int64),
    )
    reindexed = types.SimpleNamespace(
        persons_per_clone=np.asarray([2, 1], dtype=np.int64),
        entities_per_clone={
            "tax_unit": np.asarray([1, 2], dtype=np.int64),
            "spm_unit": np.asarray([1, 1], dtype=np.int64),
        },
        person_source_indices=np.asarray([2, 3, 0], dtype=np.int64),
        entity_source_indices={
            "tax_unit": np.asarray([1, 2, 0], dtype=np.int64),
            "spm_unit": np.asarray([1, 0], dtype=np.int64),
        },
    )
    data = {}

    service.apply_takeup(
        data,
        time_period=2024,
        takeup_filter=("snap",),
        selection=selection,
        source=source,
        reindexed=reindexed,
        clone_geo={"state_fips": np.asarray([6, 36])},
    )

    np.testing.assert_array_equal(
        captured["hh_blocks"],
        np.asarray(["b1", "b2"]),
    )
    np.testing.assert_array_equal(
        captured["hh_state_fips"],
        np.asarray([6, 36], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        captured["hh_ids"],
        np.asarray([20, 10], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        captured["hh_clone_indices"],
        np.asarray([3, 4], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        captured["entity_hh_indices"]["person"],
        np.asarray([0, 0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        captured["entity_hh_indices"]["tax_unit"],
        np.asarray([0, 1, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        captured["entity_hh_indices"]["spm_unit"],
        np.asarray([0, 1], dtype=np.int64),
    )
    assert captured["entity_counts"] == {
        "person": 3,
        "tax_unit": 3,
        "spm_unit": 2,
    }
    assert captured["time_period"] == 2024
    assert captured["takeup_filter"] == ("snap",)
    np.testing.assert_array_equal(
        data["snap"][2024],
        np.asarray([True, False, True]),
    )


def test_us_augmentation_service_passes_reported_takeup_anchors(monkeypatch):
    module = _install_fake_package_hierarchy(monkeypatch)
    USAugmentationService = module.USAugmentationService

    captured = {}

    def fake_takeup_fn(**kwargs):
        captured.update(kwargs)
        return {}

    service = USAugmentationService(takeup_fn=fake_takeup_fn)
    selection = types.SimpleNamespace(
        active_block_geoids=np.asarray(["b1", "b2"]),
        active_clone_indices=np.asarray([3, 4], dtype=np.int64),
        active_household_indices=np.asarray([1, 0], dtype=np.int64),
        n_household_clones=2,
    )
    source = types.SimpleNamespace(
        household_ids=np.asarray([10, 20], dtype=np.int64),
    )
    reindexed = types.SimpleNamespace(
        persons_per_clone=np.asarray([2, 1], dtype=np.int64),
        entities_per_clone={
            "tax_unit": np.asarray([1, 2], dtype=np.int64),
            "spm_unit": np.asarray([1, 1], dtype=np.int64),
        },
        person_source_indices=np.asarray([2, 3, 0], dtype=np.int64),
        entity_source_indices={
            "tax_unit": np.asarray([1, 2], dtype=np.int64),
            "spm_unit": np.asarray([1, 0], dtype=np.int64),
        },
    )
    data = {
        "person_tax_unit_id": {2024: np.asarray([1, 1, 2], dtype=np.int64)},
        "tax_unit_id": {2024: np.asarray([1, 2], dtype=np.int64)},
        "reported_has_subsidized_marketplace_health_coverage_at_interview": {
            2024: np.asarray([True, False, False], dtype=bool)
        },
        "has_medicaid_health_coverage_at_interview": {
            2024: np.asarray([False, True, False], dtype=bool)
        },
    }

    service.apply_takeup(
        data,
        time_period=2024,
        takeup_filter=("snap",),
        selection=selection,
        source=source,
        reindexed=reindexed,
        clone_geo={"state_fips": np.asarray([6, 36])},
    )

    np.testing.assert_array_equal(
        captured["reported_anchors"]["takes_up_aca_if_eligible"],
        np.asarray([True, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        captured["reported_anchors"]["takes_up_medicaid_if_eligible"],
        np.asarray([False, True, False], dtype=bool),
    )
