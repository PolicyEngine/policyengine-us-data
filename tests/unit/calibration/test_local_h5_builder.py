import importlib.util
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

    contracts = _load_module(
        "policyengine_us_data.calibration.local_h5.contracts",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "contracts.py",
        ),
    )
    weights = _load_module(
        "policyengine_us_data.calibration.local_h5.weights",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "weights.py",
        ),
    )
    selection = _load_module(
        "policyengine_us_data.calibration.local_h5.selection",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "selection.py",
        ),
    )
    entity_graph = _load_module(
        "policyengine_us_data.calibration.local_h5.entity_graph",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "entity_graph.py",
        ),
    )
    source_dataset = _load_module(
        "policyengine_us_data.calibration.local_h5.source_dataset",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "source_dataset.py",
        ),
    )
    reindexing = _load_module(
        "policyengine_us_data.calibration.local_h5.reindexing",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "reindexing.py",
        ),
    )

    fake_us_augmentations = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.us_augmentations"
    )
    fake_us_augmentations.USAugmentationService = type(
        "USAugmentationService", (), {}
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.us_augmentations",
        fake_us_augmentations,
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
    builder = _load_module(
        "policyengine_us_data.calibration.local_h5.builder",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "builder.py",
        ),
    )
    return contracts, selection, reindexing, variables, builder


def _make_selection(selection_module, *, active_blocks):
    CloneSelection = selection_module.CloneSelection
    count = len(active_blocks)
    return CloneSelection(
        active_clone_indices=np.arange(count, dtype=np.int64),
        active_household_indices=np.arange(count, dtype=np.int64),
        active_weights=np.asarray([1.5, 2.5][:count], dtype=float),
        active_block_geoids=np.asarray(active_blocks, dtype=str),
        active_cd_geoids=np.asarray(["0101", "0200"][:count], dtype=str),
        active_county_fips=np.asarray(["01001", "02020"][:count], dtype=str),
        active_state_fips=np.asarray([1, 2][:count], dtype=np.int64),
    )


def test_local_area_dataset_builder_builds_payload_and_injects_ids(monkeypatch):
    contracts, selection_module, reindexing_module, variables, builder_module = (
        _install_fake_package_hierarchy(monkeypatch)
    )

    AreaFilter = contracts.AreaFilter
    H5Payload = variables.H5Payload
    ReindexedEntities = reindexing_module.ReindexedEntities
    LocalAreaDatasetBuilder = builder_module.LocalAreaDatasetBuilder
    LocalAreaBuildArtifacts = builder_module.LocalAreaBuildArtifacts

    selection = _make_selection(selection_module, active_blocks=["block-a", "block-b"])
    reindexed = ReindexedEntities(
        household_source_indices=np.asarray([1, 0], dtype=np.int64),
        person_source_indices=np.asarray([2, 1, 0], dtype=np.int64),
        entity_source_indices={
            "tax_unit": np.asarray([1, 0], dtype=np.int64),
            "spm_unit": np.asarray([0], dtype=np.int64),
        },
        persons_per_clone=np.asarray([2, 1], dtype=np.int64),
        entities_per_clone={
            "tax_unit": np.asarray([1, 1], dtype=np.int64),
            "spm_unit": np.asarray([1, 0], dtype=np.int64),
        },
        new_household_ids=np.asarray([10, 11], dtype=np.int32),
        new_person_ids=np.asarray([20, 21, 22], dtype=np.int32),
        new_person_household_ids=np.asarray([10, 10, 11], dtype=np.int32),
        new_entity_ids={
            "tax_unit": np.asarray([30, 31], dtype=np.int32),
            "spm_unit": np.asarray([40], dtype=np.int32),
        },
        new_person_entity_ids={
            "tax_unit": np.asarray([30, 30, 31], dtype=np.int32),
            "spm_unit": np.asarray([40, 40, 40], dtype=np.int32),
        },
    )

    class FakeSelector:
        def __init__(self):
            self.calls = []

        def select(self, weights, geography, *, filters=()):
            self.calls.append((weights, geography, filters))
            return selection

    class FakeReindexer:
        def __init__(self):
            self.calls = []

        def reindex(self, source, selected):
            self.calls.append((source, selected))
            return reindexed

    class FakeVariableCloner:
        def __init__(self):
            self.calls = []

        def clone(self, source, reindexed_input, policy):
            self.calls.append((source, reindexed_input, policy))
            return H5Payload(
                variables={
                    "source_income": {2024: np.asarray([100.0, 200.0])},
                },
                attrs={"origin": "fake"},
            )

    class FakeAugmenter:
        def __init__(self):
            self.calls = []

        def apply_all(
            self,
            data,
            *,
            time_period,
            selection,
            source,
            reindexed,
            takeup_filter,
        ):
            self.calls.append(
                (time_period, selection, source, reindexed, tuple(takeup_filter or ()))
            )
            data["augmented"] = {time_period: np.asarray([1, 1], dtype=np.int8)}

    selector = FakeSelector()
    reindexer = FakeReindexer()
    cloner = FakeVariableCloner()
    augmenter = FakeAugmenter()
    builder = LocalAreaDatasetBuilder(
        selector=selector,
        reindexer=reindexer,
        variable_cloner=cloner,
        us_augmentations=augmenter,
    )

    source = types.SimpleNamespace(
        n_households=2,
        time_period=2024,
    )
    geography = types.SimpleNamespace()
    filters = (
        AreaFilter(
            geography_field="cd_geoid",
            op="in",
            value=("0101",),
        ),
    )

    built = builder.build(
        weights=np.asarray([1.0, 0.0, 0.0, 2.0], dtype=float),
        geography=geography,
        source=source,
        filters=filters,
        takeup_filter=("snap",),
    )

    assert isinstance(built, LocalAreaBuildArtifacts)
    assert built.selection is selection
    assert built.reindexed is reindexed
    assert built.time_period == 2024
    np.testing.assert_array_equal(
        built.payload.variables["household_id"][2024],
        np.asarray([10, 11], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        built.payload.variables["person_id"][2024],
        np.asarray([20, 21, 22], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        built.payload.variables["household_weight"][2024],
        np.asarray([1.5, 2.5], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        built.payload.variables["augmented"][2024],
        np.asarray([1, 1], dtype=np.int8),
    )
    assert built.payload.attrs == {"origin": "fake"}
    assert selector.calls[0][2] == filters
    assert reindexer.calls[0] == (source, selection)
    assert cloner.calls[0][0] == source
    assert cloner.calls[0][1] is reindexed
    assert augmenter.calls[0][0] == 2024
    assert augmenter.calls[0][-1] == ("snap",)


def test_local_area_dataset_builder_rejects_empty_selection(monkeypatch):
    _, selection_module, _, _, builder_module = _install_fake_package_hierarchy(
        monkeypatch
    )
    LocalAreaDatasetBuilder = builder_module.LocalAreaDatasetBuilder

    class FakeSelector:
        def select(self, *_args, **_kwargs):
            return _make_selection(selection_module, active_blocks=[])

    builder = LocalAreaDatasetBuilder(
        selector=FakeSelector(),
        reindexer=types.SimpleNamespace(),
        variable_cloner=types.SimpleNamespace(),
        us_augmentations=types.SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="No active clones after filtering"):
        builder.build(
            weights=np.asarray([0.0], dtype=float),
            geography=types.SimpleNamespace(),
            source=types.SimpleNamespace(n_households=1, time_period=2024),
            filters=(),
            takeup_filter=None,
        )


def test_local_area_dataset_builder_rejects_empty_block_geoids(monkeypatch):
    _, selection_module, _, _, builder_module = _install_fake_package_hierarchy(
        monkeypatch
    )
    LocalAreaDatasetBuilder = builder_module.LocalAreaDatasetBuilder

    class FakeSelector:
        def select(self, *_args, **_kwargs):
            return _make_selection(selection_module, active_blocks=[""])

    builder = LocalAreaDatasetBuilder(
        selector=FakeSelector(),
        reindexer=types.SimpleNamespace(),
        variable_cloner=types.SimpleNamespace(),
        us_augmentations=types.SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="empty block GEOIDs"):
        builder.build(
            weights=np.asarray([1.0], dtype=float),
            geography=types.SimpleNamespace(),
            source=types.SimpleNamespace(n_households=1, time_period=2024),
            filters=(),
            takeup_filter=None,
        )
