"""Minimal integration coverage for the build_h5 publishing seam."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import h5py
import numpy as np


FIXTURE_PATH = Path(__file__).with_name("test_fixture_50hh.h5")
SUB_ENTITIES = ("tax_unit", "spm_unit", "family", "marital_unit")
TEST_CDS = ("0200", "3701")
_CD_COUNTY = {
    "0200": "02020",
    "3701": "37183",
}


def _install_stub_packages(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]

    for name in list(sys.modules):
        if name == "policyengine_us" or name.startswith("policyengine_us."):
            sys.modules.pop(name, None)
        if name == "policyengine_us_data" or name.startswith("policyengine_us_data."):
            sys.modules.pop(name, None)

    policyengine_us = types.ModuleType("policyengine_us")
    policyengine_us.Microsimulation = object
    monkeypatch.setitem(sys.modules, "policyengine_us", policyengine_us)

    variables_mod = types.ModuleType("policyengine_us.variables")
    household_mod = types.ModuleType("policyengine_us.variables.household")
    demographic_mod = types.ModuleType(
        "policyengine_us.variables.household.demographic"
    )
    geographic_mod = types.ModuleType(
        "policyengine_us.variables.household.demographic.geographic"
    )
    county_mod = types.ModuleType(
        "policyengine_us.variables.household.demographic.geographic.county"
    )
    county_enum_mod = types.ModuleType(
        "policyengine_us.variables.household.demographic.geographic.county.county_enum"
    )

    class County:
        _member_names_ = ["UNKNOWN", "ANCHORAGE_AK", "WAKE_NC"]

    county_enum_mod.County = County

    monkeypatch.setitem(sys.modules, "policyengine_us.variables", variables_mod)
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us.variables.household",
        household_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us.variables.household.demographic",
        demographic_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us.variables.household.demographic.geographic",
        geographic_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us.variables.household.demographic.geographic.county",
        county_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us.variables.household.demographic.geographic.county.county_enum",
        county_enum_mod,
    )

    spm_calculator = types.ModuleType("spm_calculator")
    spm_calculator.__path__ = []
    spm_calculator_geoadj = types.ModuleType("spm_calculator.geoadj")

    class FakeSPMCalculator:
        def __init__(self, year):
            self.year = year

        def get_base_thresholds(self):
            return {
                "owner_with_mortgage": 10000.0,
                "owner_without_mortgage": 9000.0,
                "renter": 8000.0,
            }

    def spm_equivalence_scale(num_adults: int, num_children: int) -> float:
        return 1.0 + 0.1 * num_adults + 0.05 * num_children

    spm_calculator.SPMCalculator = FakeSPMCalculator
    spm_calculator.spm_equivalence_scale = spm_equivalence_scale
    spm_calculator_geoadj.calculate_geoadj_from_rent = lambda rent: 1.0
    monkeypatch.setitem(sys.modules, "spm_calculator", spm_calculator)
    monkeypatch.setitem(
        sys.modules,
        "spm_calculator.geoadj",
        spm_calculator_geoadj,
    )

    package = types.ModuleType("policyengine_us_data")
    package.__path__ = [str(repo_root / "policyengine_us_data")]
    calibration_package = types.ModuleType("policyengine_us_data.calibration")
    calibration_package.__path__ = [
        str(repo_root / "policyengine_us_data" / "calibration")
    ]
    local_h5_package = types.ModuleType("policyengine_us_data.calibration.local_h5")
    local_h5_package.__path__ = [
        str(repo_root / "policyengine_us_data" / "calibration" / "local_h5")
    ]
    utils_package = types.ModuleType("policyengine_us_data.utils")
    utils_package.__path__ = [str(repo_root / "policyengine_us_data" / "utils")]

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

    calibration_utils = types.ModuleType(
        "policyengine_us_data.calibration.calibration_utils"
    )
    calibration_utils.STATE_CODES = {2: "AK", 37: "NC"}
    calibration_utils.load_cd_geoadj_values = (
        lambda cds: {str(cd): 1.0 for cd in cds}
    )

    def calculate_spm_thresholds_vectorized(
        *,
        person_ages,
        person_spm_unit_ids,
        spm_unit_tenure_types,
        spm_unit_geoadj,
        year,
    ):
        n_units = len(spm_unit_geoadj)
        thresholds = np.full(n_units, 8000.0, dtype=np.float32)
        if n_units and len(person_spm_unit_ids):
            adults = np.zeros(n_units, dtype=np.int32)
            np.add.at(adults, person_spm_unit_ids, (person_ages >= 18).astype(np.int32))
            thresholds += adults.astype(np.float32) * 250.0
        return thresholds * spm_unit_geoadj.astype(np.float32)

    calibration_utils.calculate_spm_thresholds_vectorized = (
        calculate_spm_thresholds_vectorized
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.calibration_utils",
        calibration_utils,
    )


def _load_runtime_modules(monkeypatch):
    _install_stub_packages(monkeypatch)
    importlib.invalidate_caches()

    publish_local_area = importlib.import_module(
        "policyengine_us_data.calibration.publish_local_area"
    )
    clone_and_assign = importlib.import_module(
        "policyengine_us_data.calibration.clone_and_assign"
    )
    entity_graph = importlib.import_module(
        "policyengine_us_data.calibration.local_h5.entity_graph"
    )
    source_dataset = importlib.import_module(
        "policyengine_us_data.calibration.local_h5.source_dataset"
    )
    builder_module = importlib.import_module(
        "policyengine_us_data.calibration.local_h5.builder"
    )
    us_augmentations = importlib.import_module(
        "policyengine_us_data.calibration.local_h5.us_augmentations"
    )

    return (
        publish_local_area,
        clone_and_assign,
        entity_graph,
        source_dataset,
        builder_module,
        us_augmentations,
        sys.modules["policyengine_us_data.calibration.calibration_utils"],
    )


class FixtureVariableProvider:
    def __init__(self, arrays, var_defs):
        self.arrays = arrays
        self.var_defs = var_defs

    def list_variables(self) -> tuple[str, ...]:
        return tuple(sorted(self.var_defs))

    def get_known_periods(self, variable: str) -> tuple[int | str, ...]:
        periods = self.arrays.get(variable, {})
        return tuple(int(period) for period in sorted(periods))

    def get_array(self, variable: str, period: int | str) -> np.ndarray:
        return self.arrays[variable][str(period)]

    def get_variable_definition(self, variable: str):
        return self.var_defs.get(variable)

    def calculate(self, variable: str, *, map_to: str | None = None):
        if variable != "age" or map_to != "person":
            raise KeyError(f"Unsupported calculate call: {variable}, {map_to}")
        return SimpleNamespace(values=self.get_array("age", 2023))


def _load_fixture_arrays():
    arrays: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(FIXTURE_PATH, "r") as fixture:
        for variable in fixture.keys():
            arrays[variable] = {
                period: fixture[variable][period][:]
                for period in fixture[variable].keys()
            }
    return arrays


def _make_fixture_snapshot(source_dataset_module, entity_graph_module):
    arrays = _load_fixture_arrays()
    extractor = entity_graph_module.EntityGraphExtractor(SUB_ENTITIES)
    household_ids = arrays["household_id"]["2023"]
    entity_graph = extractor.extract_from_arrays(
        household_ids=household_ids,
        person_household_ids=arrays["person_household_id"]["2023"],
        entity_id_arrays={
            entity_key: arrays[f"{entity_key}_id"]["2023"]
            for entity_key in SUB_ENTITIES
        },
        person_entity_id_arrays={
            entity_key: arrays[f"person_{entity_key}_id"]["2023"]
            for entity_key in SUB_ENTITIES
        },
    )

    def _var_def(entity_key: str, value_type):
        return SimpleNamespace(entity=SimpleNamespace(key=entity_key), value_type=value_type)

    variable_provider = FixtureVariableProvider(
        arrays=arrays,
        var_defs={
            "age": _var_def("person", int),
            "employment_income": _var_def("person", float),
            "person_weight": _var_def("person", float),
            "state_fips": _var_def("household", int),
            "family_weight": _var_def("family", float),
            "spm_unit_weight": _var_def("spm_unit", float),
            "tax_unit_weight": _var_def("tax_unit", float),
            "marital_unit_weight": _var_def("marital_unit", float),
        },
    )

    snapshot = source_dataset_module.SourceDatasetSnapshot(
        dataset_path=FIXTURE_PATH,
        time_period=2023,
        household_ids=household_ids,
        entity_graph=entity_graph,
        input_variables=frozenset(variable_provider.list_variables()),
        variable_provider=variable_provider,
    )
    return snapshot, arrays


def _make_geography(geography_assignment_cls, n_households: int):
    cd_geoid = np.repeat(np.asarray(TEST_CDS, dtype=str), n_households)
    block_geoid = np.asarray(
        [
            f"{_CD_COUNTY[cd]}{idx:06d}{idx:04d}"[:15]
            for idx, cd in enumerate(cd_geoid)
        ],
        dtype="U15",
    )
    county_fips = np.asarray([block[:5] for block in block_geoid], dtype="U5")
    state_fips = np.asarray([int(block[:2]) for block in block_geoid], dtype=np.int32)
    return geography_assignment_cls(
        block_geoid=block_geoid,
        cd_geoid=cd_geoid,
        county_fips=county_fips,
        state_fips=state_fips,
        n_records=n_households,
        n_clones=len(TEST_CDS),
    )


def _fake_geography_lookup(blocks: np.ndarray):
    blocks = np.asarray(blocks).astype(str)
    county_fips = np.asarray([int(block[:5]) for block in blocks], dtype=np.int32)
    state_fips = np.asarray([int(block[:2]) for block in blocks], dtype=np.int32)
    county_index = np.asarray(
        [1 if int(block[:2]) == 2 else 2 for block in blocks],
        dtype=np.int32,
    )
    return {
        "state_fips": state_fips,
        "county_fips": county_fips,
        "county_index": county_index,
        "block_geoid": blocks.astype("S"),
        "tract_geoid": np.asarray([block[:11] for block in blocks], dtype="S11"),
        "cbsa_code": np.asarray(["99999"] * len(blocks), dtype="S5"),
        "sldu": np.asarray(["000"] * len(blocks), dtype="S3"),
        "sldl": np.asarray(["000"] * len(blocks), dtype="S3"),
        "place_fips": np.asarray(["00000"] * len(blocks), dtype="S5"),
        "vtd": np.asarray(["000000"] * len(blocks), dtype="S6"),
        "puma": np.asarray(["00000"] * len(blocks), dtype="S5"),
        "zcta": np.asarray(["00000"] * len(blocks), dtype="S5"),
    }


def _fake_county_name_lookup(county_indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"COUNTY_{int(idx)}" for idx in np.asarray(county_indices)],
        dtype="S16",
    )


def test_build_h5_writes_structural_output_from_real_fixture(monkeypatch, tmp_path):
    (
        publish_local_area,
        clone_and_assign,
        entity_graph_module,
        source_dataset_module,
        builder_module,
        us_augmentations,
        calibration_utils,
    ) = _load_runtime_modules(monkeypatch)

    snapshot, arrays = _make_fixture_snapshot(
        source_dataset_module,
        entity_graph_module,
    )
    geography = _make_geography(
        clone_and_assign.GeographyAssignment,
        len(snapshot.household_ids),
    )

    weights = np.zeros(len(snapshot.household_ids) * len(TEST_CDS), dtype=float)
    positive_households = [(0, 0, 1.25), (0, 1, 2.5), (1, 2, 1.75), (1, 3, 3.0)]
    for clone_idx, household_idx, weight in positive_households:
        weights[clone_idx * len(snapshot.household_ids) + household_idx] = weight

    expected_household_ids = snapshot.household_ids[[0, 1, 2, 3]]
    expected_household_count = len(expected_household_ids)
    expected_person_count = int(
        np.isin(arrays["person_household_id"]["2023"], expected_household_ids).sum()
    )

    real_builder = builder_module.LocalAreaDatasetBuilder
    augmentation_service = us_augmentations.USAugmentationService(
        geography_lookup=_fake_geography_lookup,
        county_name_lookup=_fake_county_name_lookup,
        cd_geoadj_loader=lambda cds: {str(cd): 1.0 for cd in cds},
        threshold_calculator=calibration_utils.calculate_spm_thresholds_vectorized,
    )
    monkeypatch.setattr(
        publish_local_area,
        "LocalAreaDatasetBuilder",
        lambda: real_builder(us_augmentations=augmentation_service),
    )

    output_path = tmp_path / "minimal_build_h5_output.h5"
    publish_local_area.build_h5(
        weights=weights,
        geography=geography,
        dataset_path=FIXTURE_PATH,
        output_path=output_path,
        cd_subset=list(TEST_CDS),
        takeup_filter=[],
        source_snapshot=snapshot,
    )

    assert output_path.exists()

    with h5py.File(output_path, "r") as h5_file:
        period = "2023"

        household_ids = h5_file["household_id"][period][:]
        person_ids = h5_file["person_id"][period][:]
        household_weights = h5_file["household_weight"][period][:]
        districts = h5_file["congressional_district_geoid"][period][:]
        state_fips = h5_file["state_fips"][period][:]
        ages = h5_file["age"][period][:]
        county = h5_file["county"][period][:]
        spm_thresholds = h5_file["spm_unit_spm_threshold"][period][:]
        snap_takeup = h5_file["takes_up_snap_if_eligible"][period][:]

        assert len(household_ids) == expected_household_count
        assert len(person_ids) == expected_person_count
        assert len(household_weights) == expected_household_count
        assert len(districts) == expected_household_count
        assert len(state_fips) == expected_household_count
        assert len(ages) == expected_person_count
        assert len(county) == expected_household_count
        assert len(spm_thresholds) > 0
        assert len(snap_takeup) > 0

        assert np.array_equal(np.unique(districts), np.asarray([200, 3701], dtype=np.int32))
        assert np.array_equal(np.unique(state_fips), np.asarray([2, 37], dtype=np.int32))
        assert np.all(household_weights > 0)
        assert np.all(np.isfinite(spm_thresholds))
        assert set(np.unique(snap_takeup)).issubset({False, True})
        assert county.dtype.kind == "S"
        assert np.array_equal(
            np.sort(household_weights),
            np.sort(np.asarray([weight for _, _, weight in positive_households], dtype=np.float32)),
        )
