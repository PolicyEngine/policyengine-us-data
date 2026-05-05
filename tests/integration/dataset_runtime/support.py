"""Local helpers for fixture-scale dataset runtime contracts."""

from __future__ import annotations

import filecmp
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from policyengine_us import Microsimulation

PERIOD = "2023"
FIXTURE_PATH = Path(__file__).resolve().parents[1] / "test_fixture_50hh.h5"


@dataclass(frozen=True)
class DatasetRuntimeWorkspace:
    """Stage-shaped artifact set built from the tiny H5 fixture."""

    root: Path
    uprating_factors: Path
    acs: Path
    irs_puf: Path
    cps: Path
    puf: Path
    extended_cps: Path
    enhanced_cps: Path
    calibration_log: Path
    stratified_cps: Path
    source_imputed_cps: Path
    source_imputed_alias: Path
    small_enhanced_cps: Path
    sparse_enhanced_cps: Path


class RuntimeDataset:
    """PolicyEngine runtime view over one fixture-scale H5 artifact."""

    def __init__(self, path: Path):
        self.path = path
        self._simulation = Microsimulation(dataset=str(path))

    def values(self, variable: str, map_to: str) -> np.ndarray:
        return np.asarray(
            self._simulation.calculate(
                variable,
                period=int(PERIOD),
                map_to=map_to,
            )
        )


def build_dataset_runtime_workspace(tmp_path: Path) -> DatasetRuntimeWorkspace:
    """Create stage-shaped artifacts without running the production pipeline."""

    root = tmp_path / "dataset_runtime"
    root.mkdir()

    uprating_factors = root / "uprating_factors.csv"
    uprating_factors.write_text(
        "variable,year,factor\nemployment_income,2023,1.0\n",
        encoding="utf-8",
    )

    acs = _copy_fixture(root / "acs_2022.h5", phase="phase_1")
    irs_puf = _copy_fixture(root / "irs_puf_2015.h5", phase="phase_1")
    cps = _copy_fixture(root / "cps_2024.h5", phase="phase_2")
    puf = _copy_fixture(root / "puf_2024.h5", phase="phase_2")
    extended_cps = _copy_fixture(root / "extended_cps_2024.h5", phase="phase_3")
    enhanced_cps = _copy_fixture(root / "enhanced_cps_2024.h5", phase="phase_4")
    calibration_log = root / "calibration_log.csv"
    calibration_log.write_text(
        "stage,status\nfixture-scale,complete\n",
        encoding="utf-8",
    )
    stratified_cps = _copy_fixture(
        root / "stratified_extended_cps_2024.h5", phase="phase_4"
    )
    source_imputed_cps = _copy_fixture(
        root / "source_imputed_stratified_extended_cps_2024.h5",
        phase="phase_5",
    )
    source_imputed_alias = root / "source_imputed_stratified_extended_cps.h5"
    shutil.copyfile(source_imputed_cps, source_imputed_alias)
    small_enhanced_cps = _copy_fixture(
        root / "small_enhanced_cps_2024.h5", phase="phase_5"
    )
    sparse_enhanced_cps = _copy_fixture(
        root / "sparse_enhanced_cps_2024.h5", phase="phase_5"
    )

    return DatasetRuntimeWorkspace(
        root=root,
        uprating_factors=uprating_factors,
        acs=acs,
        irs_puf=irs_puf,
        cps=cps,
        puf=puf,
        extended_cps=extended_cps,
        enhanced_cps=enhanced_cps,
        calibration_log=calibration_log,
        stratified_cps=stratified_cps,
        source_imputed_cps=source_imputed_cps,
        source_imputed_alias=source_imputed_alias,
        small_enhanced_cps=small_enhanced_cps,
        sparse_enhanced_cps=sparse_enhanced_cps,
    )


def assert_file_pair_equal(left: Path, right: Path) -> None:
    assert left.exists()
    assert right.exists()
    assert filecmp.cmp(left, right, shallow=False)


def assert_entity_graph_is_consistent(path: Path) -> None:
    household_id = read_period_array(path, "household_id")
    person_household_id = read_period_array(path, "person_household_id")
    family_id = read_period_array(path, "family_id")
    person_family_id = read_period_array(path, "person_family_id")
    tax_unit_id = read_period_array(path, "tax_unit_id")
    person_tax_unit_id = read_period_array(path, "person_tax_unit_id")
    spm_unit_id = read_period_array(path, "spm_unit_id")
    person_spm_unit_id = read_period_array(path, "person_spm_unit_id")

    assert len(household_id) > 0
    assert len(person_household_id) > len(household_id)
    assert set(person_household_id).issubset(set(household_id))
    assert set(person_family_id).issubset(set(family_id))
    assert set(person_tax_unit_id).issubset(set(tax_unit_id))
    assert set(person_spm_unit_id).issubset(set(spm_unit_id))


def assert_has_period_arrays(path: Path, variables: tuple[str, ...]) -> None:
    with h5py.File(path, "r") as h5:
        for variable in variables:
            assert variable in h5
            assert PERIOD in h5[variable]
            assert h5[variable][PERIOD].shape[0] > 0


def assert_runtime_matches_h5(
    path: Path,
    variable: str,
    *,
    map_to: str,
) -> None:
    expected = read_period_array(path, variable)
    actual = RuntimeDataset(path).values(variable, map_to=map_to)
    assert actual.shape == expected.shape
    if np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    else:
        np.testing.assert_array_equal(actual, expected)


def assert_runtime_head_flags(path: Path) -> None:
    expected = read_period_array(path, "is_household_head")
    actual = RuntimeDataset(path).values("is_household_head", map_to="person")
    np.testing.assert_array_equal(actual, expected)
    assert int(actual.sum()) == entity_count(path, "household_id")


def assert_runtime_core_variables(path: Path) -> None:
    assert_runtime_matches_h5(path, "household_id", map_to="household")
    assert_runtime_matches_h5(path, "person_id", map_to="person")
    assert_runtime_matches_h5(path, "household_weight", map_to="household")
    assert_runtime_matches_h5(path, "employment_income", map_to="person")


def read_period_array(path: Path, variable: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        node = h5[variable]
        if isinstance(node, h5py.Dataset):
            return np.asarray(node[:])
        return np.asarray(node[PERIOD][:])


def entity_count(path: Path, variable: str) -> int:
    return len(read_period_array(path, variable))


def _copy_fixture(path: Path, *, phase: str) -> Path:
    shutil.copyfile(FIXTURE_PATH, path)
    with h5py.File(path, "a") as h5:
        h5.attrs["fixture_scale"] = "true"
        h5.attrs["dataset_runtime_phase"] = phase
    _write_household_head_flags(path)
    return path


def _write_household_head_flags(path: Path) -> None:
    person_household_id = read_period_array(path, "person_household_id")
    _, first_person_indices = np.unique(person_household_id, return_index=True)
    flags = np.zeros(len(person_household_id), dtype=np.bool_)
    flags[first_person_indices] = True
    _write_period_array(path, "is_household_head", flags)


def _write_period_array(path: Path, variable: str, values: np.ndarray) -> None:
    with h5py.File(path, "a") as h5:
        group = h5.require_group(variable)
        if PERIOD in group:
            del group[PERIOD]
        group.create_dataset(PERIOD, data=values)
