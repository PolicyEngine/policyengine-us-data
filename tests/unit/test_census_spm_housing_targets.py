import importlib
import sys
import types

import pandas as pd
import validation.benefit_validation as benefit_validation

from policyengine_us_data.db import etl_national_targets
from policyengine_us_data.storage.calibration_targets.pull_hardcoded_targets import (
    pull_hardcoded_targets,
)
from policyengine_us_data.utils.census_spm import (
    build_census_spm_capped_housing_subsidy_target,
    get_census_spm_capped_housing_subsidy_total,
)
from policyengine_us_data.utils.hud_housing import (
    build_hud_user_housing_assistance_benchmark,
    get_hud_user_housing_benchmark,
)


def _write_census_cps(storage_dir, year=2024):
    path = storage_dir / f"census_cps_{year}.h5"
    with pd.HDFStore(path, mode="w") as store:
        store["spm_unit"] = pd.DataFrame(
            {
                "SPM_CAPHOUSESUB": [1_000.0, 0.0, 2_000.0],
                "SPM_WEIGHT": [150, 250, 100],
            }
        )
    return path


def test_get_census_spm_capped_housing_subsidy_total_uses_raw_cps_spm_values(
    tmp_path,
):
    _write_census_cps(tmp_path)

    total = get_census_spm_capped_housing_subsidy_total(2024, storage_folder=tmp_path)

    expected = (1_000 * 150 + 0 * 250 + 2_000 * 100) / 100
    assert total == expected


def test_pull_hardcoded_targets_uses_census_spm_housing_total(tmp_path):
    _write_census_cps(tmp_path)

    targets = pull_hardcoded_targets(year=2024, storage_folder=tmp_path)
    housing = targets.loc[
        targets.VARIABLE == "spm_unit_capped_housing_subsidy", "VALUE"
    ].iat[0]

    assert housing == (1_000 * 150 + 0 * 250 + 2_000 * 100) / 100


def test_extract_national_targets_uses_census_spm_housing_target(monkeypatch):
    class DummyMicrosimulation:
        def __init__(self, dataset):
            self.default_calculation_period = 2024

    policyengine_us = importlib.import_module("policyengine_us")
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        types.SimpleNamespace(
            **{
                name: getattr(policyengine_us, name)
                for name in dir(policyengine_us)
                if not name.startswith("__") and name != "Microsimulation"
            },
            Microsimulation=DummyMicrosimulation,
        ),
    )
    monkeypatch.setattr(
        etl_national_targets,
        "build_census_spm_capped_housing_subsidy_target",
        lambda year, storage_folder=None: {
            "variable": "spm_unit_capped_housing_subsidy",
            "value": 123.0,
            "source": "Census CPS ASEC public-use SPM_CAPHOUSESUB",
            "notes": "Capped SPM housing subsidy total from raw Census CPS ASEC.",
            "year": year,
        },
    )

    targets = etl_national_targets.extract_national_targets("dummy")
    housing = next(
        target
        for target in targets["direct_sum_targets"]
        if target["variable"] == "spm_unit_capped_housing_subsidy"
    )

    assert housing["value"] == 123.0
    assert housing["source"] == "Census CPS ASEC public-use SPM_CAPHOUSESUB"
    assert "Capped SPM housing subsidy" in housing["notes"]


def test_build_census_spm_capped_housing_subsidy_target_labels_concept(tmp_path):
    _write_census_cps(tmp_path)

    target = build_census_spm_capped_housing_subsidy_target(
        2024, storage_folder=tmp_path
    )

    assert target["variable"] == "spm_unit_capped_housing_subsidy"
    assert target["value"] == (1_000 * 150 + 0 * 250 + 2_000 * 100) / 100
    assert target["source"] == "Census CPS ASEC public-use SPM_CAPHOUSESUB"
    assert "not HUD spending" in target["notes"]


def test_get_hud_user_housing_benchmark_uses_annual_spending_concept():
    benchmark = get_hud_user_housing_benchmark(2022)

    assert benchmark["reported_households"] == 4_537_614
    assert benchmark["average_monthly_spending_per_unit"] == 899
    assert benchmark["annual_spending_total"] == 48_951_779_832


def test_build_hud_user_housing_assistance_benchmark_labels_admin_concept():
    benchmark = build_hud_user_housing_assistance_benchmark(2022)

    assert benchmark["variable"] == "housing_assistance"
    assert benchmark["source"] == "HUD USER Picture of Subsidized Households"
    assert benchmark["reported_households"] == 4_537_614
    assert benchmark["annual_spending_total"] == 48_951_779_832
    assert "not Census SPM capped subsidy" in benchmark["notes"]


def test_get_program_benchmarks_keeps_housing_concepts_separate(monkeypatch):
    monkeypatch.setattr(
        benefit_validation,
        "build_census_spm_capped_housing_subsidy_target",
        lambda year: {
            "variable": "spm_unit_capped_housing_subsidy",
            "value": 111e9,
            "source": "Census benchmark",
            "notes": "Census concept",
            "year": year,
        },
    )
    monkeypatch.setattr(
        benefit_validation,
        "build_hud_user_housing_assistance_benchmark",
        lambda year: {
            "variable": "housing_assistance",
            "annual_spending_total": 222e9,
            "reported_households": 3_000_000,
            "average_monthly_spending_per_unit": 999,
            "source": "HUD USER benchmark",
            "notes": "HUD concept",
            "year": year,
        },
    )

    benchmarks = benefit_validation.get_program_benchmarks(2022)

    assert (
        benchmarks["housing_spm_capped"]["variable"]
        == "spm_unit_capped_housing_subsidy"
    )
    assert benchmarks["housing_spm_capped"]["benchmark_total"] == 111
    assert benchmarks["housing_spm_capped"]["benchmark_source"] == "Census benchmark"

    assert benchmarks["housing_assistance_hud_user"]["variable"] == "housing_assistance"
    assert benchmarks["housing_assistance_hud_user"]["benchmark_total"] == 222
    assert (
        benchmarks["housing_assistance_hud_user"]["benchmark_source"]
        == "HUD USER benchmark"
    )
    assert (
        benchmarks["housing_assistance_hud_user"]["benchmark_participants_millions"]
        == 3
    )
    assert benchmarks["housing_assistance_hud_user"]["map_to"] == "household"
