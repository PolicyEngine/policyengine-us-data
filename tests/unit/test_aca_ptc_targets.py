import pytest

from policyengine_us_data.storage.calibration_targets.aca_ptc_targets import (
    load_aca_ptc_state_targets,
)


def test_load_aca_ptc_state_targets_applies_latest_state_multipliers(tmp_path):
    target_dir = tmp_path / "calibration_targets"
    target_dir.mkdir()
    (target_dir / "aca_ptc_state.csv").write_text(
        "\n".join(
            [
                "# test",
                "GEO_ID,Returns,TotalPTCAmount",
                "0400000US06,10,100",
                "0400000US36,20,500",
            ]
        )
    )
    (tmp_path / "aca_ptc_multipliers_2022_2025.csv").write_text(
        "\n".join(
            [
                "state,enroll_2022,aptc_2022,enroll_2025,aptc_2025,vol_mult,val_mult",
                "CA,0,0,0,0,2,3",
                "New York,0,0,0,0,4,0.5",
            ]
        )
    )

    result = load_aca_ptc_state_targets(2026, storage_folder=tmp_path)

    assert result[["state", "Returns", "TotalPTCAmount"]].to_dict("records") == [
        {"state": "CA", "Returns": 20.0, "TotalPTCAmount": 600.0},
        {"state": "NY", "Returns": 80.0, "TotalPTCAmount": 1_000.0},
    ]
    assert result["uprating_year"].tolist() == [2025, 2025]


def test_load_aca_ptc_state_targets_returns_raw_soi_without_multipliers(
    tmp_path,
):
    target_dir = tmp_path / "calibration_targets"
    target_dir.mkdir()
    (target_dir / "aca_ptc_state.csv").write_text(
        "\n".join(
            [
                "# test",
                "GEO_ID,Returns,TotalPTCAmount",
                "0400000US06,10,100",
            ]
        )
    )

    result = load_aca_ptc_state_targets(2025, storage_folder=tmp_path)

    assert result.iloc[0].Returns == pytest.approx(10.0)
    assert result.iloc[0].TotalPTCAmount == pytest.approx(100.0)
    assert result.iloc[0].uprating_year == 2022
