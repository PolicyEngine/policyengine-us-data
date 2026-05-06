"""ACA premium tax credit state calibration targets."""

from pathlib import Path

import pandas as pd

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.storage.calibration_targets.pull_soi_targets import (
    STATE_ABBR_TO_FIPS,
)


def _state_from_label(label: str) -> str | None:
    import us

    label = str(label).strip()
    if label.upper() in STATE_ABBR_TO_FIPS:
        return label.upper()
    state = us.states.lookup(label)
    if state is None:
        return None
    return state.abbr


def _available_multiplier_paths(storage_folder: Path) -> dict[int, Path]:
    candidates = {}
    for path in storage_folder.glob("aca_ptc_multipliers_2022_*.csv"):
        suffix = path.stem.removeprefix("aca_ptc_multipliers_2022_")
        if suffix.isdigit():
            candidates[int(suffix)] = path
    return candidates


def _load_multiplier_path(
    period: int,
    storage_folder: Path,
) -> tuple[pd.DataFrame | None, int | None]:
    candidates = _available_multiplier_paths(storage_folder)
    if not candidates:
        return None, None

    eligible_years = [year for year in candidates if year <= period]
    if eligible_years:
        year = max(eligible_years)
    else:
        year = min(candidates)
    multipliers = pd.read_csv(candidates[year])
    multipliers["state"] = multipliers["state"].map(_state_from_label)
    multipliers = multipliers.dropna(subset=["state"])
    return multipliers, year


def load_aca_ptc_state_targets(
    period: int,
    storage_folder: Path | None = None,
) -> pd.DataFrame | None:
    """Load SOI ACA PTC state targets, uprated when multipliers are available."""
    storage_folder = STORAGE_FOLDER if storage_folder is None else Path(storage_folder)
    target_path = storage_folder / "calibration_targets" / "aca_ptc_state.csv"
    if not target_path.exists():
        return None

    targets = pd.read_csv(target_path, comment="#")
    state_by_fips = {fips: state for state, fips in STATE_ABBR_TO_FIPS.items()}
    targets["FIPS"] = targets["GEO_ID"].astype(str).str[-2:]
    targets["state"] = targets["FIPS"].map(state_by_fips)
    targets = targets.dropna(subset=["state"]).copy()
    targets["source_year"] = 2022
    targets["uprating_year"] = 2022
    targets["Returns"] = targets["Returns"].astype(float)
    targets["TotalPTCAmount"] = targets["TotalPTCAmount"].astype(float)

    multipliers, multiplier_year = _load_multiplier_path(period, storage_folder)
    if multipliers is None:
        return targets

    targets = targets.merge(
        multipliers[["state", "vol_mult", "val_mult"]],
        on="state",
        how="left",
    )
    has_multiplier = targets["vol_mult"].notna() & targets["val_mult"].notna()
    targets.loc[has_multiplier, "Returns"] *= targets.loc[has_multiplier, "vol_mult"]
    targets.loc[has_multiplier, "TotalPTCAmount"] *= (
        targets.loc[has_multiplier, "vol_mult"]
        * targets.loc[has_multiplier, "val_mult"]
    )
    targets.loc[has_multiplier, "uprating_year"] = int(multiplier_year)
    return targets.drop(columns=["vol_mult", "val_mult"])
