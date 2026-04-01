"""Pre-publish validation for H5 dataset files.

Checks entity dimension consistency and weight sanity before upload.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from policyengine_us_data.utils.downsample import ENTITY_ID_VARIABLES


def _read_array(f: h5py.File, var_name: str, period: int):
    """Read a variable array, handling both H5 layouts.

    Pipeline-built files use ``variable/period`` nesting (groups at top level,
    datasets underneath keyed by year).  Storage flat files store datasets
    directly at the top level with no period sub-key.

    Returns None if the variable is not found.
    """
    if var_name not in f:
        return None
    item = f[var_name]
    if isinstance(item, h5py.Dataset):
        return item
    # Group — look for period sub-key
    period_key = str(period)
    if period_key in item:
        return item[period_key]
    return None


def validate_h5_entity_dimensions(
    h5_path: str | Path, period: int = 2024
) -> list[dict]:
    """Validate that every variable in the H5 has the correct entity length.

    Args:
        h5_path: Path to an H5 dataset file.
        period: Tax year key inside the H5.

    Returns:
        List of ``{check, status, detail}`` dicts (empty means all OK).
    """
    from policyengine_us import CountryTaxBenefitSystem

    tbs = CountryTaxBenefitSystem()
    results: list[dict] = []
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        variable_names = list(f.keys())

        entity_counts: dict[str, int] = {}
        for entity_key, id_var in ENTITY_ID_VARIABLES.items():
            arr = _read_array(f, id_var, period)
            if arr is not None:
                entity_counts[entity_key] = len(arr)

        for var_name in variable_names:
            variable_meta = tbs.variables.get(var_name)
            if variable_meta is None:
                continue
            entity_key = getattr(getattr(variable_meta, "entity", None), "key", None)
            expected = entity_counts.get(entity_key)
            if expected is None:
                continue
            arr = _read_array(f, var_name, period)
            if arr is None:
                continue
            actual = len(arr)
            if actual != expected:
                results.append(
                    {
                        "check": "dimension",
                        "status": "FAIL",
                        "detail": (
                            f"{var_name} ({entity_key}): "
                            f"expected {expected}, got {actual}"
                        ),
                    }
                )

        # household_weight existence and sanity
        hw = _read_array(f, "household_weight", period)
        if hw is None:
            results.append(
                {
                    "check": "household_weight_exists",
                    "status": "FAIL",
                    "detail": "household_weight not found in H5",
                }
            )
        else:
            if np.all(np.asarray(hw) == 0):
                results.append(
                    {
                        "check": "household_weight_nonzero",
                        "status": "FAIL",
                        "detail": "all household_weight values are zero",
                    }
                )

        hh_count = entity_counts.get("household", 0)
        if hh_count == 0:
            results.append(
                {
                    "check": "household_count",
                    "status": "FAIL",
                    "detail": "household count is zero",
                }
            )

    return results


def validate_h5_or_raise(
    h5_path: str | Path, label: str = "", period: int = 2024
) -> None:
    """Run all H5 validations and raise on any failure.

    Args:
        h5_path: Path to the H5 file.
        label: Optional label for error messages.
        period: Tax year key inside the H5.

    Raises:
        ValueError: If any validation check fails.
    """
    failures = validate_h5_entity_dimensions(h5_path, period=period)
    if failures:
        tag = f" [{label}]" if label else ""
        lines = [f"H5 validation failed{tag} for {h5_path}:"]
        for f in failures:
            lines.append(f"  {f['check']}: {f['detail']}")
        raise ValueError("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <h5_path> [period]", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    yr = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    issues = validate_h5_entity_dimensions(path, period=yr)
    if issues:
        for issue in issues:
            print(f"[{issue['status']}] {issue['check']}: {issue['detail']}")
        sys.exit(1)
    else:
        print(f"All checks passed for {path}")
