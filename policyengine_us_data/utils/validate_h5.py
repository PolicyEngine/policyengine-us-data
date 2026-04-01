"""Pre-publish validation for H5 dataset files.

Checks entity dimension consistency and weight sanity before upload.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from policyengine_us_data.utils.downsample import ENTITY_ID_VARIABLES


def validate_h5_entity_dimensions(
    h5_path: str | Path, period: int = 2024
) -> list[dict]:
    """Validate that every variable in the H5 has the correct entity length.

    Args:
        h5_path: Path to an H5 dataset file.
        period: Tax year key inside the H5 (top-level group).

    Returns:
        List of ``{check, status, detail}`` dicts.
    """
    from policyengine_us import CountryTaxBenefitSystem

    tbs = CountryTaxBenefitSystem()
    results: list[dict] = []
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        group = f[str(period)]
        variable_names = list(group.keys())

        entity_counts: dict[str, int] = {}
        for entity_key, id_var in ENTITY_ID_VARIABLES.items():
            if id_var in group:
                entity_counts[entity_key] = len(group[id_var])

        # Dimension checks
        for var_name in variable_names:
            variable_meta = tbs.variables.get(var_name)
            if variable_meta is None:
                continue
            entity_key = getattr(getattr(variable_meta, "entity", None), "key", None)
            expected = entity_counts.get(entity_key)
            if expected is None:
                continue
            actual = len(group[var_name])
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

        # household_weight existence
        if "household_weight" not in group:
            results.append(
                {
                    "check": "household_weight_exists",
                    "status": "FAIL",
                    "detail": "household_weight not found in H5",
                }
            )
        else:
            weights = np.asarray(group["household_weight"])
            if np.all(weights == 0):
                results.append(
                    {
                        "check": "household_weight_nonzero",
                        "status": "FAIL",
                        "detail": "all household_weight values are zero",
                    }
                )

        # Reasonable household count
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
