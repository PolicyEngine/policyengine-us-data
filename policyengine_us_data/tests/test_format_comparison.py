"""
ONE-OFF VALIDATION SCRIPT

This is a one-off script used to verify that the h5py-to-HDFStore
conversion logic is correct. It reads an existing h5py dataset file,
converts it to entity-level Pandas HDFStore using the production
splitting/dedup logic, then compares all variables to verify the
conversion is lossless.

This script is NOT part of the regular test suite and is not intended to
be run in CI. It exists to validate the HDFStore serialization logic
during development.

Usage (run directly to avoid policyengine_us_data __init__ imports):
    python policyengine_us_data/tests/test_format_comparison.py \
        --h5py-path path/to/STATE.h5
"""

import argparse
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.utils.hdfstore import (
    ENTITIES,
    DatasetResult,
    _save_hdfstore,
)


def _load_system():
    """Load the policyengine-us tax-benefit system."""
    from policyengine_us import system as us_system

    return us_system.system


# ---------------------------------------------------------------------------
# h5py reading helpers (test-specific; reads the flat h5py format
# and wraps it into the nested {var: {period: array}} structure
# expected by the production HDFStore utilities)
# ---------------------------------------------------------------------------


def _read_h5py_arrays(h5py_path: str):
    """Read all arrays from an h5py variable-centric file.

    The h5py format stores ``variable / period -> array``.  Periods can be
    yearly (``"2024"``), monthly (``"2024-01"``), or ``"ETERNITY"``.

    Returns ``(data, time_period, h5_vars)`` where *data* is a nested dict
    ``{variable_name: {period_key: numpy_array}}`` matching the format
    used by the production HDFStore utilities.
    """
    with h5py.File(h5py_path, "r") as f:
        h5_vars = sorted(f.keys())

        # Determine the canonical year from the first variable that has one
        year = None
        for var in h5_vars:
            subkeys = list(f[var].keys())
            for sk in subkeys:
                if sk.isdigit() and len(sk) == 4:
                    year = sk
                    break
            if year is not None:
                break
        if year is None:
            raise ValueError("Could not determine year from h5py file")

        time_period = int(year)
        data = {}

        for var in h5_vars:
            subkeys = list(f[var].keys())
            if year in subkeys:
                period_key = year
            elif "ETERNITY" in subkeys:
                period_key = "ETERNITY"
            else:
                period_key = subkeys[0]

            arr = f[var][period_key][:]
            if arr.dtype.kind in ("S", "O"):
                arr = np.array(
                    [x.decode() if isinstance(x, bytes) else str(x) for x in arr]
                )
            # Wrap in nested dict keyed by the period string
            data[var] = {period_key: arr}

    return data, time_period, h5_vars


# ---------------------------------------------------------------------------
# Main conversion + comparison logic
# ---------------------------------------------------------------------------


def h5py_to_hdfstore(h5py_path: str, hdfstore_path: str) -> dict:
    """Convert an h5py variable-centric file to entity-level HDFStore.

    Uses the production HDFStore utilities so this test validates the
    real code path rather than a local reimplementation.

    Returns a summary dict with entity row counts.
    """
    print("Loading policyengine-us system (this takes a minute)...")
    system = _load_system()

    print("Reading h5py file...")
    data, time_period, h5_vars = _read_h5py_arrays(h5py_path)
    n_persons = len(next(iter(data.get("person_id", {}).values()), []))
    print(f"  {len(h5_vars)} variables, {n_persons:,} persons, year={time_period}")

    result = DatasetResult(data=data, time_period=time_period, system=system)
    output_base = hdfstore_path.replace(".hdfstore.h5", "")

    print(f"Saving HDFStore to {hdfstore_path}...")
    _save_hdfstore(result, output_base)

    summary = {}
    for entity_name, df in entity_dfs.items():
        summary[entity_name] = {"rows": len(df), "cols": len(df.columns)}
    summary["manifest_vars"] = len(manifest_df)
    return summary


def compare_formats(h5py_path: str, hdfstore_path: str) -> dict:
    """Compare all variables between h5py and generated HDFStore.

    Returns a dict with keys: passed, failed, skipped.
    """
    passed = []
    failed = []
    skipped = []

    with h5py.File(h5py_path, "r") as f:
        h5_vars = sorted(f.keys())

        # Determine the year
        year = None
        for var in h5_vars:
            for sk in f[var].keys():
                if sk.isdigit() and len(sk) == 4:
                    year = sk
                    break
            if year is not None:
                break

        with pd.HDFStore(hdfstore_path, "r") as store:
            store_keys = [k for k in store.keys() if not k.startswith("/_")]
            entity_dfs = {k: store[k] for k in store_keys}

            for var in h5_vars:
                subkeys = list(f[var].keys())
                if year in subkeys:
                    period_key = year
                elif "ETERNITY" in subkeys:
                    period_key = "ETERNITY"
                else:
                    period_key = subkeys[0]

                h5_values = f[var][period_key][:]

                found = False
                for entity_key, df in entity_dfs.items():
                    entity_name = entity_key.lstrip("/")
                    if var in df.columns:
                        hdf_values = df[var].values

                        # For group entities, h5py is person-level while
                        # HDFStore is deduplicated by entity ID.
                        if entity_name != "person" and len(hdf_values) != len(
                            h5_values
                        ):
                            h5_unique = np.unique(h5_values)
                            hdf_unique = np.unique(hdf_values)
                            if h5_values.dtype.kind in ("U", "S", "O"):
                                match = set(
                                    (x.decode() if isinstance(x, bytes) else str(x))
                                    for x in h5_unique
                                ) == set(str(x) for x in hdf_unique)
                            else:
                                match = np.allclose(
                                    np.sort(h5_unique.astype(float)),
                                    np.sort(hdf_unique.astype(float)),
                                    rtol=1e-5,
                                    equal_nan=True,
                                )
                            if match:
                                passed.append(var)
                            else:
                                failed.append(
                                    (
                                        var,
                                        f"unique values differ "
                                        f"(h5py: {len(h5_unique)}, "
                                        f"hdfstore: {len(hdf_unique)})",
                                    )
                                )
                        else:
                            # Same length — direct comparison
                            if h5_values.dtype.kind in ("U", "S", "O"):
                                h5_str = np.array(
                                    [
                                        (x.decode() if isinstance(x, bytes) else str(x))
                                        for x in h5_values
                                    ]
                                )
                                hdf_str = np.array([str(x) for x in hdf_values])
                                if np.array_equal(h5_str, hdf_str):
                                    passed.append(var)
                                else:
                                    mismatches = np.sum(h5_str != hdf_str)
                                    failed.append(
                                        (
                                            var,
                                            f"{mismatches} string mismatches",
                                        )
                                    )
                            else:
                                h5_float = h5_values.astype(float)
                                hdf_float = hdf_values.astype(float)
                                if np.allclose(
                                    h5_float,
                                    hdf_float,
                                    rtol=1e-5,
                                    equal_nan=True,
                                ):
                                    passed.append(var)
                                else:
                                    diff = np.abs(h5_float - hdf_float)
                                    max_diff = np.max(diff)
                                    n_diff = np.sum(
                                        ~np.isclose(
                                            h5_float,
                                            hdf_float,
                                            rtol=1e-5,
                                            equal_nan=True,
                                        )
                                    )
                                    failed.append(
                                        (
                                            var,
                                            f"{n_diff} values differ, "
                                            f"max diff={max_diff:.6f}",
                                        )
                                    )
                        found = True
                        break

                if not found:
                    skipped.append(var)

    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total_h5py_vars": len(h5_vars),
    }


def print_results(result):
    """Print comparison results to stdout."""
    print(f"\n{'=' * 60}")
    print("Format Comparison Results")
    print(f"{'=' * 60}")
    print(f"Total h5py variables: {result['total_h5py_vars']}")
    print(f"Passed: {len(result['passed'])}")
    print(f"Failed: {len(result['failed'])}")
    print(f"Skipped (not in HDFStore): {len(result['skipped'])}")

    if result["failed"]:
        print("\nFailed variables:")
        for var, reason in result["failed"]:
            print(f"  {var}: {reason}")

    if result["skipped"]:
        print("\nSkipped variables (not found in HDFStore):")
        for var in result["skipped"]:
            print(f"  {var}")


# --- pytest interface ---


def pytest_addoption(parser):
    parser.addoption("--h5py-path", action="store", default=None)


@pytest.fixture
def h5py_path(request):
    path = request.config.getoption("--h5py-path")
    if path is None:
        pytest.skip("--h5py-path not provided")
    return path


def test_roundtrip(h5py_path, tmp_path):
    """Convert h5py -> HDFStore -> compare all variables."""
    hdfstore_path = str(tmp_path / "test_output.hdfstore.h5")

    summary = h5py_to_hdfstore(h5py_path, hdfstore_path)
    for entity, info in summary.items():
        if isinstance(info, dict):
            print(f"  {entity}: {info['rows']:,} rows, {info['cols']} cols")

    result = compare_formats(h5py_path, hdfstore_path)
    print_results(result)

    assert len(result["failed"]) == 0, (
        f"{len(result['failed'])} variables have mismatched values"
    )
    assert len(result["skipped"]) == 0, (
        f"{len(result['skipped'])} variables missing from HDFStore"
    )


def test_manifest(h5py_path, tmp_path):
    """Verify the generated HDFStore contains a valid manifest."""
    hdfstore_path = str(tmp_path / "test_output.hdfstore.h5")
    h5py_to_hdfstore(h5py_path, hdfstore_path)

    with pd.HDFStore(hdfstore_path, "r") as store:
        assert "/_variable_metadata" in store.keys(), "Missing _variable_metadata table"
        manifest = store["/_variable_metadata"]
        assert "variable" in manifest.columns
        assert "entity" in manifest.columns
        assert "uprating" in manifest.columns
        assert len(manifest) > 0, "Manifest is empty"
        print(f"\nManifest has {len(manifest)} variables")
        print(f"Entities: {manifest['entity'].unique().tolist()}")
        n_uprated = (manifest["uprating"] != "").sum()
        print(f"Variables with uprating: {n_uprated}")


def test_all_entities(h5py_path, tmp_path):
    """Verify the generated HDFStore contains all expected entity tables."""
    hdfstore_path = str(tmp_path / "test_output.hdfstore.h5")
    h5py_to_hdfstore(h5py_path, hdfstore_path)

    expected = set(ENTITIES)
    with pd.HDFStore(hdfstore_path, "r") as store:
        actual = {k.lstrip("/") for k in store.keys() if not k.startswith("/_")}
        missing = expected - actual
        assert not missing, f"Missing entity tables: {missing}"
        for entity in expected:
            df = store[f"/{entity}"]
            assert len(df) > 0, f"Entity {entity} has 0 rows"
            assert f"{entity}_id" in df.columns, (
                f"Entity {entity} missing {entity}_id column"
            )
            print(f"  {entity}: {len(df):,} rows, {len(df.columns)} cols")


# --- CLI interface ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert h5py dataset to HDFStore and verify roundtrip"
    )
    parser.add_argument("--h5py-path", required=True, help="Path to h5py format file")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Path for generated HDFStore (default: alongside input file)",
    )
    args = parser.parse_args()

    if args.output_path:
        hdfstore_path = args.output_path
    else:
        hdfstore_path = args.h5py_path.replace(".h5", ".hdfstore.h5")

    print(f"Converting {args.h5py_path} -> {hdfstore_path}...")
    summary = h5py_to_hdfstore(args.h5py_path, hdfstore_path)
    for entity, info in summary.items():
        if isinstance(info, dict):
            print(f"  {entity}: {info['rows']:,} rows, {info['cols']} cols")

    print("\nComparing formats...")
    result = compare_formats(args.h5py_path, hdfstore_path)
    print_results(result)

    if result["failed"] or result["skipped"]:
        sys.exit(1)
    else:
        print("\nAll variables match!")
        sys.exit(0)
