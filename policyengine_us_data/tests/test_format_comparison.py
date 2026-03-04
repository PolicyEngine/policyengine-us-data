"""
Compare h5py (variable-centric) and HDFStore (entity-level) output formats.

Verifies that both formats produced by stacked_dataset_builder contain
identical data for all variables.

Usage as pytest:
    pytest test_format_comparison.py --h5py-path path/to/STATE.h5 \
                                     --hdfstore-path path/to/STATE.hdfstore.h5

Usage as standalone script:
    python -m policyengine_us_data.tests.test_format_comparison \
        --h5py-path path/to/STATE.h5 \
        --hdfstore-path path/to/STATE.hdfstore.h5
"""

import argparse
import sys

import h5py
import numpy as np
import pandas as pd
import pytest


def compare_formats(h5py_path: str, hdfstore_path: str) -> dict:
    """Compare all variables between h5py and HDFStore formats.

    Returns a dict with keys: passed, failed, skipped, details.
    """
    passed = []
    failed = []
    skipped = []

    with h5py.File(h5py_path, "r") as f:
        h5_vars = sorted(f.keys())
        # Get the year from the first variable's subkeys
        first_var = h5_vars[0]
        year = list(f[first_var].keys())[0]

        with pd.HDFStore(hdfstore_path, "r") as store:
            # Load all entity DataFrames
            store_keys = [k for k in store.keys() if not k.startswith("/_")]
            entity_dfs = {k: store[k] for k in store_keys}

            # Load manifest
            manifest = None
            if "/_variable_metadata" in store.keys():
                manifest = store["/_variable_metadata"]

            for var in h5_vars:
                h5_values = f[var][year][:]

                # Find which entity DataFrame contains this variable
                found = False
                for entity_key, df in entity_dfs.items():
                    entity_name = entity_key.lstrip("/")
                    if var in df.columns:
                        hdf_values = df[var].values

                        # For person-level variables, arrays should be
                        # same length and directly comparable (both are
                        # ordered by row index from combined_df).
                        # For group entities, the h5py array is at person
                        # level while HDFStore is deduplicated.  We need
                        # to handle this difference.
                        if entity_name != "person" and len(hdf_values) != len(
                            h5_values
                        ):
                            # h5py stores at person level; HDFStore is
                            # deduplicated by entity ID.  We can't do a
                            # direct comparison — verify unique values match.
                            h5_unique = np.unique(h5_values)
                            hdf_unique = np.unique(hdf_values)
                            if h5_values.dtype.kind in ("U", "S", "O"):
                                match = set(h5_unique) == set(hdf_unique)
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
                                # String comparison
                                h5_str = np.array(
                                    [
                                        (
                                            x.decode()
                                            if isinstance(x, bytes)
                                            else str(x)
                                        )
                                        for x in h5_values
                                    ]
                                )
                                hdf_str = np.array(
                                    [str(x) for x in hdf_values]
                                )
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
                                # Numeric comparison
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


def pytest_addoption(parser):
    parser.addoption("--h5py-path", action="store", default=None)
    parser.addoption("--hdfstore-path", action="store", default=None)


@pytest.fixture
def h5py_path(request):
    path = request.config.getoption("--h5py-path")
    if path is None:
        pytest.skip("--h5py-path not provided")
    return path


@pytest.fixture
def hdfstore_path(request):
    path = request.config.getoption("--hdfstore-path")
    if path is None:
        pytest.skip("--hdfstore-path not provided")
    return path


def test_formats_match(h5py_path, hdfstore_path):
    """Verify h5py and HDFStore formats contain identical data."""
    result = compare_formats(h5py_path, hdfstore_path)

    print(f"\n{'='*60}")
    print(f"Format Comparison Results")
    print(f"{'='*60}")
    print(f"Total h5py variables: {result['total_h5py_vars']}")
    print(f"Passed: {len(result['passed'])}")
    print(f"Failed: {len(result['failed'])}")
    print(f"Skipped (not in HDFStore): {len(result['skipped'])}")

    if result["failed"]:
        print(f"\nFailed variables:")
        for var, reason in result["failed"]:
            print(f"  {var}: {reason}")

    if result["skipped"]:
        print(f"\nSkipped variables (not found in HDFStore):")
        for var in result["skipped"]:
            print(f"  {var}")

    assert len(result["failed"]) == 0, (
        f"{len(result['failed'])} variables have mismatched values"
    )
    assert len(result["skipped"]) == 0, (
        f"{len(result['skipped'])} variables missing from HDFStore"
    )


def test_manifest_present(hdfstore_path):
    """Verify the HDFStore contains a variable metadata manifest."""
    with pd.HDFStore(hdfstore_path, "r") as store:
        assert "/_variable_metadata" in store.keys(), (
            "Missing _variable_metadata table"
        )
        manifest = store["/_variable_metadata"]
        assert "variable" in manifest.columns
        assert "entity" in manifest.columns
        assert "uprating" in manifest.columns
        assert len(manifest) > 0, "Manifest is empty"
        print(f"\nManifest has {len(manifest)} variables")
        print(f"Entities: {manifest['entity'].unique().tolist()}")
        n_uprated = (manifest["uprating"] != "").sum()
        print(f"Variables with uprating: {n_uprated}")


def test_all_entities_present(hdfstore_path):
    """Verify the HDFStore contains all expected entity tables."""
    expected = {"person", "household", "tax_unit", "spm_unit", "family", "marital_unit"}
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare h5py and HDFStore dataset formats"
    )
    parser.add_argument(
        "--h5py-path", required=True, help="Path to h5py format file"
    )
    parser.add_argument(
        "--hdfstore-path", required=True, help="Path to HDFStore format file"
    )
    args = parser.parse_args()

    result = compare_formats(args.h5py_path, args.hdfstore_path)

    print(f"\n{'='*60}")
    print(f"Format Comparison Results")
    print(f"{'='*60}")
    print(f"Total h5py variables: {result['total_h5py_vars']}")
    print(f"Passed: {len(result['passed'])}")
    print(f"Failed: {len(result['failed'])}")
    print(f"Skipped (not in HDFStore): {len(result['skipped'])}")

    if result["failed"]:
        print(f"\nFailed variables:")
        for var, reason in result["failed"]:
            print(f"  {var}: {reason}")

    if result["skipped"]:
        print(f"\nSkipped variables (not found in HDFStore):")
        for var in result["skipped"]:
            print(f"  {var}")

    if result["failed"] or result["skipped"]:
        sys.exit(1)
    else:
        print("\nAll variables match!")
        sys.exit(0)
