"""Regression tests for the rent / real-estate-taxes dtype bug in
``policyengine_us_data.datasets.cps.cps.add_rent_and_real_estate_taxes``.

The original code wrote

    cps["rent"] = np.zeros_like(cps["age"])
    cps["rent"][mask] = imputed_values["rent"]

The integer dtype of ``cps["age"]`` propagated through ``zeros_like``,
so assigning QRF-imputed floats back into the array silently truncated
toward zero. The fix is ``np.zeros(len(cps["age"]), dtype=float)``.
"""

import ast
from pathlib import Path

import numpy as np

CPS_SOURCE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "policyengine_us_data"
    / "datasets"
    / "cps"
    / "cps.py"
)


def test_zeros_like_int_preserves_int_dtype_and_truncates_assignment():
    """Pin the bug premise: ``zeros_like`` on an int array keeps the
    int dtype, so subsequent float assignment is truncated."""
    age = np.array([10, 20, 30, 40], dtype=np.int64)
    rent_buggy = np.zeros_like(age)
    assert rent_buggy.dtype.kind == "i"
    rent_buggy[:] = np.array([123.4, 456.7, 789.1, 0.5])
    # Truncated toward zero (floors on positive values).
    assert rent_buggy.tolist() == [123, 456, 789, 0]


def test_zeros_len_dtype_float_preserves_imputed_values():
    """Verify the fix: ``np.zeros(len(age), dtype=float)`` keeps
    sub-integer precision after masked assignment."""
    age = np.array([10, 20, 30, 40], dtype=np.int64)
    rent = np.zeros(len(age), dtype=float)
    assert rent.dtype.kind == "f"
    mask = np.array([True, True, True, True])
    imputed = np.array([123.4, 456.7, 789.1, 0.5])
    rent[mask] = imputed
    np.testing.assert_allclose(rent, imputed)


def test_cps_source_does_not_use_zeros_like_age_for_rent_or_taxes():
    """Source-level invariant: the add_rent_and_real_estate_taxes
    function must not re-introduce ``np.zeros_like(cps["age"])`` for
    ``rent`` / ``real_estate_taxes``. We scan the AST for any
    assignment of the form ``cps["rent"] = np.zeros_like(cps["age"])``
    or the real_estate_taxes variant and assert there are none."""
    tree = ast.parse(CPS_SOURCE.read_text())

    def is_bad_rhs(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "zeros_like"
            and isinstance(func.value, ast.Name)
            and func.value.id == "np"
        ):
            return False
        if not node.args:
            return False
        arg = node.args[0]
        # ``cps["age"]`` -> Subscript(Name("cps"), Constant("age"))
        return (
            isinstance(arg, ast.Subscript)
            and isinstance(arg.value, ast.Name)
            and arg.value.id == "cps"
            and isinstance(arg.slice, ast.Constant)
            and arg.slice.value == "age"
        )

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if not is_bad_rhs(node.value):
                continue
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "cps"
                    and isinstance(tgt.slice, ast.Constant)
                    and tgt.slice.value in {"rent", "real_estate_taxes"}
                ):
                    offenders.append((tgt.slice.value, getattr(node, "lineno", "?")))

    assert offenders == [], (
        "np.zeros_like(cps['age']) re-introduced for rent/real_estate_taxes: "
        f"{offenders}"
    )
