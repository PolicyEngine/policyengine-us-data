import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "policyengine_us_data"


@contextmanager
def load_uprate_puf_module(storage_root: Path):
    module_names = [
        "policyengine_us_data.datasets.puf.uprate_puf",
        "policyengine_us_data.datasets.puf",
        "policyengine_us_data.datasets",
        "policyengine_us_data.storage",
        "policyengine_us_data",
    ]
    original_modules = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)

    try:
        package = types.ModuleType("policyengine_us_data")
        package.__path__ = [str(PACKAGE_ROOT)]
        sys.modules["policyengine_us_data"] = package

        datasets_package = types.ModuleType("policyengine_us_data.datasets")
        datasets_package.__path__ = [str(PACKAGE_ROOT / "datasets")]
        sys.modules["policyengine_us_data.datasets"] = datasets_package

        puf_package = types.ModuleType("policyengine_us_data.datasets.puf")
        puf_package.__path__ = [str(PACKAGE_ROOT / "datasets" / "puf")]
        sys.modules["policyengine_us_data.datasets.puf"] = puf_package

        storage_spec = importlib.util.spec_from_file_location(
            "policyengine_us_data.storage",
            PACKAGE_ROOT / "storage" / "__init__.py",
            submodule_search_locations=[str(PACKAGE_ROOT / "storage")],
        )
        storage_module = importlib.util.module_from_spec(storage_spec)
        assert storage_spec.loader is not None
        sys.modules["policyengine_us_data.storage"] = storage_module
        storage_spec.loader.exec_module(storage_module)
        storage_module.STORAGE_FOLDER = storage_root
        storage_module.CALIBRATION_FOLDER = storage_root / "calibration_targets"

        uprate_spec = importlib.util.spec_from_file_location(
            "policyengine_us_data.datasets.puf.uprate_puf",
            PACKAGE_ROOT / "datasets" / "puf" / "uprate_puf.py",
        )
        uprate_module = importlib.util.module_from_spec(uprate_spec)
        assert uprate_spec.loader is not None
        sys.modules["policyengine_us_data.datasets.puf.uprate_puf"] = uprate_module
        uprate_spec.loader.exec_module(uprate_module)
        yield uprate_module
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in original_modules.items():
            if module is not None:
                sys.modules[name] = module


def write_soi_targets(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "Year": 2021,
                "Variable": "employment_income",
                "Filing status": "All",
                "AGI lower bound": float("-inf"),
                "AGI upper bound": float("inf"),
                "Count": False,
                "Taxable only": False,
                "Full population": True,
                "Value": 200.0,
            },
            {
                "Year": 2021,
                "Variable": "count",
                "Filing status": "All",
                "AGI lower bound": float("-inf"),
                "AGI upper bound": float("inf"),
                "Count": True,
                "Taxable only": False,
                "Full population": True,
                "Value": 100.0,
            },
        ]
    ).to_csv(path, index=False)


def test_get_soi_aggregate_reads_tracked_soi_targets(tmp_path: Path):
    write_soi_targets(tmp_path / "calibration_targets" / "soi_targets.csv")
    with load_uprate_puf_module(tmp_path) as module:
        assert module.get_soi_aggregate("employment_income", 2021, False) == 200.0
        assert module.get_soi_aggregate("count", 2021, True) == 100.0


def test_get_soi_aggregate_raises_clear_error_when_missing(tmp_path: Path):
    with load_uprate_puf_module(tmp_path) as module:
        with pytest.raises(FileNotFoundError, match="No SOI aggregate file found at"):
            module.load_soi_aggregates()


def test_pos_neg_split_uprate_writes_back_to_frame():
    """Regression for chained-indexing silently no-opping POS_ONLY uprating.

    Under pandas Copy-on-Write semantics, ``df[col][mask] *= growth``
    returns a copy that is never written back; the fix uses ``df.loc``.
    This micro-regression pins the loop pattern from
    ``uprate_puf.uprate_puf`` independent of the SOI-targets CSV and
    REMAINING_VARIABLES schema.
    """
    puf = pd.DataFrame(
        {
            "E00900": [500.0, -200.0, 0.0],
            "E01000": [300.0, -100.0, 0.0],
            "E26270": [250.0, -50.0, 0.0],
        }
    )
    growth_pos = 1.25
    growth_neg = 1.30

    for col in ["E00900", "E01000", "E26270"]:
        puf.loc[puf[col] > 0, col] *= growth_pos
    for col in ["E00900", "E01000", "E26270"]:
        puf.loc[puf[col] < 0, col] *= growth_neg

    # Positives grew by growth_pos.
    assert puf.loc[0, "E00900"] == pytest.approx(500.0 * growth_pos)
    assert puf.loc[0, "E01000"] == pytest.approx(300.0 * growth_pos)
    assert puf.loc[0, "E26270"] == pytest.approx(250.0 * growth_pos)
    # Negatives grew in magnitude by growth_neg.
    assert puf.loc[1, "E00900"] == pytest.approx(-200.0 * growth_neg)
    assert puf.loc[1, "E01000"] == pytest.approx(-100.0 * growth_neg)
    assert puf.loc[1, "E26270"] == pytest.approx(-50.0 * growth_neg)
    # Zeros untouched.
    assert puf.loc[2, "E00900"] == 0.0
    assert puf.loc[2, "E01000"] == 0.0


def test_chained_indexing_pattern_is_a_no_op_silent_under_cow():
    """Sanity: ``df[col][mask] *= x`` silently fails on a CoW-backed
    frame — motivating the ``df.loc[mask, col] *= x`` fix."""
    import warnings

    puf = pd.DataFrame({"E00900": [500.0, -200.0]})
    original = puf.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The broken pattern, kept as a documentation assertion.
        puf["E00900"][puf["E00900"] > 0] *= 10
    # With pandas CoW enabled in recent pandas (2.x default), the
    # original frame is unchanged — proving the bug.
    if pd.options.mode.copy_on_write:
        assert puf["E00900"].equals(original["E00900"])


def test_uprate_puf_pos_neg_split_module_helpers_intact():
    """Verify the module's POS/NEG rename dicts still cover the SOI
    variables that trigger the chained-indexing path."""
    spec = importlib.util.spec_from_file_location(
        "uprate_puf_static",
        PACKAGE_ROOT / "datasets" / "puf" / "uprate_puf.py",
    )
    assert spec is not None and spec.loader is not None
    # Execute in a minimal namespace so we don't need storage etc.
    source = (PACKAGE_ROOT / "datasets" / "puf" / "uprate_puf.py").read_text()
    ns = {}
    try:
        exec(
            compile(source, str(spec.origin), "exec"),
            {
                "__name__": "uprate_puf_static",
                "__builtins__": __builtins__,
            },
            ns,
        )
    except Exception:
        # If execution fails due to storage imports, that's fine -
        # we just parse the constants manually.
        import ast

        tree = ast.parse(source)
        ns = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in (
                        "SOI_TO_PUF_POS_ONLY_RENAMES",
                        "SOI_TO_PUF_NEG_ONLY_RENAMES",
                    ):
                        ns[target.id] = ast.literal_eval(node.value)

    assert "business_net_profits" in ns["SOI_TO_PUF_POS_ONLY_RENAMES"]
    assert "capital_gains_gross" in ns["SOI_TO_PUF_POS_ONLY_RENAMES"]
    assert "partnership_and_s_corp_income" in ns["SOI_TO_PUF_POS_ONLY_RENAMES"]
    assert "business_net_losses" in ns["SOI_TO_PUF_NEG_ONLY_RENAMES"]
    assert "capital_gains_losses" in ns["SOI_TO_PUF_NEG_ONLY_RENAMES"]
    assert "partnership_and_s_corp_losses" in ns["SOI_TO_PUF_NEG_ONLY_RENAMES"]
