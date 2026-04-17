"""Regression tests for unseeded / globally-seeded RNG calls in
CPS and SIPP dataset generation code (N3, N4, N5 from the US-data
bug hunt).

- N3: ``add_ssn_card_type`` called ``np.random.seed(random_seed)`` and
  ``np.random.choice``, clobbering the process-wide RNG for every
  downstream helper.
- N4: ``add_personal_variables`` used ``np.random.randint`` for the
  80-84 age randomization, so each build re-drew a fresh sample.
- N5: ``sipp.train_tip_model`` and ``sipp.train_asset_model`` used
  ``np.random.choice`` for a weighted resample of training rows; the
  pickled QRFs therefore depended on whatever seeded the global RNG
  before the call.

Fix pattern in every case: replace with a local
``seeded_rng(name)`` (or ``np.random.default_rng``) so (a) the draw is
reproducible run-to-run and (b) the global ``np.random`` state is not
mutated.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CPS_SOURCE = REPO_ROOT / "policyengine_us_data" / "datasets" / "cps" / "cps.py"
SIPP_SOURCE = REPO_ROOT / "policyengine_us_data" / "datasets" / "sipp" / "sipp.py"


def _np_random_calls(source_path: Path) -> list[tuple[str, int]]:
    """Return every call of the form ``np.random.<func>(...)`` in ``source_path``.

    We treat ``np.random.default_rng`` as acceptable (it returns a
    local Generator), so exclude it here.
    """
    tree = ast.parse(source_path.read_text())
    calls: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # np.random.<attr>
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "np"
            and func.value.attr == "random"
        ):
            if func.attr == "default_rng":
                continue
            calls.append((func.attr, getattr(node, "lineno", -1)))
    return calls


def test_cps_uses_no_global_np_random_calls():
    offenders = _np_random_calls(CPS_SOURCE)
    assert offenders == [], (
        "policyengine_us_data/datasets/cps/cps.py must not call the "
        f"global np.random.* API. Found: {offenders}"
    )


def test_sipp_uses_no_global_np_random_calls():
    offenders = _np_random_calls(SIPP_SOURCE)
    assert offenders == [], (
        "policyengine_us_data/datasets/sipp/sipp.py must not call the "
        f"global np.random.* API. Found: {offenders}"
    )


def test_cps_age_randomization_uses_seeded_rng():
    """N4: the 80-84 age randomization must come from a seeded
    Generator, not the global ``np.random.randint`` or equivalent."""
    src = CPS_SOURCE.read_text()
    # Extract the add_personal_variables function text.
    tree = ast.parse(src)
    target = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "add_personal_variables"
        ),
        None,
    )
    assert target is not None, "add_personal_variables not found"
    body_text = ast.get_source_segment(src, target)
    assert body_text is not None
    assert "seeded_rng(" in body_text, (
        "age-randomization path must use seeded_rng(); got:\n" + body_text
    )
    assert "np.random.randint" not in body_text, (
        "add_personal_variables must not use the global np.random.randint"
    )


def test_select_random_subset_uses_local_generator_only():
    """N3: ``select_random_subset_to_target`` must not call
    ``np.random.seed`` / ``np.random.choice`` (global RNG)."""
    src = CPS_SOURCE.read_text()
    tree = ast.parse(src)
    # Find nested function select_random_subset_to_target inside add_ssn_card_type
    add_ssn = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "add_ssn_card_type"
        ),
        None,
    )
    assert add_ssn is not None
    inner = next(
        (
            node
            for node in ast.walk(add_ssn)
            if isinstance(node, ast.FunctionDef)
            and node.name == "select_random_subset_to_target"
        ),
        None,
    )
    assert inner is not None
    inner_src = ast.get_source_segment(src, inner)
    assert inner_src is not None
    for banned in ("np.random.seed", "np.random.choice"):
        assert banned not in inner_src, (
            f"{banned} must not appear in select_random_subset_to_target: "
            f"it clobbers the global RNG / depends on it"
        )


def test_sipp_training_samples_use_seeded_rng():
    """N5: the weighted resample for tip and asset training frames
    must come from a seeded Generator, not the global ``np.random``."""
    src = SIPP_SOURCE.read_text()
    assert "seeded_rng(" in src, "sipp.py must import/use seeded_rng()"
    tree = ast.parse(src)
    for fn_name in ("train_tip_model", "train_asset_model"):
        fn = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == fn_name
            ),
            None,
        )
        assert fn is not None, f"{fn_name} not found in sipp.py"
        fn_src = ast.get_source_segment(src, fn)
        assert fn_src is not None
        assert "np.random.choice" not in fn_src, (
            f"{fn_name} must not use np.random.choice (use a seeded_rng Generator)"
        )
        assert "seeded_rng(" in fn_src, (
            f"{fn_name} must derive its resampler from a seeded generator"
        )


def test_seeded_rng_is_reproducible():
    """Sanity: seeded_rng(name) produces the same draw on repeat calls."""
    from policyengine_us_data.utils.randomness import seeded_rng

    draws_a = seeded_rng("unit_test_marker").integers(0, 1_000_000, 16)
    draws_b = seeded_rng("unit_test_marker").integers(0, 1_000_000, 16)
    assert draws_a.tolist() == draws_b.tolist()
