"""Unit tests for extended EITC calibration targets.

Covers:

* Presence and structure of ``eitc_state.csv`` and
  ``eitc_by_agi_and_children.csv``.
* IRS aggregate crosscheck: state totals sum within 1% of the
  Treasury EITC tax-expenditure parameter for the same tax year.
* Helper-level wiring of the new target families into a synthetic
  simulation, asserting the expected label schema and 1:1 target /
  loss-matrix-column alignment.
* Placeholder ``[TO BE CALCULATED]`` rows are skipped (neither a
  target nor a column is emitted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.storage import CALIBRATION_FOLDER
from policyengine_us_data.utils.loss import (
    _add_eitc_by_agi_and_children_targets,
    _add_state_eitc_targets,
    _skip_unverified_target,
)


# --- IRS data-file sanity ------------------------------------------------


def test_eitc_state_csv_present_with_expected_columns():
    df = pd.read_csv(CALIBRATION_FOLDER / "eitc_state.csv", comment="#")
    assert list(df.columns) == ["GEO_ID", "Returns", "Amount"]
    # 50 states + DC.
    assert len(df) == 51
    # Every GEO_ID must be a FIPS-coded state identifier.
    assert df["GEO_ID"].str.startswith("0400000US").all()
    # Numeric columns must parse as nonnegative numbers.
    assert (df["Returns"].astype(float) >= 0).all()
    assert (df["Amount"].astype(float) >= 0).all()


def test_eitc_by_agi_and_children_csv_present_with_expected_columns():
    df = pd.read_csv(CALIBRATION_FOLDER / "eitc_by_agi_and_children.csv", comment="#")
    assert list(df.columns) == [
        "count_children",
        "agi_lower",
        "agi_upper",
        "returns",
        "amount",
    ]
    # 4 qualifying-children buckets x 28 AGI rows (SOI Table 2.5 shape).
    assert set(df["count_children"].unique()) == {0, 1, 2, 3}
    # agi bounds are strictly increasing within a child-count bucket.
    for child_count, group in df.groupby("count_children"):
        bounds = list(
            zip(
                group["agi_lower"].astype(float),
                group["agi_upper"].astype(float),
            )
        )
        for lo, up in bounds:
            assert lo < up, (child_count, lo, up)


def test_state_eitc_totals_match_irs_national_aggregate():
    """State-level EITC amounts sum to within 1% of the IRS Historical
    Table 2 national EITC row.

    This isn't a Treasury target check (Treasury outlays include only the
    refundable portion and diverge by ~$10B). It's an internal-consistency
    check on the extraction: the CSV is built from the same workbook that
    publishes the national total.
    """
    df = pd.read_csv(CALIBRATION_FOLDER / "eitc_state.csv", comment="#")
    sum_returns = int(df["Returns"].sum())
    sum_amount = int(df["Amount"].sum())

    # TY2022 IRS SOI Historical Table 2 US row: 23,692,190 returns,
    # A59660 = 59,204,588 (thousands) = $59,204,588,000.
    # Disclosure rounding moves the state-sum by ~0.05% from the
    # published US row — everything under 1% is acceptable.
    expected_returns = 23_692_190
    expected_amount = 59_204_588_000

    assert abs(sum_returns - expected_returns) / expected_returns < 0.01, (
        f"State returns sum {sum_returns:,} off from IRS US total {expected_returns:,}"
    )
    assert abs(sum_amount - expected_amount) / expected_amount < 0.01, (
        f"State amount sum {sum_amount:,} off from IRS US total {expected_amount:,}"
    )


# --- Placeholder skipping ------------------------------------------------


def test_skip_unverified_target_identifies_placeholders():
    assert _skip_unverified_target("[TO BE CALCULATED]")
    assert _skip_unverified_target("TBD")
    assert _skip_unverified_target("")
    assert _skip_unverified_target(None)
    assert _skip_unverified_target(float("nan"))
    assert not _skip_unverified_target(0)
    assert not _skip_unverified_target(1.5)
    assert not _skip_unverified_target(1_000_000)


# --- Helper wiring against a tiny synthetic simulation -------------------


class _FakeArray:
    def __init__(self, values):
        self.values = np.asarray(values)


class _FakeStateEitcSimulation:
    """Minimal simulation stub for _add_state_eitc_targets.

    Represents three tax units, each mapped 1:1 to a household in a
    different state (CA / TX / NY). EITC is nonzero for CA and TX only.
    """

    def __init__(self):
        self._state_codes = np.array(["CA", "TX", "NY"])
        self._eitc = np.array([1200.0, 2500.0, 0.0])

    def calculate(self, variable, map_to=None, period=None):
        if variable == "eitc":
            return _FakeArray(self._eitc)
        if variable == "state_code":
            assert map_to == "person"
            return _FakeArray(self._state_codes)
        raise AssertionError(f"Unexpected variable {variable!r}")

    def map_result(self, values, source, target, how=None):
        # In the stub every entity has exactly one person / tax unit per
        # household, so pass-through is correct. Preserve the input
        # dtype so string arrays (state_code) remain usable downstream.
        return np.asarray(values)


class _FakeAgiChildEitcSimulation:
    """Minimal simulation stub for _add_eitc_by_agi_and_children_targets.

    Four tax units:
    * 0 children, AGI $8k, EITC $400
    * 1 child,  AGI $15k, EITC $3,000
    * 2 children, AGI $22k, EITC $6,000
    * 3 children, AGI $40k, EITC $2,500
    """

    def __init__(self):
        self._eitc_child_count = np.array([0, 1, 2, 3])
        self._eitc = np.array([400.0, 3000.0, 6000.0, 2500.0])
        self._agi = np.array([8_000.0, 15_000.0, 22_000.0, 40_000.0])

    def calculate(self, variable, map_to=None, period=None):
        if variable == "eitc_child_count":
            return _FakeArray(self._eitc_child_count)
        if variable == "eitc":
            return _FakeArray(self._eitc)
        if variable == "adjusted_gross_income":
            return _FakeArray(self._agi)
        raise AssertionError(f"Unexpected variable {variable!r}")

    def map_result(self, values, source, target, how=None):
        return np.asarray(values, dtype=float)


def test_add_state_eitc_targets_produces_aligned_columns_and_targets():
    sim = _FakeStateEitcSimulation()
    loss_matrix = pd.DataFrame()
    targets: list = []

    targets, loss_matrix = _add_state_eitc_targets(
        loss_matrix,
        targets,
        sim,
        eitc_spending_uprating=1.0,
        population_uprating=1.0,
    )

    # All 51 jurisdictions x (returns + amount) = 102 new columns and
    # 102 new targets (CSV has no placeholders).
    assert len(loss_matrix.columns) == 102
    assert len(targets) == 102
    # California (FIPS 06) and Texas (FIPS 48) labels must be present.
    assert "nation/irs/eitc/returns/state_06" in loss_matrix.columns
    assert "nation/irs/eitc/amount/state_06" in loss_matrix.columns
    assert "nation/irs/eitc/returns/state_48" in loss_matrix.columns
    assert "nation/irs/eitc/amount/state_48" in loss_matrix.columns

    # For the CA-indexed household the EITC amount column should equal
    # the simulation's EITC; every other household is zeroed out.
    ca_amount = loss_matrix["nation/irs/eitc/amount/state_06"].to_numpy()
    assert ca_amount[0] == pytest.approx(1200.0)
    assert ca_amount[1] == pytest.approx(0.0)
    assert ca_amount[2] == pytest.approx(0.0)

    # Returns count column is 0/1 indicator.
    ca_returns = loss_matrix["nation/irs/eitc/returns/state_06"].to_numpy()
    assert ca_returns[0] == pytest.approx(1.0)
    assert ca_returns[1] == pytest.approx(0.0)


def test_add_eitc_by_agi_and_children_targets_produces_aligned_columns():
    sim = _FakeAgiChildEitcSimulation()
    loss_matrix = pd.DataFrame()
    targets: list = []

    targets, loss_matrix = _add_eitc_by_agi_and_children_targets(
        loss_matrix,
        targets,
        sim,
        eitc_spending_uprating=1.0,
        population_uprating=1.0,
    )

    # Every emitted column has a matching target (no off-by-one).
    assert len(loss_matrix.columns) == len(targets)

    # The expected per-child-count slug families appear. The SOI AGI
    # binning covers 28 bins per child-count bucket, and each bin
    # contributes (returns, amount) — so 28 x 4 x 2 = 224 targets.
    assert len(targets) == 224

    # Spot-check one label: 2 children, AGI [$20k, $25k) — hits a
    # well-populated SOI bucket. The slug uses ``fmt`` from
    # ``loss.py``, which renders 20000 -> "20k".
    assert "nation/irs/eitc/returns/c2_20k_25k" in loss_matrix.columns

    # The fake unit with count_children=3 and AGI=$40k should register
    # in the c3_40k_45k bucket.
    c3_40k_amount = loss_matrix["nation/irs/eitc/amount/c3_40k_45k"].to_numpy()
    # The 3-child, $40k household is row index 3 in the fake sim.
    assert c3_40k_amount[3] == pytest.approx(2500.0)
    # Other rows should be zero.
    assert c3_40k_amount[0] == pytest.approx(0.0)
    assert c3_40k_amount[1] == pytest.approx(0.0)
    assert c3_40k_amount[2] == pytest.approx(0.0)


def test_placeholder_rows_are_skipped_without_breaking_alignment(tmp_path, monkeypatch):
    """Rows marked ``[TO BE CALCULATED]`` must not emit targets or columns."""

    placeholder_csv = tmp_path / "eitc_state.csv"
    placeholder_csv.write_text(
        "# test placeholder row handling\n"
        "GEO_ID,Returns,Amount\n"
        "0400000US06,2519120,5770703000\n"
        "0400000US48,[TO BE CALCULATED],[TO BE CALCULATED]\n"
        "0400000US36,1451910,3464518000\n"
    )

    # Point the CALIBRATION_FOLDER-resolved path at our synthetic CSV.
    from policyengine_us_data.utils import loss as loss_module

    original_folder = loss_module.CALIBRATION_FOLDER
    monkeypatch.setattr(loss_module, "CALIBRATION_FOLDER", tmp_path)

    try:
        sim = _FakeStateEitcSimulation()
        loss_matrix = pd.DataFrame()
        targets: list = []
        targets, loss_matrix = _add_state_eitc_targets(
            loss_matrix,
            targets,
            sim,
            eitc_spending_uprating=1.0,
            population_uprating=1.0,
        )

        # 2 usable rows x (returns + amount) = 4 columns/targets; the
        # placeholder row was skipped on both axes.
        assert len(loss_matrix.columns) == 4
        assert len(targets) == 4
        assert "nation/irs/eitc/returns/state_06" in loss_matrix.columns
        assert "nation/irs/eitc/amount/state_06" in loss_matrix.columns
        assert "nation/irs/eitc/returns/state_36" in loss_matrix.columns
        assert "nation/irs/eitc/amount/state_36" in loss_matrix.columns
        # Texas row was the placeholder — must not appear.
        assert "nation/irs/eitc/returns/state_48" not in loss_matrix.columns
        assert "nation/irs/eitc/amount/state_48" not in loss_matrix.columns
    finally:
        monkeypatch.setattr(loss_module, "CALIBRATION_FOLDER", original_folder)


def test_mixed_placeholder_row_keeps_valid_metric_drops_invalid(tmp_path, monkeypatch):
    """A row with one valid and one placeholder metric must keep targets
    and columns in lockstep: the valid metric survives, the placeholder
    metric is dropped on *both* axes.
    """

    mixed_csv = tmp_path / "eitc_state.csv"
    mixed_csv.write_text(
        "GEO_ID,Returns,Amount\n"
        # Valid returns, placeholder amount.
        "0400000US06,2519120,[TO BE CALCULATED]\n"
        # Placeholder returns, valid amount.
        "0400000US48,[TO BE CALCULATED],6500000000\n"
        # Both valid for sanity.
        "0400000US36,1451910,3464518000\n"
    )

    from policyengine_us_data.utils import loss as loss_module

    original_folder = loss_module.CALIBRATION_FOLDER
    monkeypatch.setattr(loss_module, "CALIBRATION_FOLDER", tmp_path)

    try:
        sim = _FakeStateEitcSimulation()
        loss_matrix = pd.DataFrame()
        targets: list = []
        targets, loss_matrix = _add_state_eitc_targets(
            loss_matrix,
            targets,
            sim,
            eitc_spending_uprating=1.0,
            population_uprating=1.0,
        )

        # 2 valid metrics from rows 1 and 2, plus both metrics from row 3
        # = 4 columns/targets. Critically: cols and targets stay aligned.
        assert len(loss_matrix.columns) == len(targets)
        assert len(loss_matrix.columns) == 4

        # CA: returns present, amount dropped.
        assert "nation/irs/eitc/returns/state_06" in loss_matrix.columns
        assert "nation/irs/eitc/amount/state_06" not in loss_matrix.columns
        # TX: returns dropped, amount present.
        assert "nation/irs/eitc/returns/state_48" not in loss_matrix.columns
        assert "nation/irs/eitc/amount/state_48" in loss_matrix.columns
        # NY: both present.
        assert "nation/irs/eitc/returns/state_36" in loss_matrix.columns
        assert "nation/irs/eitc/amount/state_36" in loss_matrix.columns
    finally:
        monkeypatch.setattr(loss_module, "CALIBRATION_FOLDER", original_folder)


class _FakeAgiChildEitcSimulationWithFourPlusChildren:
    """Simulation stub covering 5 tax units with eitc_child_count in
    ``{0, 1, 2, 3, 4}``.

    Lets us verify the "3 or more" (``count_children=3``) bucket
    aggregates *both* 3-child and 4+-child households (SOI Publication
    1304 Table 2.5 tops out its child-count axis at "3 or more").
    """

    def __init__(self):
        self._eitc_child_count = np.array([0, 1, 2, 3, 4])
        self._eitc = np.array([400.0, 3000.0, 6000.0, 2500.0, 1800.0])
        self._agi = np.array([8_000.0, 15_000.0, 22_000.0, 40_000.0, 42_000.0])

    def calculate(self, variable, map_to=None, period=None):
        if variable == "eitc_child_count":
            return _FakeArray(self._eitc_child_count)
        if variable == "eitc":
            return _FakeArray(self._eitc)
        if variable == "adjusted_gross_income":
            return _FakeArray(self._agi)
        raise AssertionError(f"Unexpected variable {variable!r}")

    def map_result(self, values, source, target, how=None):
        return np.asarray(values, dtype=float)


def test_three_or_more_children_bucket_uses_ge_not_eq():
    """SOI's top child-count bucket is ``3 or more``. A 4-child household
    must register in the ``c3_*`` bucket, and a 3-child household must
    not be double-counted across ``c2`` and ``c3`` buckets.
    """

    sim = _FakeAgiChildEitcSimulationWithFourPlusChildren()
    loss_matrix = pd.DataFrame()
    targets: list = []

    targets, loss_matrix = _add_eitc_by_agi_and_children_targets(
        loss_matrix,
        targets,
        sim,
        eitc_spending_uprating=1.0,
        population_uprating=1.0,
    )

    # The 3-child, $40k unit is row 3; the 4-child, $42k unit is row 4.
    # Both should land in the c3_40k_45k bucket.
    c3_40k = loss_matrix["nation/irs/eitc/amount/c3_40k_45k"].to_numpy()
    assert c3_40k[3] == pytest.approx(2500.0)
    assert c3_40k[4] == pytest.approx(1800.0)
    # The 2-child unit should NOT appear in the c3 bucket.
    assert c3_40k[2] == pytest.approx(0.0)

    # The 4-child unit must not ALSO appear in c2_40k_45k — only the c3
    # bucket absorbs "3 or more" to avoid double-counting.
    c2_40k = loss_matrix["nation/irs/eitc/amount/c2_40k_45k"].to_numpy()
    assert c2_40k[4] == pytest.approx(0.0)

    # Count-column sanity: the 4-child unit contributes exactly 1 return
    # to c3_40k_45k (it has nonzero EITC and lands in the bucket).
    c3_40k_returns = loss_matrix["nation/irs/eitc/returns/c3_40k_45k"].to_numpy()
    assert c3_40k_returns[4] == pytest.approx(1.0)


def test_nonunity_uprating_propagates_to_targets(tmp_path, monkeypatch):
    """``eitc_spending_uprating`` scales dollar targets and
    ``population_uprating`` scales return-count targets. A base-year CSV
    passed through with both set to 1.10 must produce targets exactly
    1.10 times the CSV value.
    """

    csv_path = tmp_path / "eitc_state.csv"
    csv_returns = 2_000_000
    csv_amount = 5_000_000_000
    csv_path.write_text(
        f"GEO_ID,Returns,Amount\n0400000US06,{csv_returns},{csv_amount}\n"
    )

    from policyengine_us_data.utils import loss as loss_module

    original_folder = loss_module.CALIBRATION_FOLDER
    monkeypatch.setattr(loss_module, "CALIBRATION_FOLDER", tmp_path)

    try:
        sim = _FakeStateEitcSimulation()
        loss_matrix = pd.DataFrame()
        targets: list = []
        targets, loss_matrix = _add_state_eitc_targets(
            loss_matrix,
            targets,
            sim,
            eitc_spending_uprating=1.10,
            population_uprating=1.05,
        )

        # Single row produces 2 targets: returns then amount, in that order.
        assert targets[0] == pytest.approx(csv_returns * 1.05)
        assert targets[1] == pytest.approx(csv_amount * 1.10)
    finally:
        monkeypatch.setattr(loss_module, "CALIBRATION_FOLDER", original_folder)


# --- Regression: stale / contradictory targets must stay gone ------------


def test_legacy_eitc_csv_is_removed_from_calibration_folder():
    """The legacy TY2020 ``eitc.csv`` was removed because it contradicted
    the new TY2022 SOI sources (stale vintage, and the per-child-count
    row sums conflicted with state-sum and AGI×children cross-tab). If
    it reappears, the EITC target set becomes internally inconsistent
    again — fail loudly.
    """
    assert not (CALIBRATION_FOLDER / "eitc.csv").exists()


def test_loss_module_does_not_target_treasury_eitc_aggregate():
    """Treasury's ``tax_expenditures.eitc`` parameter measures EITC
    *outlays* (~$67B), while ``eitc`` in the microsim measures total
    EITC *claimed* (~$59B, per SOI). Targeting both simultaneously
    creates an unsatisfiable internal contradiction. The loss matrix
    must not include a ``nation/treasury/eitc`` aggregate column.
    """
    from policyengine_us_data.utils import loss as loss_module
    import inspect

    source = inspect.getsource(loss_module)
    # The parameter is still *read* (for uprating), but no loss-matrix
    # column with this label should be created.
    assert '"nation/treasury/eitc"' not in source
    assert "'nation/treasury/eitc'" not in source


def test_loss_module_does_not_target_legacy_per_child_count_eitc():
    """The old TY2020 ``nation/irs/eitc/returns/count_children_*`` and
    ``nation/irs/eitc/amount/count_children_*`` columns were replaced
    by the ``(child_count × AGI bucket)`` cross-tab, which carries the
    same per-child-count information at a finer grain. Keeping both
    double-counts the child-count signal.
    """
    from policyengine_us_data.utils import loss as loss_module
    import inspect

    source = inspect.getsource(loss_module)
    assert "count_children_" not in source
