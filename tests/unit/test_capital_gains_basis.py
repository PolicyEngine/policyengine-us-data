import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.utils.capital_gains_basis import (
    LONG_TERM_CAPITAL_GAINS_BASIS,
    LONG_TERM_CAPITAL_GAINS_YEARS_HELD,
    add_long_term_capital_gains_basis_to_puf_frame,
    impute_person_level_long_term_capital_gains_basis,
    impute_tax_unit_long_term_capital_gains_basis,
)


def test_tax_unit_imputation_is_record_stable_under_shuffle():
    gains = np.array([1_000, 20_000, -4_000, 0, 7_500, -12_000], dtype=float)
    ids = np.array([101, 102, 103, 104, 105, 106])
    weights = np.array([10, 2, 5, 1, 8, 3], dtype=float)

    direct = impute_tax_unit_long_term_capital_gains_basis(
        gains,
        tax_unit_ids=ids,
        sample_weight=weights,
        tax_year=2026,
    )

    order = np.array([4, 2, 0, 5, 1, 3])
    shuffled = impute_tax_unit_long_term_capital_gains_basis(
        gains[order],
        tax_unit_ids=ids[order],
        sample_weight=weights[order],
        tax_year=2026,
    )

    reverse_order = np.argsort(order)
    np.testing.assert_allclose(direct.basis, shuffled.basis[reverse_order])
    np.testing.assert_allclose(direct.years_held, shuffled.years_held[reverse_order])
    np.testing.assert_array_equal(
        direct.holding_period_bucket,
        shuffled.holding_period_bucket[reverse_order],
    )


def test_zero_gain_records_get_zero_basis_and_holding_period():
    imputation = impute_tax_unit_long_term_capital_gains_basis(
        np.array([0.0]),
        tax_unit_ids=np.array([1]),
        tax_year=2026,
    )

    assert imputation.basis[0] == 0
    assert imputation.years_held[0] == 0
    assert imputation.holding_period_bucket[0] == -1


def test_person_allocation_preserves_collapsed_tax_unit_basis():
    gains = np.array([100.0, -40.0, 0.0, -80.0])
    tax_unit_ids = np.array([1, 1, 2, 3])
    person_ids = np.array([11, 12, 21, 31])

    person_imputation = impute_person_level_long_term_capital_gains_basis(
        gains,
        person_tax_unit_ids=tax_unit_ids,
        person_ids=person_ids,
        tax_year=2026,
    )
    tax_unit_imputation = impute_tax_unit_long_term_capital_gains_basis(
        np.array([60.0, 0.0, -80.0]),
        tax_unit_ids=np.array([1, 2, 3]),
        tax_year=2026,
    )

    assert person_imputation.years_held[0] == pytest.approx(
        person_imputation.years_held[1]
    )
    assert person_imputation.basis[:2].sum() == pytest.approx(
        tax_unit_imputation.basis[0]
    )
    assert person_imputation.basis[2] == 0
    assert person_imputation.years_held[2] == 0
    assert person_imputation.basis[3] == pytest.approx(tax_unit_imputation.basis[2])


def test_puf_frame_helper_adds_basis_and_years():
    puf = pd.DataFrame(
        {
            "RECID": [10, 11, 12],
            "S006": [100.0, 200.0, 300.0],
            "long_term_capital_gains": [5_000.0, -2_000.0, 0.0],
        }
    )

    result = add_long_term_capital_gains_basis_to_puf_frame(puf.copy(), tax_year=2026)

    assert LONG_TERM_CAPITAL_GAINS_BASIS in result
    assert LONG_TERM_CAPITAL_GAINS_YEARS_HELD in result
    assert result.loc[0, LONG_TERM_CAPITAL_GAINS_BASIS] > 0
    assert result.loc[1, LONG_TERM_CAPITAL_GAINS_BASIS] > 0
    assert result.loc[2, LONG_TERM_CAPITAL_GAINS_BASIS] == 0
    assert result.loc[2, LONG_TERM_CAPITAL_GAINS_YEARS_HELD] == 0
