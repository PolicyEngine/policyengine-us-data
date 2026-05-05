import numpy as np
import pytest

from tests.unit.calibration.fixtures.test_local_h5_weights import (
    load_weights_exports,
)


exports = load_weights_exports()
CloneWeightMatrix = exports["CloneWeightMatrix"]
make_weight_vector = exports["make_weight_vector"]


def test_from_vector_derives_clone_count_from_record_count():
    vector = make_weight_vector(6)

    weights = CloneWeightMatrix.from_vector(vector, n_records=3)

    assert weights.n_records == 3
    assert weights.n_clones == 2
    assert np.array_equal(weights.as_vector(), vector)


def test_from_vector_with_clone_count_derives_record_count():
    vector = make_weight_vector(6)

    weights = CloneWeightMatrix.from_vector_with_clone_count(vector, n_clones=2)

    assert weights.n_records == 3
    assert weights.n_clones == 2
    assert np.array_equal(weights.as_vector(), vector)


def test_as_matrix_returns_clone_by_record_shape():
    vector = make_weight_vector(6)
    weights = CloneWeightMatrix.from_vector(vector, n_records=3)

    matrix = weights.as_matrix()

    assert matrix.shape == (2, 3)
    assert np.array_equal(matrix[0], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(matrix[1], np.array([4.0, 5.0, 6.0]))


def test_from_vector_accepts_array_like_values():
    weights = CloneWeightMatrix.from_vector([1.0, 2.0, 3.0, 4.0], n_records=2)

    assert weights.n_records == 2
    assert weights.n_clones == 2
    assert np.array_equal(weights.as_matrix(), np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_direct_construction_rejects_inconsistent_shape():
    vector = make_weight_vector(5)

    with pytest.raises(ValueError, match="does not equal n_records \\* n_clones"):
        CloneWeightMatrix(values=vector, n_records=2, n_clones=3)


def test_from_vector_rejects_non_divisible_record_shape():
    vector = make_weight_vector(5)

    with pytest.raises(ValueError, match="not divisible by n_records=2"):
        CloneWeightMatrix.from_vector(vector, n_records=2)


def test_from_vector_with_clone_count_rejects_non_divisible_clone_shape():
    vector = make_weight_vector(5)

    with pytest.raises(ValueError, match="not divisible by n_clones=2"):
        CloneWeightMatrix.from_vector_with_clone_count(vector, n_clones=2)


def test_from_vector_rejects_non_positive_dimensions():
    vector = make_weight_vector(4)

    with pytest.raises(ValueError, match="n_records must be positive"):
        CloneWeightMatrix.from_vector(vector, n_records=0)

    with pytest.raises(ValueError, match="n_clones must be positive"):
        CloneWeightMatrix.from_vector_with_clone_count(vector, n_clones=0)


def test_from_vector_rejects_non_integer_dimensions():
    vector = make_weight_vector(4)

    with pytest.raises(TypeError, match="n_records must be an integer"):
        CloneWeightMatrix.from_vector(vector, n_records=2.0)

    with pytest.raises(TypeError, match="n_clones must be an integer"):
        CloneWeightMatrix.from_vector_with_clone_count(vector, n_clones=True)


def test_normalization_rejects_empty_vectors():
    with pytest.raises(ValueError, match="must be non-empty"):
        CloneWeightMatrix.from_vector(np.array([], dtype=float), n_records=1)


def test_normalization_rejects_non_numeric_vectors():
    vector = np.array(["a", "b"], dtype=object)

    with pytest.raises(TypeError, match="numeric dtype"):
        CloneWeightMatrix.from_vector(vector, n_records=1)


def test_normalization_rejects_complex_vectors():
    vector = np.array([1.0 + 2.0j, 3.0 + 4.0j])

    with pytest.raises(TypeError, match="real numeric dtype"):
        CloneWeightMatrix.from_vector(vector, n_records=1)


def test_normalization_rejects_non_one_dimensional_vectors():
    vector = np.arange(6, dtype=float).reshape(2, 3)

    with pytest.raises(ValueError, match="one-dimensional"):
        CloneWeightMatrix.from_vector(vector, n_records=3)


def test_internal_storage_is_decoupled_from_source_vector():
    vector = make_weight_vector(4)
    weights = CloneWeightMatrix.from_vector(vector, n_records=2)

    vector[0] = 99.0

    assert weights.as_vector()[0] == 1.0


def test_weight_views_are_read_only():
    weights = CloneWeightMatrix.from_vector(make_weight_vector(4), n_records=2)

    with pytest.raises(ValueError, match="read-only"):
        weights.as_vector()[0] = 99.0

    with pytest.raises(ValueError, match="read-only"):
        weights.as_matrix()[0, 0] = 99.0
