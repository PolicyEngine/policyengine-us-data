import numpy as np
import pytest

from tests.unit.calibration.fixtures.test_local_h5_weights import (
    load_weights_exports,
    make_weight_vector,
)


exports = load_weights_exports()
CloneWeightMatrix = exports["CloneWeightMatrix"]


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


def test_normalization_rejects_empty_vectors():
    with pytest.raises(ValueError, match="must be non-empty"):
        CloneWeightMatrix.from_vector(np.array([], dtype=float), n_records=1)


def test_normalization_rejects_non_numeric_vectors():
    vector = np.array(["a", "b"], dtype=object)

    with pytest.raises(TypeError, match="numeric dtype"):
        CloneWeightMatrix.from_vector(vector, n_records=1)


def test_normalization_rejects_non_one_dimensional_vectors():
    vector = np.arange(6, dtype=float).reshape(2, 3)

    with pytest.raises(ValueError, match="one-dimensional"):
        CloneWeightMatrix.from_vector(vector, n_records=3)
