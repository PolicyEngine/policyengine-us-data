"""Clone-weight shape contracts for local H5 publication.

This module defines the narrow structural boundary around the flat
clone-level calibration weight vector used by current H5 publication paths.
It is intentionally pure and does not perform file IO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from policyengine_us_data.pipeline_metadata import pipeline_node

__all__ = ["CloneWeightMatrix"]


@pipeline_node(
    id="clone_weight_matrix",
    label="CloneWeightMatrix",
    node_type="library",
    description=("Explicit shape contract for flat clone-level calibration weights."),
    source_file="policyengine_us_data/calibration/local_h5/weights.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    artifacts_in=[
        "calibration_weights.npy",
        "national_calibration_weights.npy",
    ],
    validation_commands=[
        "uv run pytest tests/unit/calibration/test_local_h5_weights.py"
    ],
)
@dataclass(frozen=True)
class CloneWeightMatrix:
    """Structured view of clone-level household weights.

    The canonical in-memory representation remains the flat vector of length
    ``n_records * n_clones``. Matrix views are derived on demand as
    ``(n_clones, n_records)`` arrays.

    Attributes:
        values: Flat clone-level weight vector.
        n_records: Number of base records represented by each clone.
        n_clones: Number of clone copies represented in the flat vector.
    """

    values: np.ndarray
    n_records: int
    n_clones: int

    def __post_init__(self) -> None:
        vector = self._normalize_vector(self.values)
        n_records = self._normalize_positive_count("n_records", self.n_records)
        n_clones = self._normalize_positive_count("n_clones", self.n_clones)
        expected_length = n_records * n_clones

        if vector.size != expected_length:
            raise ValueError(
                f"Weight vector length {vector.size} does not equal "
                f"n_records * n_clones={expected_length}"
            )

        normalized = np.array(vector, copy=True)
        normalized.setflags(write=False)
        object.__setattr__(self, "values", normalized)
        object.__setattr__(self, "n_records", n_records)
        object.__setattr__(self, "n_clones", n_clones)

    @classmethod
    def from_vector(
        cls,
        values: ArrayLike,
        n_records: int,
    ) -> "CloneWeightMatrix":
        """Build a structured weight contract when record count is known.

        Args:
            values: Flat clone-level weight vector.
            n_records: Number of base records per clone.

        Returns:
            A validated `CloneWeightMatrix`.
        """

        vector = cls._normalize_vector(values)
        normalized_records = cls._normalize_positive_count("n_records", n_records)
        if vector.size % normalized_records != 0:
            raise ValueError(
                f"Weight vector length {vector.size} is not divisible by "
                f"n_records={normalized_records}"
            )
        return cls(
            values=vector,
            n_records=normalized_records,
            n_clones=vector.size // normalized_records,
        )

    @classmethod
    def from_vector_with_clone_count(
        cls,
        values: ArrayLike,
        n_clones: int,
    ) -> "CloneWeightMatrix":
        """Build a structured weight contract when clone count is known.

        Args:
            values: Flat clone-level weight vector.
            n_clones: Number of clone copies represented in the vector.

        Returns:
            A validated `CloneWeightMatrix`.
        """

        vector = cls._normalize_vector(values)
        normalized_clones = cls._normalize_positive_count("n_clones", n_clones)
        if vector.size % normalized_clones != 0:
            raise ValueError(
                f"Weight vector length {vector.size} is not divisible by "
                f"n_clones={normalized_clones}"
            )
        return cls(
            values=vector,
            n_records=vector.size // normalized_clones,
            n_clones=normalized_clones,
        )

    def as_vector(self) -> np.ndarray:
        """Return the flat vector representation.

        Returns:
            A read-only view of the flat clone-level weight vector.
        """

        return self.values

    def as_matrix(self) -> np.ndarray:
        """Return the clone-by-record matrix representation.

        Returns:
            A read-only ``(n_clones, n_records)`` view of the weights.
        """

        matrix = self.values.reshape(self.n_clones, self.n_records)
        matrix.setflags(write=False)
        return matrix

    @staticmethod
    def _normalize_vector(values: ArrayLike) -> np.ndarray:
        vector = np.asarray(values)
        if vector.ndim != 1:
            raise ValueError("Weight vector must be one-dimensional")
        if vector.size == 0:
            raise ValueError("Weight vector must be non-empty")
        if not np.issubdtype(vector.dtype, np.number):
            raise TypeError("Weight vector must have a numeric dtype")
        return vector

    @staticmethod
    def _normalize_positive_count(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise TypeError(f"{name} must be an integer")

        normalized = int(value)
        if normalized <= 0:
            raise ValueError(f"{name} must be positive")
        return normalized
