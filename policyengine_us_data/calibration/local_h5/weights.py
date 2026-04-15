"""Clone-weight shape contracts for local H5 worker setup.

This module introduces a narrow structural boundary around the flat
calibration weight vector used by current publication paths. It is
intentionally pure and does not perform file IO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CloneWeightMatrix:
    """Structured view of clone-level household weights.

    The canonical in-memory representation is the original flat vector of
    length ``n_records * n_clones``. Matrix views are derived on demand.
    """

    values: np.ndarray
    n_records: int
    n_clones: int

    @classmethod
    def from_vector(
        cls,
        values: np.ndarray,
        n_records: int,
    ) -> "CloneWeightMatrix":
        """Build a structured weight contract from a flat vector.

        Args:
            values: Flat clone-level weight vector.
            n_records: Number of base records per clone.
        """

        vector = cls._normalize_vector(values)
        if n_records <= 0:
            raise ValueError("n_records must be positive")
        if len(vector) % n_records != 0:
            raise ValueError(
                f"Weight vector length {len(vector)} is not divisible by "
                f"n_records={n_records}"
            )
        n_clones = len(vector) // n_records
        if n_clones <= 0:
            raise ValueError("n_clones must be positive")
        return cls(values=vector, n_records=n_records, n_clones=n_clones)

    @classmethod
    def from_vector_with_clone_count(
        cls,
        values: np.ndarray,
        n_clones: int,
    ) -> "CloneWeightMatrix":
        """Build a structured weight contract when clone count is known."""

        vector = cls._normalize_vector(values)
        if n_clones <= 0:
            raise ValueError("n_clones must be positive")
        if len(vector) % n_clones != 0:
            raise ValueError(
                f"Weight vector length {len(vector)} is not divisible by "
                f"n_clones={n_clones}"
            )
        n_records = len(vector) // n_clones
        if n_records <= 0:
            raise ValueError("n_records must be positive")
        return cls(values=vector, n_records=n_records, n_clones=n_clones)

    def as_matrix(self) -> np.ndarray:
        """Return the clone-by-record matrix view of the weight vector."""

        return self.values.reshape(self.n_clones, self.n_records)

    def as_vector(self) -> np.ndarray:
        """Return the original flat vector representation."""

        return self.values

    @staticmethod
    def _normalize_vector(values: np.ndarray) -> np.ndarray:
        vector = np.asarray(values)
        if vector.ndim != 1:
            raise ValueError("Weight vector must be one-dimensional")
        if vector.size == 0:
            raise ValueError("Weight vector must be non-empty")
        if not np.issubdtype(vector.dtype, np.number):
            raise TypeError("Weight vector must have a numeric dtype")
        return vector
