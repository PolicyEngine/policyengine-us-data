"""US-local clone-by-household weight layout helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CloneWeightMatrix:
    """A US clone-by-household weight vector with validated shape."""

    values: np.ndarray
    n_records: int
    n_clones: int

    @classmethod
    def from_vector(
        cls,
        values: np.ndarray,
        n_records: int,
    ) -> "CloneWeightMatrix":
        if n_records <= 0:
            raise ValueError("n_records must be positive")

        arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError("weight vector must be one-dimensional")
        if arr.size % n_records != 0:
            raise ValueError(
                f"Weight vector length {arr.size} is not divisible by n_records={n_records}"
            )

        return cls(
            values=arr,
            n_records=int(n_records),
            n_clones=int(arr.size // n_records),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_clones, self.n_records)

    def as_matrix(self) -> np.ndarray:
        return self.values.reshape(self.shape)
