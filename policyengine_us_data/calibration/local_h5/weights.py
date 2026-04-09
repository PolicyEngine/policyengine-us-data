"""US-local clone-by-household weight layout helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def infer_clone_count_from_weight_length(
    weight_length: int,
    n_records: int,
) -> int:
    """Derive canonical clone count from weight length and record count."""

    if n_records <= 0:
        raise ValueError("n_records must be positive")
    if weight_length <= 0:
        raise ValueError("weight_length must be positive")
    if weight_length % n_records != 0:
        raise ValueError(
            "Weight vector length "
            f"{weight_length} is not divisible by n_records={n_records}"
        )
    return int(weight_length // n_records)


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
        arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError("weight vector must be one-dimensional")

        return cls(
            values=arr,
            n_records=int(n_records),
            n_clones=infer_clone_count_from_weight_length(arr.size, n_records),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_clones, self.n_records)

    def as_matrix(self) -> np.ndarray:
        return self.values.reshape(self.shape)
