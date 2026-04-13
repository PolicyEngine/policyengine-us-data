"""Typed input contracts for local H5 publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._contract_utils import jsonable_contract_value


@dataclass(frozen=True)
class PublishingInputBundle:
    """The artifact inputs needed to coordinate one publication run."""

    weights_path: Path
    source_dataset_path: Path
    target_db_path: Path | None
    calibration_package_path: Path | None
    run_config_path: Path | None
    run_id: str
    version: str
    n_clones: int | None
    seed: int

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        if self.n_clones is not None and self.n_clones <= 0:
            raise ValueError("n_clones must be positive when provided")

    def required_paths(self) -> tuple[Path, ...]:
        required = [self.weights_path, self.source_dataset_path]
        if self.target_db_path is not None:
            required.append(self.target_db_path)
        if self.calibration_package_path is not None:
            required.append(self.calibration_package_path)
        return tuple(required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights_path": str(self.weights_path),
            "source_dataset_path": str(self.source_dataset_path),
            "target_db_path": jsonable_contract_value(self.target_db_path),
            "calibration_package_path": jsonable_contract_value(
                self.calibration_package_path
            ),
            "run_config_path": jsonable_contract_value(self.run_config_path),
            "run_id": self.run_id,
            "version": self.version,
            "n_clones": self.n_clones,
            "seed": self.seed,
        }
