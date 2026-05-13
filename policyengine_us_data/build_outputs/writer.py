"""H5 writing boundary for local-area publication outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from policyengine_us_data.pipeline_metadata import pipeline_node

from .payload import H5Payload

__all__ = ["H5WriteResult", "H5Writer"]


@pipeline_node(
    id="local_h5_write_result",
    label="H5WriteResult",
    node_type="library",
    description="Post-write verification summary for one local H5 file.",
    source_file="policyengine_us_data/build_outputs/writer.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_writer.py"],
)
@dataclass(frozen=True)
class H5WriteResult:
    """Summary of one H5 write and lightweight verification pass."""

    path: Path
    households: int | None
    persons: int | None
    household_weight_sum: float | None
    person_weight_sum: float | None


@pipeline_node(
    id="local_h5_writer",
    label="H5Writer",
    node_type="library",
    description="Write one period-grouped local H5 payload to disk.",
    source_file="policyengine_us_data/build_outputs/writer.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_writer.py"],
)
@dataclass(frozen=True)
class H5Writer:
    """Write period-grouped local H5 payloads and verify key output counts."""

    def write(
        self,
        *,
        payload: H5Payload,
        output_path: Path,
    ) -> H5WriteResult:
        """Write `payload` to `output_path` and return a verification summary."""

        payload.validate_shapes()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(path), "w") as file:
            for variable, periods in payload.data.items():
                group = file.create_group(variable)
                for period, values in periods.items():
                    group.create_dataset(str(period), data=values)
        return self.verify(path=path, time_period=payload.time_period)

    def verify(self, *, path: Path, time_period: int) -> H5WriteResult:
        """Read key output variables from a written H5 file."""

        tp = str(time_period)
        with h5py.File(str(path), "r") as file:
            households = _length_if_present(file, "household_id", tp)
            persons = _length_if_present(file, "person_id", tp)
            household_weight_sum = _sum_if_present(file, "household_weight", tp)
            person_weight_sum = _sum_if_present(file, "person_weight", tp)
        return H5WriteResult(
            path=Path(path),
            households=households,
            persons=persons,
            household_weight_sum=household_weight_sum,
            person_weight_sum=person_weight_sum,
        )


def _length_if_present(file: h5py.File, variable: str, period: str) -> int | None:
    if variable not in file or period not in file[variable]:
        return None
    return int(len(file[variable][period][:]))


def _sum_if_present(file: h5py.File, variable: str, period: str) -> float | None:
    if variable not in file or period not in file[variable]:
        return None
    return float(np.sum(file[variable][period][:]))
