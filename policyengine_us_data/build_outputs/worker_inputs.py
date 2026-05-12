"""Normalized input payloads for local H5 worker execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias

from policyengine_us_data.pipeline_metadata import pipeline_node

from .fingerprinting import PublishingInputBundle

WorkerCalibrationInputValue: TypeAlias = str | int
WorkerCalibrationInputPayload: TypeAlias = dict[
    str,
    WorkerCalibrationInputValue,
]


def _coerce_path(value: object, *, field_name: str) -> Path:
    """Return a path from a wire value or raise a clear contract error."""

    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    raise TypeError(f"{field_name} must be a path string, got {type(value).__name__}")


def _coerce_optional_path(value: object, *, field_name: str) -> Path | None:
    """Return an optional path from a wire value."""

    if value is None:
        return None
    return _coerce_path(value, field_name=field_name)


def _coerce_int(value: object, *, field_name: str) -> int:
    """Return an integer from a wire value or raise a clear contract error."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")


@pipeline_node(
    id="local_h5_worker_calibration_inputs",
    label="WorkerCalibrationInputs",
    node_type="library",
    description="Normalized worker-execution input payload for local H5 builds.",
    source_file="policyengine_us_data/build_outputs/worker_inputs.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_worker_inputs.py"
    ],
)
@dataclass(frozen=True)
class WorkerCalibrationInputs:
    """Input artifact paths and runtime settings for one H5 worker batch.

    This is the typed library contract. Modal entrypoints may still exchange the
    `to_wire_dict()` representation because it is easier to serialize and
    inspect, but all producers and consumers should normalize through this
    class before using field values.
    """

    weights_path: Path
    dataset_path: Path
    database_path: Path
    geography_path: Path | None = None
    calibration_package_path: Path | None = None
    run_config_path: Path | None = None
    n_clones: int = 430
    seed: int = 42

    @classmethod
    def from_artifact_paths(
        cls,
        *,
        weights_path: Path,
        dataset_path: Path,
        database_path: Path,
        geography_path: Path | None = None,
        calibration_package_path: Path | None = None,
        run_config_path: Path | None = None,
        n_clones: int = 430,
        seed: int = 42,
        require_optional_paths_exist: bool = True,
    ) -> "WorkerCalibrationInputs":
        """Build worker inputs from coordinator artifact paths.

        Optional paths are included only when present by default, matching the
        previous coordinator behavior for run configs and calibration packages.
        """

        if require_optional_paths_exist:
            geography_path = geography_path if _exists(geography_path) else None
            calibration_package_path = (
                calibration_package_path if _exists(calibration_package_path) else None
            )
            run_config_path = run_config_path if _exists(run_config_path) else None

        return cls(
            weights_path=weights_path,
            dataset_path=dataset_path,
            database_path=database_path,
            geography_path=geography_path,
            calibration_package_path=calibration_package_path,
            run_config_path=run_config_path,
            n_clones=n_clones,
            seed=seed,
        )

    @classmethod
    def from_wire_dict(
        cls,
        payload: Mapping[str, object] | "WorkerCalibrationInputs",
    ) -> "WorkerCalibrationInputs":
        """Normalize a Modal-safe worker input payload."""

        if isinstance(payload, cls):
            return payload

        missing = [
            key for key in ("weights", "dataset", "database") if key not in payload
        ]
        if missing:
            raise KeyError(
                "Missing required worker calibration input(s): " + ", ".join(missing)
            )

        return cls(
            weights_path=_coerce_path(payload["weights"], field_name="weights"),
            dataset_path=_coerce_path(payload["dataset"], field_name="dataset"),
            database_path=_coerce_path(payload["database"], field_name="database"),
            geography_path=_coerce_optional_path(
                payload.get("geography"),
                field_name="geography",
            ),
            calibration_package_path=_coerce_optional_path(
                payload.get("calibration_package"),
                field_name="calibration_package",
            ),
            run_config_path=_coerce_optional_path(
                payload.get("run_config"),
                field_name="run_config",
            ),
            n_clones=_coerce_int(payload.get("n_clones", 430), field_name="n_clones"),
            seed=_coerce_int(payload.get("seed", 42), field_name="seed"),
        )

    def to_wire_dict(self) -> WorkerCalibrationInputPayload:
        """Return the Modal-safe payload used by remote worker entrypoints."""

        payload: WorkerCalibrationInputPayload = {
            "weights": str(self.weights_path),
            "dataset": str(self.dataset_path),
            "database": str(self.database_path),
            "n_clones": self.n_clones,
            "seed": self.seed,
        }
        if self.geography_path is not None:
            payload["geography"] = str(self.geography_path)
        if self.calibration_package_path is not None:
            payload["calibration_package"] = str(self.calibration_package_path)
        if self.run_config_path is not None:
            payload["run_config"] = str(self.run_config_path)
        return payload

    def to_worker_cli_args(self) -> list[str]:
        """Return worker_script CLI arguments for these inputs."""

        args = [
            "--weights-path",
            str(self.weights_path),
            "--dataset-path",
            str(self.dataset_path),
            "--db-path",
            str(self.database_path),
            "--n-clones",
            str(self.n_clones),
            "--seed",
            str(self.seed),
        ]
        if self.geography_path is not None:
            args.extend(["--geography-path", str(self.geography_path)])
        if self.calibration_package_path is not None:
            args.extend(
                [
                    "--calibration-package-path",
                    str(self.calibration_package_path),
                ]
            )
        if self.run_config_path is not None:
            args.extend(["--run-config-path", str(self.run_config_path)])
        return args

    def to_publishing_input_bundle(
        self,
        *,
        run_id: str,
        version: str = "",
        legacy_blocks_path: Path | None = None,
    ) -> PublishingInputBundle:
        """Return the traceability/fingerprinting input bundle."""

        return PublishingInputBundle(
            weights_path=self.weights_path,
            source_dataset_path=self.dataset_path,
            target_db_path=self.database_path,
            exact_geography_path=self.geography_path,
            calibration_package_path=self.calibration_package_path,
            run_config_path=self.run_config_path,
            run_id=run_id,
            version=version,
            n_clones=self.n_clones,
            seed=self.seed,
            legacy_blocks_path=legacy_blocks_path,
        )


def _exists(path: Path | None) -> bool:
    """Return whether an optional artifact path exists."""

    return path is not None and path.exists()
