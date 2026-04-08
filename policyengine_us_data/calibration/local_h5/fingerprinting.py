"""Semantic fingerprinting for local H5 publish inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from policyengine_us_data.calibration.local_h5.package_geography import (
    CalibrationPackageGeographyLoader,
    require_calibration_package_path,
)


def _require_file(path: str | Path, *, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} file not found at {path}")
    return path


@dataclass(frozen=True)
class FingerprintInputs:
    weights_path: Path
    dataset_path: Path
    calibration_package_path: Path
    n_clones: int
    seed: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "weights_path": str(self.weights_path),
            "dataset_path": str(self.dataset_path),
            "calibration_package_path": str(self.calibration_package_path),
            "n_clones": self.n_clones,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class FingerprintComponents:
    weights_sha256: str
    dataset_sha256: str
    geography_sha256: str
    n_clones: int
    seed: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "weights_sha256": self.weights_sha256,
            "dataset_sha256": self.dataset_sha256,
            "geography_sha256": self.geography_sha256,
            "n_clones": self.n_clones,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FingerprintComponents":
        return cls(
            weights_sha256=str(payload["weights_sha256"]),
            dataset_sha256=str(payload["dataset_sha256"]),
            geography_sha256=str(payload["geography_sha256"]),
            n_clones=int(payload["n_clones"]),
            seed=int(payload["seed"]),
        )


@dataclass(frozen=True)
class FingerprintRecord:
    schema_version: str
    algorithm: str
    digest: str
    components: FingerprintComponents | None = None
    inputs: Mapping[str, str | int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "fingerprint": self.digest,
            "digest": self.digest,
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
        }
        if self.components is not None:
            payload["components"] = self.components.to_dict()
        if self.inputs:
            payload["inputs"] = dict(self.inputs)
        return payload


class FingerprintService:
    """Single authoritative definition of local H5 publish identity."""

    SCHEMA_VERSION = "local_h5_publish_v1"
    ALGORITHM = "sha256-truncated-16"
    DIGEST_HEX_CHARS = 16
    _GEOGRAPHY_FIELDS = (
        "block_geoid",
        "cd_geoid",
        "county_fips",
        "state_fips",
        "n_records",
        "n_clones",
    )

    def __init__(self, loader: CalibrationPackageGeographyLoader | None = None):
        self.loader = loader or CalibrationPackageGeographyLoader()

    def build_inputs(
        self,
        *,
        weights_path: str | Path,
        dataset_path: str | Path,
        calibration_package_path: str | Path,
        n_clones: int,
        seed: int,
    ) -> FingerprintInputs:
        if n_clones <= 0:
            raise ValueError("n_clones must be positive")

        return FingerprintInputs(
            weights_path=_require_file(weights_path, label="weights"),
            dataset_path=_require_file(dataset_path, label="dataset"),
            calibration_package_path=require_calibration_package_path(
                calibration_package_path
            ),
            n_clones=int(n_clones),
            seed=int(seed),
        )

    def create_publish_fingerprint(
        self,
        *,
        weights_path: str | Path,
        dataset_path: str | Path,
        calibration_package_path: str | Path,
        n_clones: int,
        seed: int,
    ) -> FingerprintRecord:
        inputs = self.build_inputs(
            weights_path=weights_path,
            dataset_path=dataset_path,
            calibration_package_path=calibration_package_path,
            n_clones=n_clones,
            seed=seed,
        )
        return self.create_from_inputs(inputs)

    def create_from_inputs(self, inputs: FingerprintInputs) -> FingerprintRecord:
        components = FingerprintComponents(
            weights_sha256=self._sha256_file(inputs.weights_path),
            dataset_sha256=self._sha256_file(inputs.dataset_path),
            geography_sha256=self._sha256_package_geography(
                inputs.calibration_package_path
            ),
            n_clones=inputs.n_clones,
            seed=inputs.seed,
        )
        digest = self._compute_digest(components)
        return FingerprintRecord(
            schema_version=self.SCHEMA_VERSION,
            algorithm=self.ALGORITHM,
            digest=digest,
            components=components,
            inputs=inputs.to_dict(),
        )

    def serialize(self, record: FingerprintRecord) -> dict[str, Any]:
        return record.to_dict()

    def deserialize(self, payload: Mapping[str, Any]) -> FingerprintRecord:
        digest = payload.get("digest") or payload.get("fingerprint")
        if not digest:
            raise ValueError("Fingerprint payload is missing digest/fingerprint")

        components_payload = payload.get("components")
        components = None
        if isinstance(components_payload, Mapping):
            components = FingerprintComponents.from_dict(components_payload)

        inputs_payload = payload.get("inputs")
        inputs: Mapping[str, str | int]
        if isinstance(inputs_payload, Mapping):
            inputs = {
                str(key): value for key, value in inputs_payload.items()
            }
        else:
            inputs = {}

        return FingerprintRecord(
            schema_version=str(payload.get("schema_version", "legacy")),
            algorithm=str(payload.get("algorithm", self.ALGORITHM)),
            digest=str(digest),
            components=components,
            inputs=inputs,
        )

    def write_record(self, path: str | Path, record: FingerprintRecord) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.serialize(record), f, indent=2, sort_keys=True)

    def read_record(self, path: str | Path) -> FingerprintRecord:
        with open(path) as f:
            payload = json.load(f)
        return self.deserialize(payload)

    def matches(
        self,
        stored: FingerprintRecord,
        current: FingerprintRecord,
    ) -> bool:
        return stored.digest == current.digest

    def legacy_record(self, digest: str) -> FingerprintRecord:
        return FingerprintRecord(
            schema_version="legacy",
            algorithm=self.ALGORITHM,
            digest=str(digest),
        )

    def _compute_digest(self, components: FingerprintComponents) -> str:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            **components.to_dict(),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(raw).hexdigest()[: self.DIGEST_HEX_CHARS]

    def _sha256_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _sha256_package_geography(self, package_path: Path) -> str:
        loaded = self.loader.load(package_path)
        if loaded is None:
            raise ValueError(
                f"Calibration package at {package_path} does not contain usable geography"
            )
        payload = self.loader.serialize_geography(loaded.geography)
        return self._sha256_geography_payload(payload)

    def _sha256_geography_payload(self, payload: Mapping[str, Any]) -> str:
        h = hashlib.sha256()
        for field in self._GEOGRAPHY_FIELDS:
            h.update(field.encode("utf-8"))
            h.update(b"\0")
            value = payload[field]
            if field in ("n_records", "n_clones"):
                h.update(str(int(value)).encode("utf-8"))
                h.update(b"\0")
                continue

            arr = np.asarray(value)
            h.update(str(arr.shape).encode("utf-8"))
            h.update(b"\0")
            if arr.dtype.kind in "iufb":
                h.update(str(arr.dtype).encode("utf-8"))
                h.update(b"\0")
                h.update(np.ascontiguousarray(arr).tobytes())
            else:
                for item in arr.reshape(-1):
                    h.update(str(item).encode("utf-8"))
                    h.update(b"\0")
        return h.hexdigest()
