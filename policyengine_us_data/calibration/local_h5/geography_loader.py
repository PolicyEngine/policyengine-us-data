"""Exact geography loading for local H5 publication.

This module owns the compatibility path for recovering the geography
assignment used during calibration. Public entrypoints may continue to
expose legacy helpers, but the real loading policy lives here.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from policyengine_us_data.calibration.clone_and_assign import (
    GeographyAssignment,
    load_geography,
    reconstruct_geography_from_blocks,
)

CALIBRATION_WEIGHTS_SUFFIX = "calibration_weights.npy"
GEOGRAPHY_FILENAME = "geography_assignment.npz"
LEGACY_BLOCKS_FILENAME = "stacked_blocks.npy"

GeographySourceKind = Literal["saved_geography", "calibration_package", "legacy_blocks"]


@dataclass(frozen=True)
class ResolvedGeographySource:
    """Resolved physical source used to recover calibration geography."""

    kind: GeographySourceKind
    path: Path


def _calibration_artifact_prefix(weights_path: Path) -> str:
    if weights_path.name.endswith(CALIBRATION_WEIGHTS_SUFFIX):
        return weights_path.name[: -len(CALIBRATION_WEIGHTS_SUFFIX)]
    return ""


def _sibling_artifact_path(weights_path: Path, artifact_name: str) -> Path:
    prefix = _calibration_artifact_prefix(weights_path)
    return weights_path.with_name(f"{prefix}{artifact_name}")


class CalibrationGeographyLoader:
    """Resolve and load exact geography artifacts for publication flows."""

    def resolve_source(
        self,
        *,
        weights_path: Path,
        geography_path: Path | None = None,
        blocks_path: Path | None = None,
        calibration_package_path: Path | None = None,
    ) -> ResolvedGeographySource | None:
        """Resolve the preferred available geometry source.

        Resolution order matches the migration plan:
        saved geography artifact first,
        calibration package payload when available,
        legacy blocks fallback last.
        """

        geo_candidates = []
        if geography_path is not None:
            geo_candidates.append(Path(geography_path))
        geo_candidates.append(_sibling_artifact_path(weights_path, GEOGRAPHY_FILENAME))

        for candidate in geo_candidates:
            if candidate.exists():
                return ResolvedGeographySource(
                    kind="saved_geography",
                    path=candidate,
                )

        if calibration_package_path is not None:
            package_path = Path(calibration_package_path)
            if package_path.exists():
                return ResolvedGeographySource(
                    kind="calibration_package",
                    path=package_path,
                )

        block_candidates = []
        if blocks_path is not None:
            block_candidates.append(Path(blocks_path))
        block_candidates.append(
            _sibling_artifact_path(weights_path, LEGACY_BLOCKS_FILENAME)
        )
        block_candidates.append(weights_path.with_name(LEGACY_BLOCKS_FILENAME))

        for candidate in block_candidates:
            if candidate.exists():
                return ResolvedGeographySource(
                    kind="legacy_blocks",
                    path=candidate,
                )
        return None

    def load(
        self,
        *,
        weights_path: Path,
        n_records: int,
        n_clones: int | None = None,
        geography_path: Path | None = None,
        blocks_path: Path | None = None,
        calibration_package_path: Path | None = None,
    ) -> GeographyAssignment:
        """Load geography using the configured compatibility order."""

        resolved = self.resolve_source(
            weights_path=Path(weights_path),
            geography_path=geography_path,
            blocks_path=blocks_path,
            calibration_package_path=calibration_package_path,
        )
        if resolved is None:
            geo_hint = _sibling_artifact_path(Path(weights_path), GEOGRAPHY_FILENAME)
            legacy_hint = _sibling_artifact_path(
                Path(weights_path),
                LEGACY_BLOCKS_FILENAME,
            )
            raise FileNotFoundError(
                "No saved calibration geography found. Expected either "
                f"{geo_hint} or {legacy_hint}. Re-run calibration on this branch or "
                "provide --geography-path."
            )

        if resolved.kind == "saved_geography":
            return self._load_saved_geography(
                path=resolved.path,
                n_records=n_records,
                n_clones=n_clones,
            )
        if resolved.kind == "calibration_package":
            return self._load_from_package(
                path=resolved.path,
                n_records=n_records,
                n_clones=n_clones,
            )
        return self._load_from_blocks(
            path=resolved.path,
            n_records=n_records,
            n_clones=n_clones,
        )

    def compute_canonical_checksum(
        self,
        *,
        weights_path: Path,
        n_records: int,
        n_clones: int | None = None,
        geography_path: Path | None = None,
        blocks_path: Path | None = None,
        calibration_package_path: Path | None = None,
    ) -> str:
        """Hash the normalized geography payload independent of source format."""

        import hashlib

        geography = self.load(
            weights_path=weights_path,
            n_records=n_records,
            n_clones=n_clones,
            geography_path=geography_path,
            blocks_path=blocks_path,
            calibration_package_path=calibration_package_path,
        )
        digest = hashlib.sha256()
        digest.update(np.asarray(geography.block_geoid, dtype="U").tobytes())
        digest.update(np.asarray(geography.cd_geoid, dtype="U").tobytes())
        digest.update(np.asarray(geography.county_fips, dtype="U").tobytes())
        digest.update(np.asarray(geography.state_fips, dtype=np.int32).tobytes())
        digest.update(str(int(geography.n_records)).encode())
        digest.update(str(int(geography.n_clones)).encode())
        return f"sha256:{digest.hexdigest()}"

    def _load_saved_geography(
        self,
        *,
        path: Path,
        n_records: int,
        n_clones: int | None,
    ) -> GeographyAssignment:
        geography = load_geography(path)
        if geography.n_records != n_records:
            raise ValueError(
                f"Geography artifact {path} has n_records={geography.n_records}, "
                f"expected {n_records}"
            )
        if n_clones is not None and geography.n_clones != n_clones:
            raise ValueError(
                f"Geography artifact {path} has n_clones={geography.n_clones}, "
                f"expected {n_clones}"
            )
        return geography

    def _load_from_package(
        self,
        *,
        path: Path,
        n_records: int,
        n_clones: int | None,
    ) -> GeographyAssignment:
        with open(path, "rb") as package_file:
            package = pickle.load(package_file)

        raw_block_geoids = package.get("block_geoid")
        raw_cd_geoids = package.get("cd_geoid")
        if raw_block_geoids is None or raw_cd_geoids is None:
            raise ValueError(
                f"Calibration package {path} does not contain geography arrays"
            )
        block_geoids = np.asarray(raw_block_geoids, dtype=str)
        cd_geoids = np.asarray(raw_cd_geoids, dtype=str)
        if len(block_geoids) == 0 or len(cd_geoids) == 0:
            raise ValueError(
                f"Calibration package {path} does not contain geography arrays"
            )
        if len(block_geoids) != len(cd_geoids):
            raise ValueError(
                f"Calibration package {path} has mismatched geography lengths "
                f"({len(block_geoids)} blocks vs {len(cd_geoids)} CDs)"
            )
        if len(block_geoids) % n_records != 0:
            raise ValueError(
                f"Calibration package {path} has {len(block_geoids)} geography rows, "
                f"not divisible by n_records={n_records}"
            )

        inferred_n_clones = len(block_geoids) // n_records
        if n_clones is not None and inferred_n_clones != n_clones:
            raise ValueError(
                f"Calibration package {path} implies n_clones={inferred_n_clones}, "
                f"expected {n_clones}"
            )

        return GeographyAssignment(
            block_geoid=block_geoids,
            cd_geoid=cd_geoids,
            county_fips=np.fromiter(
                (str(block)[:5] for block in block_geoids),
                dtype="U5",
                count=len(block_geoids),
            ),
            state_fips=np.fromiter(
                (int(str(block)[:2]) for block in block_geoids),
                dtype=np.int32,
                count=len(block_geoids),
            ),
            n_records=n_records,
            n_clones=inferred_n_clones,
        )

    def _load_from_blocks(
        self,
        *,
        path: Path,
        n_records: int,
        n_clones: int | None,
    ) -> GeographyAssignment:
        block_geoids = np.asarray(np.load(path, allow_pickle=True), dtype=str)
        if len(block_geoids) % n_records != 0:
            raise ValueError(
                f"Legacy blocks artifact {path} has {len(block_geoids)} rows, "
                f"not divisible by n_records={n_records}"
            )
        inferred_n_clones = len(block_geoids) // n_records
        if n_clones is not None and inferred_n_clones != n_clones:
            raise ValueError(
                f"Legacy blocks artifact {path} implies n_clones={inferred_n_clones}, "
                f"expected {n_clones}"
            )
        return reconstruct_geography_from_blocks(
            block_geoids=block_geoids,
            n_records=n_records,
            n_clones=inferred_n_clones,
        )
