"""Load and serialize calibration-package geography for H5 publishing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class LoadedPackageGeography:
    """Resolved geography plus provenance for publisher logging/tests."""

    geography: Any
    source: str
    warnings: tuple[str, ...] = ()


class CalibrationPackageGeographyLoader:
    """Read exact geography assignments from calibration packages.

    This loader prefers the newer serialized ``geography`` payload,
    but can still reconstruct a ``GeographyAssignment`` from the older
    top-level ``block_geoid``/``cd_geoid`` arrays.
    """

    def serialize_geography(self, geography: Any) -> dict[str, Any]:
        return {
            "block_geoid": np.asarray(geography.block_geoid, dtype=str),
            "cd_geoid": np.asarray(geography.cd_geoid, dtype=str),
            "county_fips": np.asarray(geography.county_fips, dtype=str),
            "state_fips": np.asarray(geography.state_fips, dtype=np.int64),
            "n_records": int(geography.n_records),
            "n_clones": int(geography.n_clones),
        }

    def load(
        self,
        package_path: str | Path,
        *,
        fallback_n_records: int | None = None,
        fallback_n_clones: int | None = None,
    ) -> LoadedPackageGeography | None:
        import pickle

        with open(package_path, "rb") as f:
            package = pickle.load(f)
        return self.load_from_package_dict(
            package,
            fallback_n_records=fallback_n_records,
            fallback_n_clones=fallback_n_clones,
        )

    def load_from_package_dict(
        self,
        package: Mapping[str, Any],
        *,
        fallback_n_records: int | None = None,
        fallback_n_clones: int | None = None,
    ) -> LoadedPackageGeography | None:
        payload = package.get("geography")
        if isinstance(payload, Mapping):
            geography = self._build_from_serialized_payload(
                payload,
                fallback_n_records=fallback_n_records,
                fallback_n_clones=fallback_n_clones,
            )
            return LoadedPackageGeography(
                geography=geography,
                source="serialized_package",
            )

        if package.get("block_geoid") is not None and package.get("cd_geoid") is not None:
            geography = self._build_from_legacy_package(
                package,
                fallback_n_records=fallback_n_records,
                fallback_n_clones=fallback_n_clones,
            )
            return LoadedPackageGeography(
                geography=geography,
                source="legacy_package",
                warnings=(
                    "Calibration package does not include serialized geography; "
                    "reconstructed from legacy arrays.",
                ),
            )

        return None

    def resolve_for_weights(
        self,
        *,
        package_path: str | Path | None,
        weights_length: int,
        n_records: int,
        n_clones: int,
        seed: int,
    ) -> LoadedPackageGeography:
        warnings: list[str] = []

        if package_path:
            package_path = Path(package_path)
            if package_path.exists():
                load_error = None
                try:
                    loaded = self.load(
                        package_path,
                        fallback_n_records=n_records,
                        fallback_n_clones=n_clones,
                    )
                except ValueError as error:
                    loaded = None
                    load_error = str(error)
                if loaded is not None:
                    actual_len = len(np.asarray(loaded.geography.block_geoid))
                    if actual_len == weights_length:
                        if loaded.warnings:
                            warnings.extend(loaded.warnings)
                        return LoadedPackageGeography(
                            geography=loaded.geography,
                            source=loaded.source,
                            warnings=tuple(warnings),
                        )
                    warnings.append(
                        "Calibration package geography length "
                        f"({actual_len}) does not match weights length "
                        f"({weights_length}); regenerating from seed."
                    )
                else:
                    if load_error is not None:
                        warnings.append(
                            "Calibration package geography could not be loaded "
                            f"({load_error}); regenerating from seed."
                        )
                    else:
                        warnings.append(
                            "Calibration package does not include usable geography; "
                            "regenerating from seed."
                        )
            else:
                warnings.append(
                    f"Calibration package not found at {package_path}; regenerating from seed."
                )

        return LoadedPackageGeography(
            geography=self._generate_geography(
                n_records=n_records,
                n_clones=n_clones,
                seed=seed,
            ),
            source="generated",
            warnings=tuple(warnings),
        )

    def _build_from_serialized_payload(
        self,
        payload: Mapping[str, Any],
        *,
        fallback_n_records: int | None,
        fallback_n_clones: int | None,
    ) -> Any:
        blocks = self._string_array(payload["block_geoid"])
        cds = self._string_array(payload["cd_geoid"])
        n_records, n_clones = self._infer_dimensions(
            total_length=len(blocks),
            n_records=payload.get("n_records"),
            n_clones=payload.get("n_clones"),
            fallback_n_records=fallback_n_records,
            fallback_n_clones=fallback_n_clones,
        )
        county_fips = payload.get("county_fips")
        if county_fips is None:
            county_fips = self._derive_county_fips(blocks)
        else:
            county_fips = self._string_array(county_fips)
        state_fips = payload.get("state_fips")
        if state_fips is None:
            state_fips = self._derive_state_fips(blocks)
        else:
            state_fips = np.asarray(state_fips, dtype=np.int64)
        return self._build_assignment(
            block_geoid=blocks,
            cd_geoid=cds,
            county_fips=county_fips,
            state_fips=state_fips,
            n_records=n_records,
            n_clones=n_clones,
        )

    def _build_from_legacy_package(
        self,
        package: Mapping[str, Any],
        *,
        fallback_n_records: int | None,
        fallback_n_clones: int | None,
    ) -> Any:
        blocks = self._string_array(package["block_geoid"])
        cds = self._string_array(package["cd_geoid"])
        metadata = package.get("metadata") or {}
        n_records, n_clones = self._infer_dimensions(
            total_length=len(blocks),
            n_records=metadata.get("base_n_records"),
            n_clones=metadata.get("n_clones"),
            fallback_n_records=fallback_n_records,
            fallback_n_clones=fallback_n_clones,
        )
        return self._build_assignment(
            block_geoid=blocks,
            cd_geoid=cds,
            county_fips=self._derive_county_fips(blocks),
            state_fips=self._derive_state_fips(blocks),
            n_records=n_records,
            n_clones=n_clones,
        )

    def _infer_dimensions(
        self,
        *,
        total_length: int,
        n_records: int | None,
        n_clones: int | None,
        fallback_n_records: int | None,
        fallback_n_clones: int | None,
    ) -> tuple[int, int]:
        resolved_records = self._as_int(n_records) or self._as_int(fallback_n_records)
        resolved_clones = self._as_int(n_clones) or self._as_int(fallback_n_clones)

        if resolved_records is None and resolved_clones is not None:
            if total_length % resolved_clones != 0:
                raise ValueError(
                    "Cannot infer base record count from package geometry length "
                    f"{total_length} and n_clones={resolved_clones}"
                )
            resolved_records = total_length // resolved_clones
        if resolved_clones is None and resolved_records is not None:
            if total_length % resolved_records != 0:
                raise ValueError(
                    "Cannot infer clone count from package geometry length "
                    f"{total_length} and n_records={resolved_records}"
                )
            resolved_clones = total_length // resolved_records

        if resolved_records is None or resolved_clones is None:
            raise ValueError(
                "Calibration package geography is missing n_records/n_clones metadata"
            )
        if resolved_records * resolved_clones != total_length:
            raise ValueError(
                "Calibration package geography dimensions do not match array length: "
                f"{resolved_records} x {resolved_clones} != {total_length}"
            )
        return resolved_records, resolved_clones

    def _generate_geography(self, *, n_records: int, n_clones: int, seed: int) -> Any:
        from policyengine_us_data.calibration.clone_and_assign import (
            assign_random_geography,
        )

        return assign_random_geography(
            n_records=n_records,
            n_clones=n_clones,
            seed=seed,
        )

    def _build_assignment(
        self,
        *,
        block_geoid: np.ndarray,
        cd_geoid: np.ndarray,
        county_fips: np.ndarray,
        state_fips: np.ndarray,
        n_records: int,
        n_clones: int,
    ) -> Any:
        from policyengine_us_data.calibration.clone_and_assign import (
            GeographyAssignment,
        )

        return GeographyAssignment(
            block_geoid=block_geoid,
            cd_geoid=cd_geoid,
            county_fips=county_fips,
            state_fips=state_fips,
            n_records=n_records,
            n_clones=n_clones,
        )

    def _derive_county_fips(self, blocks: np.ndarray) -> np.ndarray:
        return np.asarray([str(block)[:5] for block in blocks], dtype=str)

    def _derive_state_fips(self, blocks: np.ndarray) -> np.ndarray:
        return np.asarray([int(str(block)[:2]) for block in blocks], dtype=np.int64)

    def _string_array(self, values: Any) -> np.ndarray:
        return np.asarray(values, dtype=str)

    def _as_int(self, value: Any) -> int | None:
        if value is None:
            return None
        return int(value)
