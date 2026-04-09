"""US-specific payload augmentation for local H5 publishing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from policyengine_us_data.calibration.block_assignment import (
    derive_geography_from_blocks,
)
from policyengine_us_data.calibration.calibration_utils import (
    calculate_spm_thresholds_vectorized,
    load_cd_geoadj_values,
)
from policyengine_us_data.calibration.local_h5.reindexing import (
    ReindexedEntities,
)
from policyengine_us_data.calibration.local_h5.selection import CloneSelection
from policyengine_us_data.calibration.local_h5.source_dataset import (
    SourceDatasetSnapshot,
)
from policyengine_us_data.utils.takeup import (
    apply_block_takeup_to_arrays,
    reported_subsidized_marketplace_by_tax_unit,
)


def _default_county_name_lookup(county_indices: np.ndarray) -> np.ndarray:
    from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
        County,
    )

    return np.asarray(
        [County._member_names_[int(index)] for index in county_indices],
        dtype="S",
    )


def build_reported_takeup_anchors(
    data: dict[str, dict[int | str, np.ndarray]],
    time_period: int | str,
) -> dict[str, np.ndarray]:
    reported_anchors: dict[str, np.ndarray] = {}
    if (
        "reported_has_subsidized_marketplace_health_coverage_at_interview" in data
        and "person_tax_unit_id" in data
        and "tax_unit_id" in data
        and time_period
        in data["reported_has_subsidized_marketplace_health_coverage_at_interview"]
        and time_period in data["person_tax_unit_id"]
        and time_period in data["tax_unit_id"]
    ):
        reported_anchors["takes_up_aca_if_eligible"] = (
            reported_subsidized_marketplace_by_tax_unit(
                data["person_tax_unit_id"][time_period],
                data["tax_unit_id"][time_period],
                data[
                    "reported_has_subsidized_marketplace_health_coverage_at_interview"
                ][time_period],
            )
        )
    if (
        "has_medicaid_health_coverage_at_interview" in data
        and time_period in data["has_medicaid_health_coverage_at_interview"]
    ):
        reported_anchors["takes_up_medicaid_if_eligible"] = data[
            "has_medicaid_health_coverage_at_interview"
        ][time_period].astype(bool)
    return reported_anchors


@dataclass(frozen=True)
class USAugmentationService:
    geography_lookup: Callable[[np.ndarray], Mapping[str, np.ndarray]] = (
        derive_geography_from_blocks
    )
    county_name_lookup: Callable[[np.ndarray], np.ndarray] = (
        _default_county_name_lookup
    )
    cd_geoadj_loader: Callable[[Sequence[str]], Mapping[str, float]] = (
        load_cd_geoadj_values
    )
    threshold_calculator: Callable[..., np.ndarray] = (
        calculate_spm_thresholds_vectorized
    )
    takeup_fn: Callable[..., Mapping[str, np.ndarray]] = (
        apply_block_takeup_to_arrays
    )

    def apply_geography(
        self,
        data: dict[str, dict[int | str, np.ndarray]],
        *,
        time_period: int | str,
        active_blocks: np.ndarray,
        active_clone_cds: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        unique_blocks, block_inv = np.unique(active_blocks, return_inverse=True)
        unique_geo = self.geography_lookup(unique_blocks)
        clone_geo = {
            key: np.asarray(values)[block_inv]
            for key, values in unique_geo.items()
        }

        data["state_fips"] = {
            time_period: clone_geo["state_fips"].astype(np.int32)
        }
        data["county"] = {
            time_period: self.county_name_lookup(clone_geo["county_index"])
        }
        data["county_fips"] = {
            time_period: clone_geo["county_fips"].astype(np.int32)
        }

        for variable in (
            "block_geoid",
            "tract_geoid",
            "cbsa_code",
            "sldu",
            "sldl",
            "place_fips",
            "vtd",
            "puma",
            "zcta",
        ):
            if variable in clone_geo:
                data[variable] = {
                    time_period: clone_geo[variable].astype("S")
                }

        data["congressional_district_geoid"] = {
            time_period: np.asarray(
                [int(cd) for cd in active_clone_cds],
                dtype=np.int32,
            )
        }
        return clone_geo

    def apply_zip_code_patch(
        self,
        data: dict[str, dict[int | str, np.ndarray]],
        *,
        time_period: int | str,
        county_fips: np.ndarray,
    ) -> None:
        la_mask = county_fips.astype(str) == "06037"
        if not la_mask.any():
            return
        zip_codes = np.full(len(la_mask), "UNKNOWN")
        zip_codes[la_mask] = "90001"
        data["zip_code"] = {time_period: zip_codes.astype("S")}

    def apply_spm_thresholds(
        self,
        data: dict[str, dict[int | str, np.ndarray]],
        *,
        time_period: int,
        active_clone_cds: np.ndarray,
        source: SourceDatasetSnapshot,
        reindexed: ReindexedEntities,
    ) -> None:
        provider = source.variable_provider
        unique_cds_list = sorted(set(active_clone_cds))
        cd_geoadj_values = self.cd_geoadj_loader(unique_cds_list)

        spm_entities_per_clone = reindexed.entities_per_clone["spm_unit"]
        spm_clone_ids = np.repeat(
            np.arange(len(spm_entities_per_clone), dtype=np.int64),
            spm_entities_per_clone,
        )
        spm_unit_geoadj = np.asarray(
            [
                cd_geoadj_values[str(active_clone_cds[clone_id])]
                for clone_id in spm_clone_ids
            ],
            dtype=np.float64,
        )

        person_ages = provider.calculate("age", map_to="person").values[
            reindexed.person_source_indices
        ]
        spm_tenure_periods = provider.get_known_periods("spm_unit_tenure_type")
        if spm_tenure_periods:
            raw_tenure = provider.get_array(
                "spm_unit_tenure_type",
                spm_tenure_periods[0],
            )
            if hasattr(raw_tenure, "decode_to_str"):
                raw_tenure = raw_tenure.decode_to_str().astype("S")
            else:
                raw_tenure = np.asarray(raw_tenure).astype("S")
            spm_tenure_cloned = raw_tenure[
                reindexed.entity_source_indices["spm_unit"]
            ]
        else:
            spm_tenure_cloned = np.full(
                len(reindexed.entity_source_indices["spm_unit"]),
                b"RENTER",
                dtype="S30",
            )

        data["spm_unit_spm_threshold"] = {
            time_period: self.threshold_calculator(
                person_ages=person_ages,
                person_spm_unit_ids=reindexed.new_person_entity_ids["spm_unit"],
                spm_unit_tenure_types=spm_tenure_cloned,
                spm_unit_geoadj=spm_unit_geoadj,
                year=time_period,
            )
        }

    def apply_takeup(
        self,
        data: dict[str, dict[int | str, np.ndarray]],
        *,
        time_period: int | str,
        takeup_filter: Sequence[str] | None,
        selection: CloneSelection,
        source: SourceDatasetSnapshot,
        reindexed: ReindexedEntities,
        clone_geo: Mapping[str, np.ndarray],
    ) -> None:
        entity_hh_indices = {
            "person": np.repeat(
                np.arange(selection.n_household_clones, dtype=np.int64),
                reindexed.persons_per_clone,
            ).astype(np.int64),
            "tax_unit": np.repeat(
                np.arange(selection.n_household_clones, dtype=np.int64),
                reindexed.entities_per_clone["tax_unit"],
            ).astype(np.int64),
            "spm_unit": np.repeat(
                np.arange(selection.n_household_clones, dtype=np.int64),
                reindexed.entities_per_clone["spm_unit"],
            ).astype(np.int64),
        }
        entity_counts = {
            "person": len(reindexed.person_source_indices),
            "tax_unit": len(reindexed.entity_source_indices["tax_unit"]),
            "spm_unit": len(reindexed.entity_source_indices["spm_unit"]),
        }
        original_hh_ids = source.household_ids[
            selection.active_household_indices
        ].astype(np.int64)
        reported_anchors = build_reported_takeup_anchors(data, time_period)

        takeup_results = self.takeup_fn(
            hh_blocks=selection.active_block_geoids,
            hh_state_fips=clone_geo["state_fips"].astype(np.int32),
            hh_ids=original_hh_ids,
            hh_clone_indices=selection.active_clone_indices.astype(np.int64),
            entity_hh_indices=entity_hh_indices,
            entity_counts=entity_counts,
            time_period=time_period,
            takeup_filter=takeup_filter,
            reported_anchors=reported_anchors,
        )
        for variable, values in takeup_results.items():
            data[variable] = {time_period: values}

    def apply_all(
        self,
        data: dict[str, dict[int | str, np.ndarray]],
        *,
        time_period: int,
        selection: CloneSelection,
        source: SourceDatasetSnapshot,
        reindexed: ReindexedEntities,
        takeup_filter: Sequence[str] | None,
    ) -> dict[str, dict[int | str, np.ndarray]]:
        clone_geo = self.apply_geography(
            data,
            time_period=time_period,
            active_blocks=selection.active_block_geoids,
            active_clone_cds=selection.active_cd_geoids,
        )
        self.apply_zip_code_patch(
            data,
            time_period=time_period,
            county_fips=clone_geo["county_fips"],
        )
        self.apply_spm_thresholds(
            data,
            time_period=time_period,
            active_clone_cds=selection.active_cd_geoids,
            source=source,
            reindexed=reindexed,
        )
        self.apply_takeup(
            data,
            time_period=time_period,
            takeup_filter=takeup_filter,
            selection=selection,
            source=source,
            reindexed=reindexed,
            clone_geo=clone_geo,
        )
        return data
