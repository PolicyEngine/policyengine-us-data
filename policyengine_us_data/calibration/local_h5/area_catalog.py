"""Concrete US area request catalog for local H5 publishing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from policyengine_us_data.calibration.calibration_utils import (
    STATE_CODES,
    get_all_cds_from_database,
)

from .contracts import AreaBuildRequest, AreaFilter


AT_LARGE_DISTRICTS = {0, 98}
NYC_COUNTY_FIPS = {"36005", "36047", "36061", "36081", "36085"}
CITY_WEIGHTS = {"NYC": 11}


@dataclass(frozen=True)
class USCatalogEntry:
    request: AreaBuildRequest
    weight: int

    @property
    def key(self) -> str:
        return f"{self.request.area_type}:{self.request.area_id}"

    def to_partition_item(self) -> dict[str, object]:
        return {
            "type": self.request.area_type,
            "id": self.request.area_id,
            "weight": self.weight,
        }


class USAreaCatalog:
    """Build concrete US local-area requests from the calibration artifacts."""

    def __init__(
        self,
        *,
        state_codes: Mapping[int, str] = STATE_CODES,
        at_large_districts: set[int] | None = None,
        nyc_county_fips: set[str] | None = None,
        city_weights: Mapping[str, int] | None = None,
    ) -> None:
        self.state_codes = dict(state_codes)
        self.at_large_districts = set(at_large_districts or AT_LARGE_DISTRICTS)
        self.nyc_county_fips = set(nyc_county_fips or NYC_COUNTY_FIPS)
        self.city_weights = dict(city_weights or CITY_WEIGHTS)

    def load_regional_entries(self, db_uri: str) -> tuple[USCatalogEntry, ...]:
        cds = tuple(str(cd) for cd in get_all_cds_from_database(db_uri))
        return self.regional_entries_from_cds(cds)

    def regional_entries_from_cds(
        self,
        cds: Sequence[str],
    ) -> tuple[USCatalogEntry, ...]:
        cds = tuple(str(cd) for cd in cds)
        districts = tuple(self._district_friendly_name(cd) for cd in cds)
        cds_per_state = Counter(district.split("-")[0] for district in districts)
        states_with_cds = [
            state_code
            for state_code in self.state_codes.values()
            if cds_per_state.get(state_code, 0) > 0
        ]

        entries: list[USCatalogEntry] = []
        for state_code in states_with_cds:
            entries.append(
                USCatalogEntry(
                    request=AreaBuildRequest(
                        area_type="state",
                        area_id=state_code,
                        display_name=state_code,
                        output_relative_path=f"states/{state_code}.h5",
                    ),
                    weight=cds_per_state.get(state_code, 1),
                )
            )

        for cd, friendly_name in zip(cds, districts):
            entries.append(
                USCatalogEntry(
                    request=AreaBuildRequest(
                        area_type="district",
                        area_id=friendly_name,
                        display_name=friendly_name,
                        output_relative_path=f"districts/{friendly_name}.h5",
                    ),
                    weight=1,
                )
            )

        entries.append(
            USCatalogEntry(
                request=AreaBuildRequest(
                    area_type="city",
                    area_id="NYC",
                    display_name="NYC",
                    output_relative_path="cities/NYC.h5",
                ),
                weight=self.city_weights.get("NYC", 3),
            )
        )
        return tuple(entries)

    def national_entry(self) -> USCatalogEntry:
        return USCatalogEntry(
            request=AreaBuildRequest.national(),
            weight=1,
        )

    def _district_friendly_name(self, cd_geoid: str) -> str:
        cd_int = int(cd_geoid)
        state_fips = cd_int // 100
        district_num = cd_int % 100
        if district_num in self.at_large_districts:
            district_num = 1
        state_code = self.state_codes.get(state_fips, str(state_fips))
        return f"{state_code}-{district_num:02d}"

    def filters_for_request(
        self,
        request: AreaBuildRequest,
        *,
        cds: Sequence[str],
    ) -> tuple[AreaFilter, ...]:
        if request.area_type == "state":
            state_code = request.area_id
            state_fips = self._state_fips_for_code(state_code)
            cd_subset = tuple(cd for cd in cds if int(cd) // 100 == state_fips)
            return (
                AreaFilter(
                    geography_field="cd_geoid",
                    op="in",
                    value=cd_subset,
                ),
            )

        if request.area_type == "district":
            matching_cd = next(
                cd for cd in cds if self._district_friendly_name(cd) == request.area_id
            )
            return (
                AreaFilter(
                    geography_field="cd_geoid",
                    op="in",
                    value=(matching_cd,),
                ),
            )

        if request.area_type == "city":
            return (
                AreaFilter(
                    geography_field="county_fips",
                    op="in",
                    value=tuple(sorted(self.nyc_county_fips)),
                ),
            )

        return ()

    def with_filters(
        self,
        entry: USCatalogEntry,
        *,
        cds: Sequence[str],
        geography=None,
    ) -> USCatalogEntry:
        request = entry.request
        filters = self.filters_for_request(request, cds=cds)
        validation_geo_level = None
        validation_geographic_ids: tuple[str, ...] = ()

        if request.area_type == "state":
            validation_geo_level = "state"
            validation_geographic_ids = (str(self._state_fips_for_code(request.area_id)),)
        elif request.area_type == "district":
            validation_geo_level = "district"
            validation_geographic_ids = tuple(item.value[0] for item in filters)
        elif request.area_type == "city":
            validation_geo_level = "district"
            validation_geographic_ids = self._city_validation_cd_geoids(
                cds=cds,
                geography=geography,
            )
        elif request.area_type == "national":
            validation_geo_level = "national"
            validation_geographic_ids = ("US",)

        return USCatalogEntry(
            request=AreaBuildRequest(
                area_type=request.area_type,
                area_id=request.area_id,
                display_name=request.display_name,
                output_relative_path=request.output_relative_path,
                filters=filters,
                validation_geo_level=validation_geo_level,
                validation_geographic_ids=validation_geographic_ids,
                metadata=dict(request.metadata),
            ),
            weight=entry.weight,
        )

    def resolved_regional_entries(
        self,
        db_uri: str,
        *,
        geography=None,
    ) -> tuple[USCatalogEntry, ...]:
        cds = tuple(str(cd) for cd in get_all_cds_from_database(db_uri))
        return tuple(
            self.with_filters(entry, cds=cds, geography=geography)
            for entry in self.regional_entries_from_cds(cds)
        )

    def resolved_national_entry(self) -> USCatalogEntry:
        return self.with_filters(self.national_entry(), cds=(), geography=None)

    def _state_fips_for_code(self, state_code: str) -> int:
        for fips, code in self.state_codes.items():
            if code == state_code:
                return int(fips)
        raise ValueError(f"Unknown state code: {state_code}")

    def _city_validation_cd_geoids(
        self,
        *,
        cds: Sequence[str],
        geography,
    ) -> tuple[str, ...]:
        if geography is None:
            return ()

        county_fips = getattr(geography, "county_fips", None)
        cd_geoids = getattr(geography, "cd_geoid", None)
        if county_fips is None or cd_geoids is None:
            return ()

        available_cds = set(str(cd) for cd in cds)
        return tuple(
            sorted(
                {
                    str(cd)
                    for cd, county in zip(cd_geoids, county_fips)
                    if str(county) in self.nyc_county_fips
                    and str(cd) in available_cds
                }
            )
        )
