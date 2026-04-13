"""US-specific request construction for local H5 publication.

This module owns the translation from US geography and legacy worker
items into typed ``AreaBuildRequest`` values. New request rules should
be added here rather than inside worker adapters.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from .requests import AreaBuildRequest, AreaFilter


class USAreaCatalog:
    """Construct typed local H5 requests for the current US publication flow."""

    def __init__(
        self,
        *,
        state_codes: Mapping[int, str],
        nyc_county_fips: Collection[str],
        at_large_districts: Collection[int],
    ) -> None:
        self._state_codes = dict(state_codes)
        self._state_fips_by_code = {code: fips for fips, code in state_codes.items()}
        self._nyc_county_fips = tuple(sorted(str(item) for item in nyc_county_fips))
        self._nyc_county_fips_set = set(self._nyc_county_fips)
        self._at_large_districts = {int(item) for item in at_large_districts}

    def build_state_requests(self, geography: Any) -> tuple[AreaBuildRequest, ...]:
        """Enumerate state requests from the current calibration geography."""

        cd_geoids = self._unique_cd_geoids(geography.cd_geoid)
        requests = []
        for state_fips, state_code in self._state_codes.items():
            state_cd_geoids = tuple(
                cd for cd in cd_geoids if self._state_fips_from_cd(cd) == state_fips
            )
            if not state_cd_geoids:
                continue
            requests.append(
                self._build_state_request(
                    state_code=state_code,
                    state_fips=state_fips,
                    cd_geoids=state_cd_geoids,
                )
            )
        return tuple(requests)

    def build_district_requests(self, geography: Any) -> tuple[AreaBuildRequest, ...]:
        """Enumerate district requests from the current calibration geography."""

        cd_geoids = self._unique_cd_geoids(geography.cd_geoid)
        return tuple(self._build_district_request(cd_geoid) for cd_geoid in cd_geoids)

    def build_city_requests(self, geography: Any) -> tuple[AreaBuildRequest, ...]:
        """Enumerate city requests supported by the current US flow."""

        request = self.build_city_request("NYC", geography=geography)
        if request is None:
            return ()
        return (request,)

    def build_city_request(
        self,
        city_id: str,
        *,
        geography: Any,
    ) -> AreaBuildRequest | None:
        """Build a single city request from geography-aware rules."""

        if city_id != "NYC":
            raise ValueError(f"Unknown city: {city_id}")

        nyc_cd_geoids = self._nyc_cd_geoids(geography)
        if not nyc_cd_geoids:
            return None

        return AreaBuildRequest(
            area_type="city",
            area_id="NYC",
            display_name="NYC",
            output_relative_path="cities/NYC.h5",
            filters=(
                AreaFilter(
                    geography_field="county_fips",
                    op="in",
                    value=self._nyc_county_fips,
                ),
            ),
            validation_geo_level="district",
            validation_geographic_ids=nyc_cd_geoids,
        )

    def build_national_request(self) -> AreaBuildRequest:
        """Build the single national request used by the current flow."""

        return AreaBuildRequest(
            area_type="national",
            area_id="US",
            display_name="US",
            output_relative_path="national/US.h5",
            validation_geo_level="national",
            validation_geographic_ids=("US",),
        )

    def build_request_from_work_item(
        self,
        work_item: Mapping[str, Any],
        *,
        geography: Any,
    ) -> AreaBuildRequest:
        """Convert one legacy worker item into a typed build request."""

        item_type = str(work_item["type"])
        item_id = str(work_item["id"])
        cd_geoids = self._unique_cd_geoids(geography.cd_geoid)

        if item_type == "state":
            state_fips = self._state_fips_from_code(item_id)
            state_cd_geoids = tuple(
                cd for cd in cd_geoids if self._state_fips_from_cd(cd) == state_fips
            )
            if not state_cd_geoids:
                raise ValueError(f"No CDs for {item_id}")
            return self._build_state_request(
                state_code=item_id,
                state_fips=state_fips,
                cd_geoids=state_cd_geoids,
            )

        if item_type == "district":
            geoid = self._resolve_district_geoid(item_id=item_id, cd_geoids=cd_geoids)
            return self._build_district_request(geoid)

        if item_type == "city":
            request = self.build_city_request(item_id, geography=geography)
            if request is None:
                raise ValueError(f"No matching geography found for city: {item_id}")
            return request

        if item_type == "national":
            if item_id != "US":
                raise ValueError(f"Unknown national request: {item_id}")
            return self.build_national_request()

        raise ValueError(f"Unknown item type: {item_type}")

    def build_requests_from_work_items(
        self,
        work_items: Sequence[Mapping[str, Any]],
        *,
        geography: Any,
    ) -> tuple[AreaBuildRequest, ...]:
        """Convert a legacy worker batch into typed build requests."""

        return tuple(
            self.build_request_from_work_item(item, geography=geography)
            for item in work_items
        )

    def _build_state_request(
        self,
        *,
        state_code: str,
        state_fips: int,
        cd_geoids: tuple[str, ...],
    ) -> AreaBuildRequest:
        return AreaBuildRequest(
            area_type="state",
            area_id=state_code,
            display_name=state_code,
            output_relative_path=f"states/{state_code}.h5",
            filters=(
                AreaFilter(
                    geography_field="cd_geoid",
                    op="in",
                    value=cd_geoids,
                ),
            ),
            validation_geo_level="state",
            validation_geographic_ids=(str(state_fips),),
        )

    def _build_district_request(self, cd_geoid: str) -> AreaBuildRequest:
        friendly_name = self.get_district_friendly_name(cd_geoid)
        return AreaBuildRequest(
            area_type="district",
            area_id=friendly_name,
            display_name=friendly_name,
            output_relative_path=f"districts/{friendly_name}.h5",
            filters=(
                AreaFilter(
                    geography_field="cd_geoid",
                    op="in",
                    value=(cd_geoid,),
                ),
            ),
            validation_geo_level="district",
            validation_geographic_ids=(str(cd_geoid),),
        )

    def get_district_friendly_name(self, cd_geoid: str) -> str:
        """Convert a congressional district GEOID into its friendly name."""

        cd_int = int(cd_geoid)
        state_fips = cd_int // 100
        district_num = cd_int % 100
        if district_num in self._at_large_districts:
            district_num = 1
        state_code = self._state_codes.get(state_fips, str(state_fips))
        return f"{state_code}-{district_num:02d}"

    def _resolve_district_geoid(
        self,
        *,
        item_id: str,
        cd_geoids: tuple[str, ...],
    ) -> str:
        state_code, dist_num = item_id.split("-", 1)
        state_fips = self._state_fips_from_code(state_code)
        candidate = f"{state_fips}{int(dist_num):02d}"
        if candidate in cd_geoids:
            return candidate

        state_cd_geoids = tuple(
            cd for cd in cd_geoids if self._state_fips_from_cd(cd) == state_fips
        )
        if len(state_cd_geoids) == 1:
            return state_cd_geoids[0]

        raise ValueError(
            f"CD {candidate} not found and state {state_code} "
            f"has {len(state_cd_geoids)} CDs"
        )

    def _nyc_cd_geoids(self, geography: Any) -> tuple[str, ...]:
        county_fips = getattr(geography, "county_fips", None)
        if county_fips is None:
            return ()
        nyc_cd_geoids = {
            str(cd_geoid)
            for cd_geoid, county in zip(geography.cd_geoid, county_fips, strict=False)
            if str(county) in self._nyc_county_fips_set
        }
        return tuple(sorted(nyc_cd_geoids))

    def _state_fips_from_code(self, state_code: str) -> int:
        try:
            return self._state_fips_by_code[state_code]
        except KeyError as exc:
            raise ValueError(f"Unknown state code: {state_code}") from exc

    @staticmethod
    def _state_fips_from_cd(cd_geoid: str) -> int:
        return int(cd_geoid) // 100

    @staticmethod
    def _unique_cd_geoids(cd_geoids: Sequence[Any]) -> tuple[str, ...]:
        return tuple(sorted({str(cd_geoid) for cd_geoid in cd_geoids}))
