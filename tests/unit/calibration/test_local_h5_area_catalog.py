from tests.unit.calibration.fixtures.test_local_h5_area_catalog import (
    load_area_catalog_exports,
    make_geography,
)


exports = load_area_catalog_exports()
USAreaCatalog = exports["USAreaCatalog"]


def make_catalog():
    return USAreaCatalog(
        state_codes={1: "AL", 2: "AK", 36: "NY"},
        nyc_county_fips={"36005", "36047", "36061", "36081", "36085"},
        at_large_districts={0, 98},
    )


def test_build_state_requests_enumerates_paths_and_validation_ids():
    catalog = make_catalog()
    geography = make_geography(cd_geoids=["101", "102", "201"])

    requests = catalog.build_state_requests(geography)

    assert [request.area_id for request in requests] == ["AL", "AK"]
    assert requests[0].output_relative_path == "states/AL.h5"
    assert requests[0].validation_geographic_ids == ("1",)
    assert requests[0].filters[0].value == ("101", "102")


def test_build_district_requests_uses_friendly_names_for_at_large_geos():
    catalog = make_catalog()
    geography = make_geography(cd_geoids=["298"])

    requests = catalog.build_district_requests(geography)

    assert len(requests) == 1
    assert requests[0].area_id == "AK-01"
    assert requests[0].output_relative_path == "districts/AK-01.h5"
    assert requests[0].validation_geographic_ids == ("298",)


def test_build_city_requests_emits_nyc_request_with_district_validation_ids():
    catalog = make_catalog()
    geography = make_geography(
        cd_geoids=["3601", "3603", "101"],
        county_fips=["36061", "36081", "01001"],
    )

    requests = catalog.build_city_requests(geography)

    assert len(requests) == 1
    assert requests[0].area_id == "NYC"
    assert requests[0].output_relative_path == "cities/NYC.h5"
    assert requests[0].validation_geo_level == "district"
    assert requests[0].validation_geographic_ids == ("3601", "3603")


def test_build_national_request_returns_canonical_us_request():
    catalog = make_catalog()

    request = catalog.build_national_request()

    assert request.area_type == "national"
    assert request.area_id == "US"
    assert request.output_relative_path == "national/US.h5"
    assert request.validation_geographic_ids == ("US",)


def test_build_request_from_work_item_preserves_legacy_district_fallback():
    catalog = make_catalog()
    geography = make_geography(cd_geoids=["298"])

    request = catalog.build_request_from_work_item(
        {"type": "district", "id": "AK-01"},
        geography=geography,
    )

    assert request.area_id == "AK-01"
    assert request.output_relative_path == "districts/AK-01.h5"
    assert request.validation_geographic_ids == ("298",)
