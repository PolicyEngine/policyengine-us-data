import json

import pytest

from tests.unit.calibration.fixtures.test_local_h5_requests import (
    load_requests_exports,
)


requests = load_requests_exports()
AreaBuildRequest = requests["AreaBuildRequest"]
AreaFilter = requests["AreaFilter"]
make_national_request = requests["make_national_request"]


def test_area_filter_validates_eq_vs_in_shape():
    AreaFilter(geography_field="state_fips", op="eq", value=6)
    AreaFilter(geography_field="county_fips", op="in", value=("06037", "06059"))

    with pytest.raises(ValueError, match="must be a tuple"):
        AreaFilter(geography_field="county_fips", op="in", value="06037")

    with pytest.raises(ValueError, match="must not be a tuple"):
        AreaFilter(geography_field="state_fips", op="eq", value=(6, 12))


def test_area_build_request_requires_validation_level_if_ids_provided():
    with pytest.raises(ValueError, match="validation_geo_level"):
        AreaBuildRequest(
            area_type="district",
            area_id="CA-12",
            display_name="CA-12",
            output_relative_path="districts/CA-12.h5",
            validation_geographic_ids=("612",),
        )


def test_area_build_request_round_trips_through_json_dict():
    request = AreaBuildRequest(
        area_type="state",
        area_id="CA",
        display_name="California",
        output_relative_path="states/CA.h5",
        filters=(AreaFilter(geography_field="state_fips", op="eq", value=6),),
        validation_geo_level="state",
        validation_geographic_ids=("6",),
        metadata={"takeup_filter": "snap,ssi"},
    )

    roundtrip = AreaBuildRequest.from_dict(json.loads(json.dumps(request.to_dict())))

    assert roundtrip == request


def test_national_request_fixture_builds_canonical_request():
    request = make_national_request(AreaBuildRequest)

    assert request.area_type == "national"
    assert request.area_id == "US"
    assert request.output_relative_path == "national/US.h5"
    assert request.validation_geo_level == "national"
    assert request.validation_geographic_ids == ("US",)
    assert request.filters == ()
