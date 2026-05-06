from policyengine_us_data.db.etl_irs_soi import _get_eitc_recipient_constraints


def _constraint_tuples(geo_info):
    return {
        (
            constraint.constraint_variable,
            constraint.operation,
            constraint.value,
        )
        for constraint in _get_eitc_recipient_constraints(geo_info)
    }


def test_national_eitc_targets_are_limited_to_recipients():
    assert _constraint_tuples({"type": "national"}) == {
        ("tax_unit_is_filer", "==", "1"),
        ("eitc", ">", "0"),
    }


def test_state_eitc_targets_are_limited_to_state_recipients():
    assert _constraint_tuples({"type": "state", "state_fips": 6}) == {
        ("tax_unit_is_filer", "==", "1"),
        ("eitc", ">", "0"),
        ("state_fips", "==", "6"),
    }


def test_district_eitc_targets_are_limited_to_district_recipients():
    assert _constraint_tuples(
        {
            "type": "district",
            "congressional_district_geoid": "5000100US0601",
        }
    ) == {
        ("tax_unit_is_filer", "==", "1"),
        ("eitc", ">", "0"),
        ("congressional_district_geoid", "==", "5000100US0601"),
    }
