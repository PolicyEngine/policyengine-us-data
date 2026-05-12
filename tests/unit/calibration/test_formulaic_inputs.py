from policyengine_us_data.calibration.formulaic_inputs import (
    FORMULAIC_SPM_INPUTS_TO_DROP,
    drop_formulaic_spm_inputs,
)


def test_drop_formulaic_spm_inputs_removes_poverty_formula_outputs():
    variable_names = {
        "person_in_poverty",
        "in_poverty",
        "in_deep_poverty",
        "spm_unit_spm_threshold",
        "spm_unit_geographic_adjustment",
        "household_weight",
    }

    drop_formulaic_spm_inputs(variable_names)

    assert variable_names == {"household_weight"}


def test_formulaic_spm_inputs_includes_person_poverty():
    assert "person_in_poverty" in FORMULAIC_SPM_INPUTS_TO_DROP
