from policyengine_us import CountryTaxBenefitSystem


def test_policyengine_us_defines_housing_assistance_takeup_input():
    tax_benefit_system = CountryTaxBenefitSystem()
    variable = tax_benefit_system.variables["takes_up_housing_assistance_if_eligible"]

    assert variable.entity.key == "spm_unit"
    assert not getattr(variable, "formulas", None)
    assert not getattr(variable, "adds", None)
    assert not getattr(variable, "subtracts", None)


def test_policyengine_us_defines_housing_assistance_formulas():
    tax_benefit_system = CountryTaxBenefitSystem()

    for variable_name in (
        "housing_assistance",
        "spm_unit_capped_housing_subsidy",
    ):
        variable = tax_benefit_system.variables[variable_name]

        assert variable.entity.key == "spm_unit"
        assert getattr(variable, "formulas", None)


def test_policyengine_us_defines_housing_income_limit_contract():
    from policyengine_us.variables.gov.hud.is_eligible_for_housing_assistance import (
        housing_assistance_eligibility_from_income_limits,
    )

    assert callable(housing_assistance_eligibility_from_income_limits)

    tax_benefit_system = CountryTaxBenefitSystem()
    for variable_name in (
        "hud_extremely_low_income_limit",
        "hud_very_low_income_limit",
        "hud_low_income_limit",
    ):
        variable = tax_benefit_system.variables[variable_name]
        assert variable.entity.key == "spm_unit"
