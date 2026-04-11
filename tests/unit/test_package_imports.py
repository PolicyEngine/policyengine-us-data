import subprocess
import sys
from pathlib import Path

import numpy as np

import policyengine_us_data


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_create_database_tables_imports_cleanly_in_fresh_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import policyengine_us_data.db.create_database_tables",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_validate_national_h5_imports_cleanly_in_fresh_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import policyengine_us_data.calibration.validate_national_h5",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_package_root_lazily_exports_dataset_classes():
    assert policyengine_us_data.EnhancedCPS_2024.__name__ == "EnhancedCPS_2024"
    assert policyengine_us_data.ExtendedCPS_2024.__name__ == "ExtendedCPS_2024"
    assert policyengine_us_data.CPS_2024.__name__ == "CPS_2024"
    assert policyengine_us_data.PUF_2024.__name__ == "PUF_2024"


def test_policyengine_us_compat_variables_are_registered():
    from policyengine_us import CountryTaxBenefitSystem

    tbs = CountryTaxBenefitSystem()

    for variable in [
        "sstb_self_employment_income",
        "sstb_w2_wages_from_qualified_business",
        "sstb_unadjusted_basis_qualified_property",
        "sstb_self_employment_income_would_be_qualified",
        "sstb_qualified_business_income",
        "total_self_employment_income",
    ]:
        assert variable in tbs.variables

    assert (
        tbs.variables["sstb_w2_wages_from_qualified_business"].uprating
        == tbs.variables["w2_wages_from_qualified_business"].uprating
    )


def test_policyengine_us_compat_qbid_supports_sstb_only_schedule_c():
    from policyengine_us import CountryTaxBenefitSystem

    tbs = CountryTaxBenefitSystem()
    params = tbs.parameters
    qbi_formula = type(tbs.variables["qualified_business_income"]).formula
    sstb_qbi_formula = type(tbs.variables["sstb_qualified_business_income"]).formula
    qbid_formula = type(tbs.variables["qbid_amount"]).formula
    deduction_formula = type(
        tbs.variables["qualified_business_income_deduction"]
    ).formula
    person_entity = tbs.variables["qualified_business_income"].entity

    class FakeTaxUnit:
        def __init__(self):
            self.members = None

        def __call__(self, variable, period):
            values = {
                "taxable_income_less_qbid": np.array([100_000.0]),
                "filing_status": np.array(["SINGLE"], dtype=object),
                "adjusted_net_capital_gain": np.array([0.0]),
            }
            return values[variable]

        def sum(self, values):
            return np.asarray(values)

    class FakePerson:
        def __init__(self):
            self.entity = type(
                "FakeEntity",
                (),
                {
                    "is_person": True,
                    "key": person_entity.key,
                    "plural": person_entity.plural,
                    "get_variable": staticmethod(tbs.get_variable),
                },
            )()
            self.tax_unit = FakeTaxUnit()
            self.values = {
                "self_employment_income": np.array([0.0]),
                "self_employment_income_would_be_qualified": np.array([True]),
                "partnership_s_corp_income": np.array([0.0]),
                "partnership_s_corp_income_would_be_qualified": np.array([True]),
                "farm_rent_income": np.array([0.0]),
                "farm_rent_income_would_be_qualified": np.array([True]),
                "farm_operations_income": np.array([0.0]),
                "farm_operations_income_would_be_qualified": np.array([True]),
                "rental_income": np.array([0.0]),
                "rental_income_would_be_qualified": np.array([True]),
                "estate_income": np.array([0.0]),
                "estate_income_would_be_qualified": np.array([True]),
                "sstb_self_employment_income": np.array([100_000.0]),
                "sstb_self_employment_income_would_be_qualified": np.array([True]),
                "self_employment_tax_ald_person": np.array([0.0]),
                "self_employed_health_insurance_ald_person": np.array([0.0]),
                "self_employed_pension_contribution_ald_person": np.array([0.0]),
                "business_is_sstb": np.array([True]),
                "w2_wages_from_qualified_business": np.array([0.0]),
                "sstb_w2_wages_from_qualified_business": np.array([0.0]),
                "unadjusted_basis_qualified_property": np.array([0.0]),
                "sstb_unadjusted_basis_qualified_property": np.array([0.0]),
                "qualified_reit_and_ptp_income": np.array([0.0]),
            }

        def __call__(self, variable, period, *args, **kwargs):
            return self.values[variable]

    person = FakePerson()
    qualified_business_income = qbi_formula(person, 2024, params)
    sstb_qualified_business_income = sstb_qbi_formula(person, 2024, params)
    person.values["qualified_business_income"] = qualified_business_income
    person.values["sstb_qualified_business_income"] = sstb_qualified_business_income
    qbid_amount = qbid_formula(person, 2024, params)
    person.values["qbid_amount"] = qbid_amount
    person.tax_unit.members = person
    qualified_business_income_deduction = deduction_formula(
        person.tax_unit, 2024, params
    )

    np.testing.assert_allclose(qualified_business_income, np.array([100_000.0]))
    np.testing.assert_allclose(sstb_qualified_business_income, np.array([100_000.0]))
    np.testing.assert_allclose(qbid_amount, np.array([20_000.0]))
    np.testing.assert_allclose(
        qualified_business_income_deduction, np.array([20_000.0])
    )


def test_policyengine_us_compat_qbid_keeps_non_sstb_qbi_when_sstb_is_negative():
    from policyengine_us import CountryTaxBenefitSystem

    tbs = CountryTaxBenefitSystem()
    params = tbs.parameters
    qbi_formula = type(tbs.variables["qualified_business_income"]).formula
    sstb_qbi_formula = type(tbs.variables["sstb_qualified_business_income"]).formula
    qbid_formula = type(tbs.variables["qbid_amount"]).formula
    deduction_formula = type(
        tbs.variables["qualified_business_income_deduction"]
    ).formula
    person_entity = tbs.variables["qualified_business_income"].entity

    class FakeTaxUnit:
        def __init__(self):
            self.members = None

        def __call__(self, variable, period):
            values = {
                "taxable_income_less_qbid": np.array([100_000.0]),
                "filing_status": np.array(["SINGLE"], dtype=object),
                "adjusted_net_capital_gain": np.array([0.0]),
            }
            return values[variable]

        def sum(self, values):
            return np.asarray(values)

    class FakePerson:
        def __init__(self):
            self.entity = type(
                "FakeEntity",
                (),
                {
                    "is_person": True,
                    "key": person_entity.key,
                    "plural": person_entity.plural,
                    "get_variable": staticmethod(tbs.get_variable),
                },
            )()
            self.tax_unit = FakeTaxUnit()
            self.values = {
                "self_employment_income": np.array([100.0]),
                "self_employment_income_would_be_qualified": np.array([True]),
                "partnership_s_corp_income": np.array([0.0]),
                "partnership_s_corp_income_would_be_qualified": np.array([True]),
                "farm_rent_income": np.array([0.0]),
                "farm_rent_income_would_be_qualified": np.array([True]),
                "farm_operations_income": np.array([0.0]),
                "farm_operations_income_would_be_qualified": np.array([True]),
                "rental_income": np.array([0.0]),
                "rental_income_would_be_qualified": np.array([True]),
                "estate_income": np.array([0.0]),
                "estate_income_would_be_qualified": np.array([True]),
                "sstb_self_employment_income": np.array([-50.0]),
                "sstb_self_employment_income_would_be_qualified": np.array([True]),
                "self_employment_tax_ald_person": np.array([0.0]),
                "self_employed_health_insurance_ald_person": np.array([0.0]),
                "self_employed_pension_contribution_ald_person": np.array([0.0]),
                "business_is_sstb": np.array([False]),
                "w2_wages_from_qualified_business": np.array([0.0]),
                "sstb_w2_wages_from_qualified_business": np.array([0.0]),
                "unadjusted_basis_qualified_property": np.array([0.0]),
                "sstb_unadjusted_basis_qualified_property": np.array([0.0]),
                "qualified_reit_and_ptp_income": np.array([0.0]),
            }

        def __call__(self, variable, period, *args, **kwargs):
            return self.values[variable]

    person = FakePerson()
    qualified_business_income = qbi_formula(person, 2024, params)
    sstb_qualified_business_income = sstb_qbi_formula(person, 2024, params)
    person.values["qualified_business_income"] = qualified_business_income
    person.values["sstb_qualified_business_income"] = sstb_qualified_business_income
    qbid_amount = qbid_formula(person, 2024, params)
    person.values["qbid_amount"] = qbid_amount
    person.tax_unit.members = person
    qualified_business_income_deduction = deduction_formula(
        person.tax_unit, 2024, params
    )

    np.testing.assert_allclose(qualified_business_income, np.array([50.0]))
    np.testing.assert_allclose(sstb_qualified_business_income, np.array([0.0]))
    np.testing.assert_allclose(qbid_amount, np.array([20.0]))
    np.testing.assert_allclose(qualified_business_income_deduction, np.array([20.0]))
