"""Role classifications for PUF-sourced PolicyEngine variables."""

REPORTED_CALCULATED_TAX_OUTPUT_ROLE = "reported_calculated_tax_output"

PUF_REPORTED_CALCULATED_TAX_OUTPUT_VARIABLES = frozenset(
    (
        "taxable_unemployment_compensation",
        "foreign_tax_credit",
        "american_opportunity_credit",
        "general_business_credit",
        "energy_efficient_home_improvement_credit",
        "amt_foreign_tax_credit",
        "excess_withheld_payroll_tax",
        "savers_credit",
        "early_withdrawal_penalty",
        "prior_year_minimum_tax_credit",
        "other_credits",
        "unreported_payroll_tax",
        "recapture_of_investment_credit",
    )
)

PUF_SOURCE_VARIABLE_ROLES = {
    variable: REPORTED_CALCULATED_TAX_OUTPUT_ROLE
    for variable in PUF_REPORTED_CALCULATED_TAX_OUTPUT_VARIABLES
}
