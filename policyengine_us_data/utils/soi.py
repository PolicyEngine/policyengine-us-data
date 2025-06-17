import pandas as pd
import numpy as np
from .uprating import create_policyengine_uprating_factors_table
from policyengine_us_data.storage import STORAGE_FOLDER


def pe_to_soi(pe_dataset, year):
    from policyengine_us import Microsimulation

    pe_sim = Microsimulation(dataset=pe_dataset)
    pe_sim.default_calculation_period = year
    df = pd.DataFrame()

    pe = lambda variable: np.array(
        pe_sim.calculate(variable, map_to="tax_unit")
    )

    df["adjusted_gross_income"] = pe("adjusted_gross_income")
    df["exemption"] = pe("exemptions")
    df["itemded"] = pe("itemized_taxable_income_deductions")
    df["income_tax_after_credits"] = pe("income_tax")
    df["total_income_tax"] = pe("income_tax_before_credits")
    df["taxable_income"] = pe("taxable_income")
    df["business_net_profits"] = pe("self_employment_income") * (
        pe("self_employment_income") > 0
    )
    df["business_net_losses"] = -pe("self_employment_income") * (
        pe("self_employment_income") < 0
    )
    df["capital_gains_distributions"] = pe("non_sch_d_capital_gains")
    df["capital_gains_gross"] = pe("loss_limited_net_capital_gains") * (
        pe("loss_limited_net_capital_gains") > 0
    )
    df["capital_gains_losses"] = -pe("loss_limited_net_capital_gains") * (
        pe("loss_limited_net_capital_gains") < 0
    )
    df["estate_income"] = pe("estate_income") * (pe("estate_income") > 0)
    df["estate_losses"] = -pe("estate_income") * (pe("estate_income") < 0)
    df["exempt_interest"] = pe("tax_exempt_interest_income")
    df["ira_distributions"] = pe("taxable_ira_distributions")
    df["count_of_exemptions"] = pe("exemptions_count")
    df["ordinary_dividends"] = pe("non_qualified_dividend_income") + pe(
        "qualified_dividend_income"
    )
    df["partnership_and_s_corp_income"] = pe("partnership_s_corp_income") * (
        pe("partnership_s_corp_income") > 0
    )
    df["partnership_and_s_corp_losses"] = -pe("partnership_s_corp_income") * (
        pe("partnership_s_corp_income") < 0
    )
    df["total_pension_income"] = pe("pension_income")
    df["taxable_pension_income"] = pe("taxable_pension_income")
    df["qualified_dividends"] = pe("qualified_dividend_income")
    df["rent_and_royalty_net_income"] = pe("rental_income") * (
        pe("rental_income") > 0
    )
    df["rent_and_royalty_net_losses"] = -pe("rental_income") * (
        pe("rental_income") < 0
    )
    df["total_social_security"] = pe("social_security")
    df["taxable_social_security"] = pe("taxable_social_security")
    df["income_tax_before_credits"] = pe("income_tax_before_credits")
    df["taxable_interest_income"] = pe("taxable_interest_income")
    df["unemployment_compensation"] = pe("taxable_unemployment_compensation")
    df["employment_income"] = pe("irs_employment_income")
    df["qualified_business_income_deduction"] = pe(
        "qualified_business_income_deduction"
    )
    df["charitable_contributions_deduction"] = pe("charitable_deduction")
    df["interest_paid_deductions"] = pe("interest_deduction")
    df["medical_expense_deductions_uncapped"] = pe("medical_expense_deduction")
    df["state_and_local_tax_deductions"] = pe("salt_deduction")
    df["itemized_state_income_and_sales_tax_deductions"] = pe(
        "state_and_local_sales_or_income_tax"
    )
    df["itemized_real_estate_tax_deductions"] = pe("real_estate_taxes")
    df["is_tax_filer"] = pe("tax_unit_is_filer")
    df["count"] = 1

    df["filing_status"] = pe("filing_status")
    df["weight"] = pe("tax_unit_weight")
    df["household_id"] = pe("household_id")

    return df


def puf_to_soi(puf, year):
    df = pd.DataFrame()

    df["adjusted_gross_income"] = puf.E00100
    df["total_income_tax"] = puf.E06500
    df["employment_income"] = puf.E00200
    df["capital_gains_distributions"] = puf.E01100
    df["capital_gains_gross"] = puf["E01000"] * (puf["E01000"] > 0)
    df["capital_gains_losses"] = -puf["E01000"] * (puf["E01000"] < 0)
    df["estate_income"] = puf.E26390
    df["estate_losses"] = puf.E26400
    df["exempt_interest"] = puf.E00400
    df["ira_distributions"] = puf.E01400
    df["count_of_exemptions"] = puf.XTOT
    df["ordinary_dividends"] = puf.E00600
    df["partnership_and_s_corp_income"] = puf.E26270 * (puf.E26270 > 0)
    df["partnership_and_s_corp_losses"] = -puf.E26270 * (puf.E26270 < 0)
    df["total_pension_income"] = puf.E01500
    df["taxable_pension_income"] = puf.E01700
    df["qualified_dividends"] = puf.E00650
    df["rent_and_royalty_net_income"] = puf.E25850
    df["rent_and_royalty_net_losses"] = puf.E25860
    df["total_social_security"] = puf.E02400
    df["taxable_social_security"] = puf.E02500
    df["income_tax_before_credits"] = puf.E06500
    df["taxable_interest_income"] = puf.E00300
    df["unemployment_compensation"] = puf.E02300
    df["employment_income"] = puf.E00200
    df["charitable_contributions_deduction"] = puf.E19700
    df["interest_paid_deductions"] = puf.E19200
    df["medical_expense_deductions_uncapped"] = puf.E17500
    df["itemized_state_income_and_sales_tax_deductions"] = puf.E18400
    df["itemized_real_estate_tax_deductions"] = puf.E18500
    df["state_and_local_tax_deductions"] = puf.E18400 + puf.E18500
    df["income_tax_after_credits"] = puf.E08800
    df["business_net_profits"] = puf.E00900 * (puf.E00900 > 0)
    df["business_net_losses"] = -puf.E00900 * (puf.E00900 < 0)
    df["taxable_income"] = puf.E04800
    df["is_tax_filer"] = True
    df["count"] = 1
    df["filing_status"] = puf.MARS.map(
        {
            0: "SINGLE",  # Assume the aggregate record is single
            1: "SINGLE",
            2: "JOINT",
            3: "SEPARATE",
            4: "HEAD_OF_HOUSEHOLD",
        }
    )

    df["weight"] = puf["S006"] / 100

    return df


def get_soi(year: int) -> pd.DataFrame:
    uprating = create_policyengine_uprating_factors_table()

    uprating_map = {
        "adjusted_gross_income": "adjusted_gross_income",
        "count": "population",
        "employment_income": "employment_income",
        "business_net_profits": "self_employment_income",
        "capital_gains_gross": "long_term_capital_gains",
        "ordinary_dividends": "non_qualified_dividend_income",
        "partnership_and_s_corp_income": "partnership_s_corp_income",
        "qualified_dividends": "qualified_dividend_income",
        "taxable_interest_income": "taxable_interest_income",
        "total_pension_income": "pension_income",
        "total_social_security": "social_security",
        "business_net_losses": "self_employment_income",
        "capital_gains_distributions": "long_term_capital_gains",
        "capital_gains_losses": "long_term_capital_gains",
        "estate_income": "estate_income",
        "estate_losses": "estate_income",
        "exempt_interest": "tax_exempt_interest_income",
        "ira_distributions": "taxable_ira_distributions",
        "partnership_and_s_corp_losses": "partnership_s_corp_income",
        "rent_and_royalty_net_income": "rental_income",
        "rent_and_royalty_net_losses": "rental_income",
        "taxable_pension_income": "taxable_pension_income",
        "taxable_social_security": "taxable_social_security",
        "unemployment_compensation": "unemployment_compensation",
    }
    soi = pd.read_csv(STORAGE_FOLDER / "soi.csv")
    soi = soi[soi.Year == soi.Year.max()]

    uprating_factors = {}
    for variable in uprating_map:
        pe_name = uprating_map.get(variable)
        if pe_name in uprating.index:
            uprating_factors[variable] = (
                uprating.loc[pe_name, year]
                / uprating.loc[pe_name, soi.Year.max()]
            )
        else:
            uprating_factors[variable] = (
                uprating.loc["employment_income", year]
                / uprating.loc["employment_income", soi.Year.max()]
            )

    for variable, uprating_factor in uprating_factors.items():
        soi.loc[soi.Variable == variable, "Value"] *= uprating_factor

    return soi


def compare_soi_replication_to_soi(df, soi):
    variables = []
    filing_statuses = []
    agi_lower_bounds = []
    agi_upper_bounds = []
    counts = []
    taxables = []
    full_pops = []
    values = []
    soi_values = []

    for i, row in soi.iterrows():
        if row.Variable not in df.columns:
            continue

        subset = df[df.adjusted_gross_income >= row["AGI lower bound"]][
            df.adjusted_gross_income < row["AGI upper bound"]
        ]

        variable = row["Variable"]

        fs = row["Filing status"]
        if fs == "Single":
            subset = subset[subset.filing_status == "SINGLE"]
        elif fs == "Head of Household":
            subset = subset[subset.filing_status == "HEAD_OF_HOUSEHOLD"]
        elif fs == "Married Filing Jointly/Surviving Spouse":
            subset = subset[subset.filing_status.isin(["JOINT", "WIDOW"])]
        elif fs == "Married Filing Separately":
            subset = subset[subset.filing_status == "SEPARATE"]

        if row["Taxable only"]:
            subset = subset[subset.total_income_tax > 0]
        else:
            subset = subset[subset.is_tax_filer.values > 0]

        if row["Count"]:
            value = subset[subset[variable] > 0].weight.sum()
        else:
            value = (subset[variable] * subset.weight).sum()

        variables.append(row["Variable"])
        filing_statuses.append(row["Filing status"])
        agi_lower_bounds.append(row["AGI lower bound"])
        agi_upper_bounds.append(row["AGI upper bound"])
        counts.append(row["Count"] or (row["Variable"] == "count"))
        taxables.append(row["Taxable only"])
        full_pops.append(row["Full population"])
        values.append(value)
        soi_values.append(row["Value"])

    soi_replication = pd.DataFrame(
        {
            "Variable": variables,
            "Filing status": filing_statuses,
            "AGI lower bound": agi_lower_bounds,
            "AGI upper bound": agi_upper_bounds,
            "Count": counts,
            "Taxable only": taxables,
            "Full population": full_pops,
            "Value": values,
            "SOI Value": soi_values,
        }
    )

    soi_replication["Error"] = (
        soi_replication["Value"] - soi_replication["SOI Value"]
    )
    soi_replication["Absolute error"] = soi_replication["Error"].abs()
    soi_replication["Relative error"] = (
        (soi_replication["Error"] / soi_replication["SOI Value"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    soi_replication["Absolute relative error"] = soi_replication[
        "Relative error"
    ].abs()

    return soi_replication
