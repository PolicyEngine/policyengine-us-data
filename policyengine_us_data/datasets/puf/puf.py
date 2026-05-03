import h5py
import yaml
from importlib.resources import files

import numpy as np
import pandas as pd
from microdf import MicroDataFrame

from policyengine_core.data import Dataset
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.puf.uprate_puf import uprate_puf
from policyengine_us_data.datasets.puf.irs_puf import IRS_PUF_2015
from policyengine_us_data.datasets.puf.disaggregate_puf import (
    disaggregate_aggregate_records,
)
from policyengine_us_data.utils.mortgage_interest import (
    STRUCTURAL_MORTGAGE_VARIABLES,
    convert_mortgage_interest_to_structural_inputs,
)
from policyengine_us_data.utils.policyengine import (
    has_policyengine_us_variables,
)
from policyengine_us_data.utils.uprating import (
    create_policyengine_uprating_factors_table,
)

rng = np.random.default_rng(seed=64)

# Get Qualified Business Income simulation parameters ---
yamlfilename = (
    files("policyengine_us_data") / "datasets" / "puf" / "qbi_assumptions.yaml"
)
with open(yamlfilename, "r", encoding="utf-8") as yamlfile:
    QBI_PARAMS = yaml.safe_load(yamlfile)
assert isinstance(QBI_PARAMS, dict)

QBI_SOURCE_NAMES = tuple(QBI_PARAMS["qbi_qualification_probabilities"])
QBI_QUALIFICATION_FLAG_BY_SOURCE = {
    source: f"{source}_would_be_qualified" for source in QBI_SOURCE_NAMES
}
SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG = (
    "sstb_self_employment_income_would_be_qualified"
)
QBI_QUALIFICATION_SEED = 41
QBI_W2_UBIA_SEED = 42
QBI_INVESTMENT_SEED = 43
QBI_SSTB_SEED = 64
QBI_SIMULATION_REQUIRED_VARIABLES = frozenset(
    (
        *QBI_QUALIFICATION_FLAG_BY_SOURCE.values(),
        SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG,
        "business_is_sstb",
        "sstb_self_employment_income",
        "w2_wages_from_qualified_business",
        "unadjusted_basis_qualified_property",
        "sstb_w2_wages_from_qualified_business",
        "sstb_unadjusted_basis_qualified_property",
        "qualified_reit_and_ptp_income",
        "qualified_bdc_income",
    )
)


# Helper functions ---
def conditionally_sample_lognormal(flag, target_mean, log_sigma, rng):
    """Generate a lognormal conditional on a binary flag."""
    flag = np.asarray(flag, dtype=bool)
    target_mean = np.asarray(target_mean, dtype=float)
    eligible = flag & (target_mean > 0)
    safe_target_mean = np.where(target_mean > 0, target_mean, 1.0)
    mu = np.log(safe_target_mean) - (log_sigma**2 / 2)
    return np.where(
        eligible,
        rng.lognormal(
            mean=mu,
            sigma=log_sigma,
        ),
        0.0,
    )


def draw_qbi_qualification_flags(n, *, seed=None):
    """Draw source-level QBI qualification inputs."""
    flag_rng = np.random.default_rng(seed)
    return {
        source: flag_rng.random(n) < prob
        for source, prob in QBI_PARAMS["qbi_qualification_probabilities"].items()
    }


def add_qbi_qualification_flags_to_puf(puf, *, seed=None):
    """Draw and persist source-level QBI qualification inputs."""
    flags = draw_qbi_qualification_flags(len(puf), seed=seed)
    for source, qualified in flags.items():
        puf[QBI_QUALIFICATION_FLAG_BY_SOURCE[source]] = qualified
    return puf


def qualified_qbi_components(puf):
    """Return source amounts after applying persisted QBI qualification flags."""
    components = {}
    for source, prob in QBI_PARAMS["qbi_qualification_probabilities"].items():
        source_values = puf[source].fillna(0).to_numpy(dtype=float)
        flag_name = QBI_QUALIFICATION_FLAG_BY_SOURCE[source]
        if flag_name in puf:
            qualified = puf[flag_name].fillna(False).astype(bool).to_numpy()
            components[source] = source_values * qualified
        else:
            components[source] = source_values * prob
    return pd.DataFrame(components, index=puf.index)


def positive_qbi_source_amounts(qbi_components, params_by_source=None):
    """Return positive QBI components limited to modeled sources."""
    sources = (
        [source for source in params_by_source if source in qbi_components]
        if params_by_source is not None
        else list(qbi_components.columns)
    )
    positive_components = qbi_components[sources].fillna(0).clip(lower=0)
    positive_total = positive_components.sum(axis=1).to_numpy(dtype=float)
    return positive_components, positive_total


def source_weighted_parameter(qbi_components, params_by_source):
    """Weight source-level scalar parameters by positive qualified QBI."""
    positive_components, positive_total = positive_qbi_source_amounts(
        qbi_components, params_by_source
    )
    weighted_value = np.zeros(len(qbi_components), dtype=float)
    for source, value in params_by_source.items():
        if source in positive_components:
            weighted_value += positive_components[source].to_numpy(dtype=float) * float(
                value
            )
    return np.divide(
        weighted_value,
        positive_total,
        out=np.zeros_like(positive_total, dtype=float),
        where=positive_total > 0,
    )


def draw_source_weighted_beta(qbi_components, params_by_source, rng):
    """Draw source-level beta values and QBI-weight them within each record."""
    positive_components, positive_total = positive_qbi_source_amounts(
        qbi_components, params_by_source
    )
    weighted_draw = np.zeros(len(qbi_components), dtype=float)
    for source, params in params_by_source.items():
        if source not in positive_components:
            continue
        draw = rng.beta(
            params["beta_a"], params["beta_b"], len(qbi_components)
        ) * params.get("scale", 1.0) + params.get("shift", 0.0)
        weighted_draw += positive_components[source].to_numpy(dtype=float) * draw
    return np.divide(
        weighted_draw,
        positive_total,
        out=np.zeros_like(positive_total, dtype=float),
        where=positive_total > 0,
    )


def capital_intensity_probability(qbi_components):
    """Estimate UBIA eligibility probability from positive qualified QBI sources."""
    source_probs = QBI_PARAMS["ubia_simulation"]["capital_intensity_probabilities"]
    return source_weighted_parameter(qbi_components, source_probs).clip(0, 1)


def logistic(values):
    """Numerically stable logistic transform."""
    return 1.0 / (1.0 + np.exp(-np.clip(values, -700, 700)))


def calibrate_logit_intercept(revenues, slope, target_share):
    """Solve the employee-presence logit intercept for positive-receipt records."""
    revenues = np.asarray(revenues, dtype=float)
    positive = revenues > 0
    if not np.any(positive):
        return 0.0

    target_share = np.clip(float(target_share), 1e-9, 1 - 1e-9)
    slope_term = float(slope) * revenues[positive]
    target_logit = np.log(target_share / (1 - target_share))
    lower = target_logit - np.max(slope_term) - 80.0
    upper = target_logit - np.min(slope_term) + 80.0
    for _ in range(100):
        midpoint = (lower + upper) / 2
        mean_probability = logistic(midpoint + slope_term).mean()
        if mean_probability < target_share:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def puf_column_values(puf, column):
    """Return a PUF column as float values, or zeros when absent."""
    if column not in puf:
        return np.zeros(len(puf), dtype=float)
    return puf[column].fillna(0).to_numpy(dtype=float)


def non_qualified_dividend_income_from_puf(puf):
    """Recover ordinary dividends that are not qualified dividends."""
    if "non_qualified_dividend_income" in puf:
        return puf_column_values(puf, "non_qualified_dividend_income")
    if "E00600" in puf and "E00650" in puf:
        return puf_column_values(puf, "E00600") - puf_column_values(puf, "E00650")
    if "ordinary_dividend_income" in puf:
        return puf_column_values(puf, "ordinary_dividend_income")
    return np.zeros(len(puf), dtype=float)


def sample_exposure_scaled_beta(base, params, rng):
    """Sample a positive share of an observed exposure base."""
    base = np.maximum(np.asarray(base, dtype=float), 0)
    receives = (base > 0) & (
        rng.random(len(base)) < float(params["probability_of_receiving"])
    )
    share = rng.beta(params["beta_a"], params["beta_b"], len(base)) * params.get(
        "scale", 1.0
    ) + params.get("shift", 0.0)
    share = np.clip(share, 0, 1)
    return np.where(receives, base * share, 0.0)


def simulate_investment_qbi_income_from_puf(puf, *, rng):
    """Simulate qualified REIT/PTP and BDC income from observed exposures."""
    exposure_bases = {
        "non_qualified_dividend_income": non_qualified_dividend_income_from_puf(puf),
        "partnership_s_corp_income": puf_column_values(
            puf, "partnership_s_corp_income"
        ),
    }

    qualified_reit_and_ptp_income = np.zeros(len(puf), dtype=float)
    for exposure_source, params in QBI_PARAMS["reit_ptp_income_distribution"].items():
        base = exposure_bases.get(exposure_source)
        if base is None:
            continue
        qualified_reit_and_ptp_income += sample_exposure_scaled_beta(base, params, rng)

    qualified_bdc_income = np.zeros(len(puf), dtype=float)
    for exposure_source, params in QBI_PARAMS["bdc_income_distribution"].items():
        base = exposure_bases.get(exposure_source)
        if base is None:
            continue
        qualified_bdc_income += sample_exposure_scaled_beta(base, params, rng)

    return {
        "qualified_reit_and_ptp_income": qualified_reit_and_ptp_income,
        "qualified_bdc_income": qualified_bdc_income,
    }


def simulate_business_is_sstb(puf, *, rng, probability_map=None):
    """Draw SSTB status only from positive qualified mapped SSTB sources."""
    sstb_probs = probability_map or QBI_PARAMS["sstb_prob_map_by_name"]
    available_sources = [source for source in sstb_probs if source in puf]
    if not available_sources:
        return np.zeros(len(puf), dtype=bool)
    sstb_sources = puf[available_sources].fillna(0).clip(lower=0)
    for source in available_sources:
        flag_name = QBI_QUALIFICATION_FLAG_BY_SOURCE.get(source)
        if flag_name is None or flag_name not in puf:
            continue
        qualified = puf[flag_name].fillna(False).astype(bool).to_numpy()
        sstb_sources[source] = sstb_sources[source].to_numpy(dtype=float) * qualified
    has_sstb_source = sstb_sources.sum(axis=1).to_numpy() > 0
    largest_sstb_source = sstb_sources.idxmax(axis=1)
    pr_sstb = largest_sstb_source.map(sstb_probs).fillna(0.0).to_numpy()
    pr_sstb = np.where(has_sstb_source, pr_sstb, 0.0)
    return rng.binomial(n=1, p=pr_sstb).astype(bool)


def simulate_w2_and_ubia_from_puf(puf, *, seed=None, diagnostics=True):
    """
    Simulate two Section 199A guard-rail quantities for every record
      - W-2 wages paid by the business
      - Unadjusted basis immediately after acquisition (UBIA) of property

    Parameters
    ----------
    puf : pandas.DataFrame
        Must contain the income columns created in your preprocessing block.
    seed : int, optional
        For reproducible random draws.
    diagnostics : bool, default True
        Print high-level checks after the simulation runs.

    Returns
    -------
    w2_wages : 1-D NumPy array
    ubia     : 1-D NumPy array
    """
    rng = np.random.default_rng(seed)

    # Extract Qualified Business Income simulation parameters
    margin_params = QBI_PARAMS["profit_margin_distribution"]
    logit_params = QBI_PARAMS["has_employees_logit"]
    labor_params = QBI_PARAMS["labor_ratio_distribution"]

    ubia_params = QBI_PARAMS["ubia_simulation"]
    ubia_sigma = ubia_params["sigma"]

    # Estimate qualified business income
    qbi_components = qualified_qbi_components(puf)
    qbi = qbi_components.sum(axis=1).to_numpy()

    # Simulate gross receipts by drawing source-weighted profit margins.
    margins = draw_source_weighted_beta(qbi_components, margin_params, rng)
    revenues = np.divide(
        np.maximum(qbi, 0),
        margins,
        out=np.zeros_like(qbi, dtype=float),
        where=margins > 0,
    )

    intercept = calibrate_logit_intercept(
        revenues,
        logit_params["slope_per_dollar"],
        logit_params["target_share_among_positive_receipts"],
    )
    logit = intercept + logit_params["slope_per_dollar"] * revenues

    # Set p = 0 when simulated receipts == 0 (no revenue means no payroll)
    pr_has_employees = np.where(revenues == 0.0, 0.0, logistic(logit))
    has_employees = rng.binomial(1, pr_has_employees)

    labor_ratios = draw_source_weighted_beta(qbi_components, labor_params, rng)
    w2_wages = revenues * labor_ratios * has_employees

    # UBIA simulation: lognormal, but only for capital-heavy records.
    pr_capital_intensive = capital_intensity_probability(qbi_components)
    is_capital_intensive = rng.binomial(1, pr_capital_intensive).astype(bool)
    ubia_multiple_of_qbi = source_weighted_parameter(
        qbi_components,
        ubia_params["multiple_of_qbi"],
    )

    ubia = conditionally_sample_lognormal(
        is_capital_intensive,
        ubia_multiple_of_qbi * np.maximum(qbi, 0),
        ubia_sigma,
        rng,
    )

    if diagnostics:
        share_qbi_pos = np.mean(qbi > 0)
        qbi_positive = qbi > 0
        qbi_positive_count = np.sum(qbi_positive)
        share_wages = (
            np.sum((w2_wages > 0) & qbi_positive) / qbi_positive_count
            if qbi_positive_count
            else 0.0
        )
        print(f"Share with QBI > 0: {share_qbi_pos:6.2%}")
        print(f"Among those, share with W-2 wages: {share_wages:6.2%}")
        if np.any(w2_wages > 0):
            print(f"Mean W-2 (if >0): ${np.mean(w2_wages[w2_wages > 0]):,.0f}")
        if np.any(ubia > 0):
            print(f"Median UBIA (if >0): ${np.median(ubia[ubia > 0]):,.0f}")

    return w2_wages, ubia


def impute_pension_contributions_to_puf(puf_df):
    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets.cps import CPS_2024

    # CPS_2024 may not exist yet during parallel CI builds.
    # Fall back to CPS_2021 release artifact if needed.
    try:
        cps = Microsimulation(dataset=CPS_2024)
    except Exception:
        from policyengine_us_data.datasets.cps import CPS_2021

        cps = Microsimulation(dataset=CPS_2021)
    cps.subsample(10_000)

    predictors = [
        "employment_income",
        "age",
        "is_male",
    ]

    cps_df = cps.calculate_dataframe(
        predictors + ["household_weight", "pre_tax_contributions"]
    )

    from microimpute.models.qrf import QRF

    qrf = QRF()

    # Combine predictors and target into single DataFrame for models.QRF
    cps_train = cps_df[predictors + ["pre_tax_contributions"]]

    fitted_model = qrf.fit(
        X_train=cps_train,
        predictors=predictors,
        imputed_variables=["pre_tax_contributions"],
    )

    # Predict using the fitted model
    predictions = fitted_model.predict(X_test=puf_df[predictors])

    return predictions["pre_tax_contributions"]


def impute_missing_demographics(
    puf: pd.DataFrame, demographics: pd.DataFrame
) -> pd.DataFrame:
    from microimpute.models.qrf import QRF

    puf_with_demographics = (
        puf[puf.RECID.isin(demographics.RECID)]
        .merge(demographics, on="RECID")
        .fillna(0)
    )

    puf_with_demographics = puf_with_demographics.sample(n=10_000, random_state=0)

    DEMOGRAPHIC_VARIABLES = [
        "AGEDP1",
        "AGEDP2",
        "AGEDP3",
        "AGERANGE",
        "EARNSPLIT",
        "GENDER",
    ]
    NON_DEMOGRAPHIC_VARIABLES = [
        "E00200",
        "MARS",
        "DSI",
        "EIC",
        "XTOT",
    ]

    qrf = QRF()

    # Prepare training data with predictors and variables to impute
    train_data = puf_with_demographics[
        NON_DEMOGRAPHIC_VARIABLES + DEMOGRAPHIC_VARIABLES
    ]

    fitted_model = qrf.fit(
        X_train=train_data,
        predictors=NON_DEMOGRAPHIC_VARIABLES,
        imputed_variables=DEMOGRAPHIC_VARIABLES,
    )

    puf_without_demographics = puf[
        ~puf.RECID.isin(puf_with_demographics.RECID)
    ].reset_index()

    # Predict demographics
    predicted_demographics = fitted_model.predict(
        X_test=puf_without_demographics[NON_DEMOGRAPHIC_VARIABLES]
    )

    puf_with_imputed_demographics = pd.concat(
        [puf_without_demographics, predicted_demographics], axis=1
    )

    weighted_puf_with_demographics = MicroDataFrame(
        puf_with_demographics, weights="S006"
    )
    weighted_puf_with_imputed_demographics = MicroDataFrame(
        puf_with_imputed_demographics, weights="S006"
    )

    puf_combined = pd.concat(
        [
            weighted_puf_with_demographics,
            weighted_puf_with_imputed_demographics,
        ]
    )

    return puf_combined


def decode_age_filer(age_range: int) -> int:
    if age_range == 0:
        return 40
    AGERANGE_FILER_DECODE = {
        1: 18,
        2: 26,
        3: 35,
        4: 45,
        5: 55,
        6: 65,
        7: 80,
    }
    lower = AGERANGE_FILER_DECODE[age_range]
    upper = AGERANGE_FILER_DECODE[age_range + 1]
    return rng.integers(low=lower, high=upper, endpoint=False)


def decode_age_dependent(age_range: int) -> int:
    if age_range == 0:
        return 0
    AGERANGE_DEPENDENT_DECODE = {
        0: 0,
        1: 0,
        2: 5,
        3: 13,
        4: 17,
        5: 19,
        6: 25,
        7: 30,
    }
    lower = AGERANGE_DEPENDENT_DECODE[age_range]
    upper = AGERANGE_DEPENDENT_DECODE[age_range + 1]
    return rng.integers(low=lower, high=upper, endpoint=False)


def preprocess_puf(puf: pd.DataFrame) -> pd.DataFrame:
    # Add variable renames
    puf.S006 = puf.S006 / 100
    # puf["adjusted_gross_income"] = puf.E00100
    puf["alimony_expense"] = puf.E03500
    puf["alimony_income"] = puf.E00800
    puf["casualty_loss"] = puf.E20500
    puf["cdcc_relevant_expenses"] = puf.E32800
    puf["charitable_cash_donations"] = puf.E19800
    puf["charitable_non_cash_donations"] = puf.E20100
    puf["domestic_production_ald"] = puf.E03240
    puf["early_withdrawal_penalty"] = puf.E03400
    puf["educator_expense"] = puf.E03220
    puf["employment_income"] = puf.E00200
    puf["estate_income"] = puf.E26390 - puf.E26400
    # Schedule J, separate from QBI
    puf["farm_income"] = puf.T27800
    puf["health_savings_account_ald"] = puf.E03290
    puf["interest_deduction"] = puf.E19200
    puf["long_term_capital_gains"] = puf.P23250
    puf["long_term_capital_gains_on_collectibles"] = puf.E24518
    # Split medical expenses using CPS fractions
    for (
        medical_category,
        fraction,
    ) in MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS.items():
        puf[medical_category] = puf.E17500 * fraction
    # Use unreimbursed business employee expenses as a proxy for all miscellaneous expenses
    # that can be deducted under the miscellaneous deduction.
    puf["unreimbursed_business_employee_expenses"] = puf.E20400
    puf["non_qualified_dividend_income"] = puf.E00600 - puf.E00650
    puf["qualified_dividend_income"] = puf.E00650
    puf["qualified_tuition_expenses"] = puf.E03230
    puf["real_estate_taxes"] = puf.E18500
    # Schedule E rent and royalty
    puf["rental_income"] = puf.E25850 - puf.E25860
    # Schedule E active S-Corp income
    s_corp_income = puf.E26190 - puf.E26180
    # Schedule E active partnership income
    partnership_income = puf.E25980 - puf.E25960
    puf["partnership_s_corp_income"] = s_corp_income + partnership_income
    # Schedule F active farming operations
    puf["farm_operations_income"] = puf.E02100
    # Schedule E farm rental income
    puf["farm_rent_income"] = puf.E27200
    # Schedule C Sole Proprietorship
    puf["self_employment_income"] = puf.E00900
    puf["self_employed_health_insurance_ald"] = puf.E03270
    puf["self_employed_pension_contribution_ald"] = puf.E03300
    puf["short_term_capital_gains"] = puf.P22250
    puf["social_security"] = puf.E02400
    puf["state_and_local_sales_or_income_tax"] = puf.E18400
    puf["student_loan_interest"] = puf.E03210
    puf["taxable_interest_income"] = puf.E00300
    puf["taxable_pension_income"] = puf.E01700
    puf["taxable_unemployment_compensation"] = puf.E02300
    puf["taxable_ira_distributions"] = puf.E01400
    puf["tax_exempt_interest_income"] = puf.E00400
    puf["tax_exempt_pension_income"] = puf.E01500 - puf.E01700
    puf["traditional_ira_contributions"] = puf.E03150
    puf["unrecaptured_section_1250_gain"] = puf.E24515

    puf["foreign_tax_credit"] = puf.E07300
    puf["amt_foreign_tax_credit"] = puf.E62900
    puf["miscellaneous_income"] = puf.E01200
    puf["salt_refund_income"] = puf.E00700
    puf["investment_income_elected_form_4952"] = puf.E58990
    puf["general_business_credit"] = puf.E07400
    puf["prior_year_minimum_tax_credit"] = puf.E07600
    puf["excess_withheld_payroll_tax"] = puf.E11200
    puf["non_sch_d_capital_gains"] = puf.E01100
    puf["american_opportunity_credit"] = puf.E87521
    puf["energy_efficient_home_improvement_credit"] = puf.E07260
    puf["early_withdrawal_penalty"] = puf.E09900
    # puf["qualified_tuition_expenses"] = puf.E87530 # PE uses the same variable for qualified tuition (general) and qualified tuition (Lifetime Learning Credit). Revisit here.
    puf["other_credits"] = puf.P08000
    puf["savers_credit"] = puf.E07240
    puf["recapture_of_investment_credit"] = puf.E09700
    puf["unreported_payroll_tax"] = puf.E09800
    # Ignore f2441 (AMT form attached)
    # Ignore cmbtp (estimate of AMT income not in AGI)

    # Partnership self-employment income from Schedule K-1 Box 14
    # This is the portion of partnership income subject to SE tax (general partners)
    # Derived from total SE income minus Schedule C and Schedule F income
    # Based on Yale Budget Lab's Tax-Data process_puf.R approach:
    #   E30400 = taxpayer's TAXABLE SE income (already * 0.9235)
    #   E30500 = spouse's TAXABLE SE income (already * 0.9235)
    #   E00900 = Schedule C net profit/loss (gross)
    #   E02100 = Schedule F farm income (gross)
    # Since E30400/E30500 are post-deduction (taxable), we gross them up
    # by dividing by 0.9235 before subtracting Sch C/F.
    # PolicyEngine applies the 0.9235 factor itself in taxable_self_employment_income.
    SE_DEDUCTION_FACTOR = 0.9235  # 1 - 0.5 * 0.153 (half of SE tax rate)
    taxable_se = puf["E30400"].fillna(0) + puf["E30500"].fillna(0)
    gross_se = taxable_se / SE_DEDUCTION_FACTOR
    schedule_c_f_income = puf["E00900"].fillna(0) + puf["E02100"].fillna(0)
    # Only compute when there's partnership activity (net partnership income != 0)
    has_partnership = (
        puf["E25940"].fillna(0)
        + puf["E25980"].fillna(0)
        - puf["E25920"].fillna(0)
        - puf["E25960"].fillna(0)
    ) != 0
    partnership_se = np.where(has_partnership, gross_se - schedule_c_f_income, 0)
    puf["partnership_se_income"] = partnership_se

    # --- Qualified Business Income Deduction (QBID) simulation ---
    puf = add_qbi_qualification_flags_to_puf(puf, seed=QBI_QUALIFICATION_SEED)
    w2, ubia = simulate_w2_and_ubia_from_puf(puf, seed=QBI_W2_UBIA_SEED)
    puf["w2_wages_from_qualified_business"] = w2
    puf["unadjusted_basis_qualified_property"] = ubia

    puf["business_is_sstb"] = simulate_business_is_sstb(
        puf,
        rng=np.random.default_rng(QBI_SSTB_SEED),
        probability_map=QBI_PARAMS["sstb_prob_map_by_source_name"],
    )
    is_sstb = puf["business_is_sstb"].astype(bool)

    # The current PUF pipeline only imputes an all-or-nothing SSTB flag.
    # Use that to split Schedule C self-employment and allocable W-2/UBIA
    # inputs for policyengine-us without pretending to observe mixed cases.
    legacy_self_employment_income = puf["self_employment_income"].fillna(0)
    self_employment_would_be_qualified = puf[
        "self_employment_income_would_be_qualified"
    ].astype(bool)
    puf["sstb_self_employment_income"] = np.where(
        is_sstb, legacy_self_employment_income, 0.0
    )
    puf["self_employment_income"] = np.where(
        is_sstb, 0.0, legacy_self_employment_income
    )
    puf[SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG] = np.where(
        is_sstb, self_employment_would_be_qualified, False
    )
    puf["self_employment_income_would_be_qualified"] = np.where(
        is_sstb, False, self_employment_would_be_qualified
    )
    puf["sstb_w2_wages_from_qualified_business"] = np.where(is_sstb, w2, 0.0)
    puf["sstb_unadjusted_basis_qualified_property"] = np.where(is_sstb, ubia, 0.0)

    investment_qbi = simulate_investment_qbi_income_from_puf(
        puf, rng=np.random.default_rng(QBI_INVESTMENT_SEED)
    )
    for variable, values in investment_qbi.items():
        puf[variable] = values
    # -------- End of Qualified Business Income Deduction (QBID) -------
    puf["filing_status"] = puf.MARS.map(
        {
            1: "SINGLE",
            2: "JOINT",
            3: "SEPARATE",
            4: "HEAD_OF_HOUSEHOLD",
        }
    )
    puf["household_id"] = puf.RECID
    puf["household_weight"] = puf.S006
    puf["exemptions_count"] = puf.XTOT

    return puf


FINANCIAL_SUBSET = [
    # "adjusted_gross_income",
    "alimony_expense",
    "alimony_income",
    "casualty_loss",
    "cdcc_relevant_expenses",
    "charitable_cash_donations",
    "charitable_non_cash_donations",
    "domestic_production_ald",
    "early_withdrawal_penalty",
    "educator_expense",
    "employment_income",
    "estate_income",
    "farm_operations_income",
    "farm_income",
    "farm_rent_income",
    "health_savings_account_ald",
    "interest_deduction",
    "long_term_capital_gains",
    "long_term_capital_gains_on_collectibles",
    "unreimbursed_business_employee_expenses",
    "non_qualified_dividend_income",
    "non_sch_d_capital_gains",
    "qualified_dividend_income",
    "qualified_tuition_expenses",
    "real_estate_taxes",
    "rental_income",
    "self_employment_income",
    "self_employed_health_insurance_ald",
    "self_employed_pension_contribution_ald",
    "short_term_capital_gains",
    "social_security",
    "state_and_local_sales_or_income_tax",
    "student_loan_interest",
    "taxable_interest_income",
    "taxable_pension_income",
    "taxable_unemployment_compensation",
    "taxable_ira_distributions",
    "tax_exempt_interest_income",
    "tax_exempt_pension_income",
    "traditional_ira_contributions",
    "unrecaptured_section_1250_gain",
    "foreign_tax_credit",
    "amt_foreign_tax_credit",
    "miscellaneous_income",
    "salt_refund_income",
    "investment_income_elected_form_4952",
    "general_business_credit",
    "prior_year_minimum_tax_credit",
    "excess_withheld_payroll_tax",
    "american_opportunity_credit",
    "energy_efficient_home_improvement_credit",
    "other_credits",
    "savers_credit",
    "recapture_of_investment_credit",
    "unreported_payroll_tax",
    "pre_tax_contributions",
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
    "self_employment_income_would_be_qualified",
    "sstb_self_employment_income_would_be_qualified",
    "w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "business_is_sstb",
    "sstb_self_employment_income",
    "sstb_w2_wages_from_qualified_business",
    "sstb_unadjusted_basis_qualified_property",
    "deductible_mortgage_interest",
    "partnership_s_corp_income",
    "partnership_se_income",
    "qualified_reit_and_ptp_income",
    "qualified_bdc_income",
]


class PUF(Dataset):
    time_period = None
    data_format = Dataset.ARRAYS

    @staticmethod
    def _replace_array(file_handle, key: str, values: np.ndarray) -> None:
        if key in file_handle:
            del file_handle[key]
        file_handle.create_dataset(key, data=values)

    @staticmethod
    def _values_from_file_or_overrides(
        file_handle, key: str, overrides: dict[str, np.ndarray], length: int
    ) -> np.ndarray:
        if key in overrides:
            return np.asarray(overrides[key])
        if key in file_handle:
            return np.asarray(file_handle[key])
        return np.zeros(length)

    def _sstb_split_overrides(self) -> dict[str, np.ndarray]:
        if not self.file_path.exists():
            return {}

        with h5py.File(self.file_path, "r") as file_handle:
            if "business_is_sstb" not in file_handle:
                return {}
            keys = set(file_handle.keys())
            is_sstb = np.asarray(file_handle["business_is_sstb"]).astype(bool)
            overrides = {}
            if "self_employment_income" in keys:
                self_employment_income = np.asarray(
                    file_handle["self_employment_income"]
                )
                existing_sstb_self_employment_income = (
                    np.asarray(file_handle["sstb_self_employment_income"])
                    if "sstb_self_employment_income" in keys
                    else np.zeros_like(self_employment_income)
                )
                corrected_sstb_self_employment_income = np.where(
                    is_sstb,
                    np.where(
                        existing_sstb_self_employment_income != 0,
                        existing_sstb_self_employment_income,
                        self_employment_income,
                    ),
                    0.0,
                )
                corrected_self_employment_income = np.where(
                    is_sstb, 0.0, self_employment_income
                )
                if (
                    "sstb_self_employment_income" not in keys
                    or not np.array_equal(
                        existing_sstb_self_employment_income,
                        corrected_sstb_self_employment_income,
                    )
                    or not np.array_equal(
                        self_employment_income,
                        corrected_self_employment_income,
                    )
                ):
                    overrides["sstb_self_employment_income"] = (
                        corrected_sstb_self_employment_income
                    )
                    overrides["self_employment_income"] = (
                        corrected_self_employment_income
                    )

            for source_key, target_key in (
                (
                    "w2_wages_from_qualified_business",
                    "sstb_w2_wages_from_qualified_business",
                ),
                (
                    "unadjusted_basis_qualified_property",
                    "sstb_unadjusted_basis_qualified_property",
                ),
            ):
                if source_key not in keys:
                    continue
                corrected_target = np.where(
                    is_sstb, np.asarray(file_handle[source_key]), 0.0
                )
                if target_key not in keys or not np.array_equal(
                    np.asarray(file_handle[target_key]),
                    corrected_target,
                ):
                    overrides[target_key] = corrected_target

        return overrides

    def _qbi_simulation_overrides(
        self, existing_overrides: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if not self.file_path.exists():
            return {}

        with h5py.File(self.file_path, "r") as file_handle:
            keys = set(file_handle.keys())
            if QBI_SIMULATION_REQUIRED_VARIABLES.issubset(keys):
                return {}

            length = None
            for key in (
                "household_id",
                "self_employment_income",
                "partnership_s_corp_income",
            ):
                if key in file_handle:
                    length = len(file_handle[key])
                    break
            if length is None:
                return {}

            raw_qualification_flags = draw_qbi_qualification_flags(
                length, seed=QBI_QUALIFICATION_SEED
            )
            has_existing_sstb = (
                "business_is_sstb" in file_handle
                or "business_is_sstb" in existing_overrides
            )
            is_sstb_existing = self._values_from_file_or_overrides(
                file_handle, "business_is_sstb", existing_overrides, length
            ).astype(bool)
            self_employment_would_be_qualified = raw_qualification_flags[
                "self_employment_income"
            ]
            flag_overrides = {
                "self_employment_income_would_be_qualified": np.where(
                    is_sstb_existing, False, self_employment_would_be_qualified
                ),
                SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG: np.where(
                    is_sstb_existing, self_employment_would_be_qualified, False
                ),
            }
            for source, qualified in raw_qualification_flags.items():
                flag = QBI_QUALIFICATION_FLAG_BY_SOURCE[source]
                if source != "self_employment_income":
                    flag_overrides[flag] = qualified

            source_arrays = {}
            for source in QBI_SOURCE_NAMES:
                source_arrays[source] = self._values_from_file_or_overrides(
                    file_handle, source, existing_overrides, length
                ).astype(float)

            source_arrays["self_employment_income"] = (
                self._values_from_file_or_overrides(
                    file_handle,
                    "self_employment_income",
                    existing_overrides,
                    length,
                ).astype(float)
                + self._values_from_file_or_overrides(
                    file_handle,
                    "sstb_self_employment_income",
                    existing_overrides,
                    length,
                ).astype(float)
            )

            qbi_frame = pd.DataFrame(source_arrays)
            for source, qualified in raw_qualification_flags.items():
                qbi_frame[QBI_QUALIFICATION_FLAG_BY_SOURCE[source]] = qualified
            for source in (
                "qualified_dividend_income",
                "non_qualified_dividend_income",
                "ordinary_dividend_income",
                "E00600",
                "E00650",
            ):
                if source in file_handle or source in existing_overrides:
                    qbi_frame[source] = self._values_from_file_or_overrides(
                        file_handle, source, existing_overrides, length
                    ).astype(float)

            simulated_w2, simulated_ubia = simulate_w2_and_ubia_from_puf(
                qbi_frame, seed=QBI_W2_UBIA_SEED, diagnostics=False
            )
            w2 = (
                self._values_from_file_or_overrides(
                    file_handle,
                    "w2_wages_from_qualified_business",
                    existing_overrides,
                    length,
                ).astype(float)
                if (
                    "w2_wages_from_qualified_business" in file_handle
                    or "w2_wages_from_qualified_business" in existing_overrides
                )
                else simulated_w2
            )
            ubia = (
                self._values_from_file_or_overrides(
                    file_handle,
                    "unadjusted_basis_qualified_property",
                    existing_overrides,
                    length,
                ).astype(float)
                if (
                    "unadjusted_basis_qualified_property" in file_handle
                    or "unadjusted_basis_qualified_property" in existing_overrides
                )
                else simulated_ubia
            )
            is_sstb = (
                is_sstb_existing
                if has_existing_sstb
                else simulate_business_is_sstb(
                    qbi_frame,
                    rng=np.random.default_rng(QBI_SSTB_SEED),
                    probability_map=QBI_PARAMS["sstb_prob_map_by_source_name"],
                )
            )
            investment_qbi = simulate_investment_qbi_income_from_puf(
                qbi_frame, rng=np.random.default_rng(QBI_INVESTMENT_SEED)
            )
            legacy_self_employment_income = source_arrays["self_employment_income"]

            overrides = {
                **flag_overrides,
                **investment_qbi,
                "business_is_sstb": is_sstb,
                "self_employment_income": np.where(
                    is_sstb, 0.0, legacy_self_employment_income
                ),
                "sstb_self_employment_income": np.where(
                    is_sstb, legacy_self_employment_income, 0.0
                ),
                "w2_wages_from_qualified_business": w2,
                "unadjusted_basis_qualified_property": ubia,
                "sstb_w2_wages_from_qualified_business": np.where(is_sstb, w2, 0.0),
                "sstb_unadjusted_basis_qualified_property": np.where(
                    is_sstb, ubia, 0.0
                ),
                "self_employment_income_would_be_qualified": np.where(
                    is_sstb, False, self_employment_would_be_qualified
                ),
                SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG: np.where(
                    is_sstb, self_employment_would_be_qualified, False
                ),
            }

        return overrides

    def _ensure_sstb_split_inputs(self) -> dict[str, np.ndarray]:
        overrides = self._sstb_split_overrides()
        overrides.update(self._qbi_simulation_overrides(overrides))
        if not overrides:
            return {}

        try:
            with h5py.File(self.file_path, "r+") as file_handle:
                for key, values in overrides.items():
                    self._replace_array(file_handle, key, values)
        except OSError:
            pass

        return overrides

    class _OverrideView:
        def __init__(self, backing, overrides: dict[str, np.ndarray]):
            self._backing = backing
            self._overrides = overrides

        def __getitem__(self, key):
            if key in self._overrides:
                return self._overrides[key]
            return self._backing[key]

        def __contains__(self, key):
            return key in self._overrides or key in self._backing

        def keys(self):
            if hasattr(self._backing, "keys"):
                return tuple(dict.fromkeys((*self._backing.keys(), *self._overrides)))
            return tuple(self._overrides)

        def get(self, key, default=None):
            if key in self:
                return self[key]
            return default

        def items(self):
            for key in self.keys():
                yield key, self[key]

        def values(self):
            for key in self.keys():
                yield self[key]

        def __iter__(self):
            return iter(self.keys())

        def close(self):
            if hasattr(self._backing, "close"):
                self._backing.close()

        def __enter__(self):
            if hasattr(self._backing, "__enter__"):
                self._backing.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            if hasattr(self._backing, "__exit__"):
                return self._backing.__exit__(exc_type, exc, traceback)
            return None

        def __getattr__(self, name):
            return getattr(self._backing, name)

    def load(self, key=None, mode="r"):
        if mode == "r":
            overrides = self._ensure_sstb_split_inputs()
            if key in overrides:
                return overrides[key]
            if key is None and overrides:
                return self._OverrideView(super().load(key=key, mode=mode), overrides)
        return super().load(key=key, mode=mode)

    def load_dataset(self):
        overrides = self._ensure_sstb_split_inputs()
        arrays = super().load_dataset()
        arrays.update(overrides)
        return arrays

    def generate(self):
        from policyengine_us.system import system

        irs_puf = IRS_PUF_2015(require=True)

        puf = irs_puf.load("puf")
        demographics = irs_puf.load("puf_demographics")

        if self.time_period == 2021:
            puf = uprate_puf(puf, 2015, self.time_period)
        elif self.time_period >= 2021:
            puf_2021 = PUF_2021(require=True)
            uprating = create_policyengine_uprating_factors_table()
            arrays = puf_2021.load_dataset()
            for variable in uprating:
                if variable in arrays:
                    current_index = uprating[uprating.Variable == variable][
                        self.time_period
                    ].values[0]
                    start_index = uprating[uprating.Variable == variable][2021].values[
                        0
                    ]
                    growth = current_index / start_index
                    arrays[variable] = arrays[variable] * growth
            self.save_dataset(arrays)
            return

        puf = disaggregate_aggregate_records(puf)  # 4 rows → ~120 weighted

        original_recid = puf.RECID.values.copy()
        puf = preprocess_puf(puf)
        puf = impute_missing_demographics(puf, demographics)
        # Derive age and is_male for pension imputation predictors
        puf["age"] = puf["AGERANGE"].apply(decode_age_filer)
        puf["is_male"] = (puf["GENDER"] == 1).astype(float)
        puf["pre_tax_contributions"] = impute_pension_contributions_to_puf(
            puf[["employment_income", "age", "is_male"]]
        )

        # Sort in original PUF order
        puf = puf.set_index("RECID").loc[original_recid].reset_index()
        puf = puf.fillna(0)
        self.variable_to_entity = {
            variable: system.variables[variable].entity.key
            for variable in system.variables
        }

        # Filter FINANCIAL_SUBSET to only include variables defined in
        # policyengine-us. This allows us-data to be updated before or after
        # policyengine-us without breaking.
        self.available_financial_vars = [
            v for v in FINANCIAL_SUBSET if v in self.variable_to_entity
        ]

        VARIABLES = [
            "person_id",
            "tax_unit_id",
            "marital_unit_id",
            "spm_unit_id",
            "family_id",
            "household_id",
            "person_tax_unit_id",
            "person_marital_unit_id",
            "person_spm_unit_id",
            "person_family_id",
            "person_household_id",
            "age",
            "household_weight",
            "is_male",
            "filing_status",
            "is_tax_unit_head",
            "is_tax_unit_spouse",
            "is_tax_unit_dependent",
        ] + self.available_financial_vars

        self.holder = {variable: [] for variable in VARIABLES}

        i = 0
        self.earn_splits = []
        for _, row in puf.iterrows():
            i += 1
            exemptions = int(row["exemptions_count"])
            tax_unit_id = row["household_id"]
            self.add_tax_unit(row, tax_unit_id)
            self.add_filer(row, tax_unit_id)
            exemptions -= 1
            if row["filing_status"] == "JOINT":
                self.add_spouse(row, tax_unit_id)
                exemptions -= 1

            for j in range(min(3, exemptions)):
                self.add_dependent(row, tax_unit_id, j)

        groups_assumed_to_be_tax_unit_like = [
            "family",
            "spm_unit",
            "household",
        ]

        for group in groups_assumed_to_be_tax_unit_like:
            self.holder[f"{group}_id"] = self.holder["tax_unit_id"]
            self.holder[f"person_{group}_id"] = self.holder["person_tax_unit_id"]

        for key in self.holder:
            if key == "filing_status":
                self.holder[key] = np.array(self.holder[key]).astype("S")
            else:
                self.holder[key] = np.array(self.holder[key]).astype(float)
                assert not np.isnan(self.holder[key]).any(), f"{key} has NaNs."

        holder_tp = {
            variable: {self.time_period: values}
            for variable, values in self.holder.items()
        }
        if has_policyengine_us_variables(*STRUCTURAL_MORTGAGE_VARIABLES):
            holder_tp = convert_mortgage_interest_to_structural_inputs(
                holder_tp,
                self.time_period,
            )
        self.holder = {
            variable: values[self.time_period] for variable, values in holder_tp.items()
        }
        self.save_dataset(self.holder)

    def add_tax_unit(self, row, tax_unit_id):
        self.holder["tax_unit_id"].append(tax_unit_id)

        for key in self.available_financial_vars:
            if self.variable_to_entity[key] == "tax_unit":
                self.holder[key].append(row[key])

        earnings_split = round(row["EARNSPLIT"])
        if earnings_split > 0:
            SPLIT_DECODES = {
                1: 0.0,
                2: 0.25,
                3: 0.75,
                4: 1.0,
            }
            lower = SPLIT_DECODES[earnings_split]
            upper = SPLIT_DECODES[earnings_split + 1]
            frac = (upper - lower) * rng.random() + lower
            self.earn_splits.append(1.0 - frac)
        else:
            self.earn_splits.append(1.0)

        self.holder["filing_status"].append(row["filing_status"])

    def add_filer(self, row, tax_unit_id):
        person_id = int(tax_unit_id * 1e2 + 1)
        self.holder["person_id"].append(person_id)
        self.holder["person_tax_unit_id"].append(tax_unit_id)
        self.holder["person_marital_unit_id"].append(person_id)
        self.holder["marital_unit_id"].append(person_id)
        self.holder["is_tax_unit_head"].append(True)
        self.holder["is_tax_unit_spouse"].append(False)
        self.holder["is_tax_unit_dependent"].append(False)

        self.holder["age"].append(decode_age_filer(round(row["AGERANGE"])))

        self.holder["household_weight"].append(row["household_weight"])
        self.holder["is_male"].append(row["GENDER"] == 1)

        # Assume all of the interest deduction is the filer's deductible mortgage interest

        self.holder["deductible_mortgage_interest"].append(row["interest_deduction"])

        for key in self.available_financial_vars:
            if key == "deductible_mortgage_interest":
                # Skip this one- we are adding it artificially at the filer level.
                continue
            if self.variable_to_entity[key] == "person":
                self.holder[key].append(row[key] * self.earn_splits[-1])

    def add_spouse(self, row, tax_unit_id):
        person_id = int(tax_unit_id * 1e2 + 2)
        self.holder["person_id"].append(person_id)
        self.holder["person_tax_unit_id"].append(tax_unit_id)
        self.holder["person_marital_unit_id"].append(person_id - 1)
        self.holder["is_tax_unit_head"].append(False)
        self.holder["is_tax_unit_spouse"].append(True)
        self.holder["is_tax_unit_dependent"].append(False)

        self.holder["age"].append(
            decode_age_filer(round(row["AGERANGE"]))
        )  # Assume same age as filer for now

        # 96% of joint filers are opposite-gender

        is_opposite_gender = rng.random() < 0.96
        opposite_gender_code = 0 if row["GENDER"] == 1 else 1
        same_gender_code = 1 - opposite_gender_code
        self.holder["is_male"].append(
            opposite_gender_code if is_opposite_gender else same_gender_code
        )

        # Assume all of the interest deduction is the filer's deductible mortgage interest

        self.holder["deductible_mortgage_interest"].append(0)

        for key in self.available_financial_vars:
            if key == "deductible_mortgage_interest":
                # Skip this one- we are adding it artificially at the filer level.
                continue
            if self.variable_to_entity[key] == "person":
                self.holder[key].append(row[key] * (1 - self.earn_splits[-1]))

    def add_dependent(self, row, tax_unit_id, dependent_id):
        person_id = int(tax_unit_id * 1e2 + 3 + dependent_id)
        self.holder["person_id"].append(person_id)
        self.holder["person_tax_unit_id"].append(tax_unit_id)
        self.holder["person_marital_unit_id"].append(person_id)
        self.holder["marital_unit_id"].append(person_id)
        self.holder["is_tax_unit_head"].append(False)
        self.holder["is_tax_unit_spouse"].append(False)
        self.holder["is_tax_unit_dependent"].append(True)

        age = decode_age_dependent(round(row[f"AGEDP{dependent_id + 1}"]))
        self.holder["age"].append(age)

        # Assume all of the interest deduction is the filer's deductible mortgage interest

        self.holder["deductible_mortgage_interest"].append(0)

        for key in self.available_financial_vars:
            if key == "deductible_mortgage_interest":
                # Skip this one- we are adding it artificially at the filer level.
                continue
            if self.variable_to_entity[key] == "person":
                self.holder[key].append(0)

        self.holder["is_male"].append(rng.choice([0, 1]))


class PUF_2015(PUF):
    label = "PUF 2015"
    name = "puf_2015"
    time_period = 2015
    file_path = STORAGE_FOLDER / "puf_2015.h5"


class PUF_2021(PUF):
    label = "PUF 2021"
    name = "puf_2021"
    time_period = 2021
    file_path = STORAGE_FOLDER / "puf_2021.h5"
    url = "release://policyengine/irs-soi-puf/1.8.0/puf_2021.h5"


class PUF_2023(PUF):
    label = "PUF 2023"
    name = "puf_2023"
    time_period = 2023
    file_path = STORAGE_FOLDER / "puf_2023.h5"


class PUF_2024(PUF):
    label = "PUF 2024 (2015-based)"
    name = "puf_2024"
    time_period = 2024
    file_path = STORAGE_FOLDER / "puf_2024.h5"
    url = "release://policyengine/irs-soi-puf/1.8.0/puf_2024.h5"


# Leave Medicare Part B out of the generic PUF medical-expense split:
# the baseline model now derives Part B premiums separately.
MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS = {
    "health_insurance_premiums_without_medicare_part_b": 0.453,
    "other_medical_expenses": 0.325,
    "over_the_counter_health_expenses": 0.085,
    "medicare_part_b_premium": 0.137,
}

if __name__ == "__main__":
    PUF_2015().generate()
    PUF_2021().generate()
    PUF_2023().generate()
    PUF_2024().generate()
