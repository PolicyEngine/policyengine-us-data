# Qualified Business Income Simulation

This page documents the model behind
`policyengine_us_data/datasets/puf/qbi_assumptions.yaml`. The YAML file stores
parameters; this page stores the reasoning. Parameter changes should update both
places and should cite reproducible public sources, not chat transcripts or
one-off agent conversations.

The IRS PUF does not directly report the Section 199A inputs used by
PolicyEngine US: source-level QBI qualification flags, W-2 wages from qualified
businesses, UBIA of qualified property, SSTB status, qualified REIT/PTP income,
and qualified BDC income. The simulation is intentionally source-based: it
starts from observed PUF income fields, draws source-level qualification and
industry status, then uses source-weighted operating assumptions for receipts,
payroll, and property.

## Statutory Frame

The model follows the structure in the IRS Form 8995 instructions. QBI can come
from qualified items of income, gain, deduction, and loss from sole
proprietorships, partnerships, S corporations, and certain estates and trusts.
Wage income, reasonable compensation, guaranteed payments, capital gains,
dividends, qualified REIT dividends, and qualified PTP income are not included
in the trade or business QBI base. REIT dividends and PTP income enter the
deduction separately.

SSTBs are modeled separately because PolicyEngine US applies the taxable-income
phaseout to the SSTB path. The IRS SSTB categories include health, law,
accounting, actuarial science, performing arts, consulting, athletics, financial
services, brokerage, investing and investment management, trading, dealing, and
businesses where reputation or skill is the principal asset. The current PUF
does not identify these industries directly, so the model assigns probabilities
by mapped source field and only draws SSTB status for records with a positive
mapped SSTB source.

## Source Mapping

| Model source | Main PUF construction | Reason for separate treatment |
| --- | --- | --- |
| `self_employment_income` | Schedule C net profit or loss, `E00900` | Usually a direct trade or business, with meaningful SSTB mix. |
| `farm_operations_income` | Schedule F income, `E02100` | Active business income, higher capital intensity than typical services. |
| `farm_rent_income` | Farm rental income, `E27200` | Rental or land-heavy income with higher property exposure. |
| `rental_income` | Schedule E rents and royalties, `E25850 - E25860` | Must satisfy the Section 162 trade or business standard or rental safe harbor. |
| `estate_income` | Estate and trust income, `E26390 - E26400` | QBI treatment depends on entity-level reporting and allocation. |
| `partnership_s_corp_income` | Active partnership and S-corp income from Schedule E fields | Pass-through entities report Section 199A detail on K-1 attachments, which the PUF lacks. |

## Qualification Probabilities

The source-level `*_would_be_qualified` probabilities represent whether an
observed income item is eligible to enter the non-SSTB or SSTB QBI path before
taxable-income limitations. They are not meant to decide the final deduction.

| Source | Probability | Rationale |
| --- | ---: | --- |
| `self_employment_income` | 0.95 | Schedule C net business income is generally in scope, with a small discount for items that fail the trade or business, domestic, or reporting tests. |
| `farm_operations_income` | 0.98 | Schedule F active farming is treated as nearly always in scope. |
| `farm_rent_income` | 0.80 | Farm rents are often business or self-rental income, but can include passive land-rent cases. |
| `rental_income` | 0.70 | Schedule E rental income can qualify, but only if it satisfies the Section 162 standard or the rental real estate safe harbor. |
| `estate_income` | 0.60 | Estate and trust amounts require entity-level allocation, which is only partially observed in the PUF. |
| `partnership_s_corp_income` | 0.90 | Ordinary pass-through business income is usually in scope, but K-1 Section 199A detail can exclude some items. |

## SSTB Probabilities

The Schedule C SSTB probability is anchored to IRS 2022 sole proprietorship
industry data. Using positive-net-income rows, selected SSTB-like industries
sum to $165.1 billion of net income out of $543.9 billion for all nonfarm sole
proprietorships, or 30.4 percent. The selected industries are legal services,
accounting, consulting, financial investment and brokerage categories, health
care, performing arts, spectator sports, and related industries. The YAML uses
0.30 for `E00900`.

Partnership and S-corp SSTB status is harder to identify from public aggregate
tables because the industry categories are broader than Section 199A SSTB
categories. For positive-income partnerships, broad IRS 2022 industry totals
show large amounts in finance, professional services, management, health care,
and arts. The model uses 0.25 for `partnership_s_corp_income`: higher than a
pure professional-services-only share, lower than assigning all finance and
management income to SSTBs. Estates and trusts use 0.15 because their PUF fields
do not reveal the underlying trade or business industry.

## W-2 Wages

The wage model has three steps.

1. For each positive qualified source, draw a source-specific profit margin from
   a beta distribution. The record margin is the positive-QBI-weighted average
   across sources, and gross receipts equal positive QBI divided by that margin.
2. Draw whether the record has employees from a logit model:
   `logit(p) = intercept + slope * receipts`. The slope is fixed in YAML and
   the intercept is solved inside the current PUF so the mean employee
   probability among positive-receipt QBI records equals
   `target_share_among_positive_receipts`.
3. Draw a source-weighted labor share of receipts from beta distributions and
   set W-2 wages to `receipts * labor_share` for records with employees.

The 0.18 employee-presence target is a record-level target, not an aggregate
payroll target. It is anchored to the fact that most U.S. businesses have no
paid employees, as measured by Census Nonemployer Statistics, while allowing
larger receipt records to have much higher employee probabilities through the
logit slope. Aggregate payroll levels should be checked against SOI tables after
full PUF regeneration.

| Source | Mean margin | Mean labor share of receipts |
| --- | ---: | ---: |
| `self_employment_income` | 0.300 | 0.125 |
| `farm_operations_income` | 0.180 | 0.098 |
| `farm_rent_income` | 0.230 | 0.013 |
| `rental_income` | 0.277 | 0.013 |
| `estate_income` | 0.230 | 0.089 |
| `partnership_s_corp_income` | 0.214 | 0.110 |

The Schedule C margin is close to the IRS 2022 sole proprietorship positive-net
income margin for all nonfarm industries, 31.5 percent. The farm margin follows
the agriculture, forestry, hunting, and fishing sole-proprietorship margin, 17.2
percent. Real estate and rental sources use low labor shares because SOI
payroll-to-receipts ratios are much lower for those industries than for health,
food, and other labor-heavy industries.

## UBIA

UBIA is modeled as a source-weighted capital-intensity draw. The model first
draws a Bernoulli capital-intensive flag using source-weighted probabilities,
then draws a lognormal amount with mean equal to a source-weighted multiple of
positive QBI and `sigma = 1.0`.

| Source | Capital-intensive probability | UBIA mean multiple of QBI |
| --- | ---: | ---: |
| `self_employment_income` | 0.25 | 1.5 |
| `farm_operations_income` | 0.70 | 4.0 |
| `farm_rent_income` | 0.90 | 8.0 |
| `rental_income` | 0.95 | 10.0 |
| `estate_income` | 0.50 | 3.0 |
| `partnership_s_corp_income` | 0.45 | 3.0 |

These are property-stock assumptions, not depreciation deductions. Rental and
farm-rent income receive the highest probabilities and multiples because real
property and land-heavy businesses have the clearest qualified-property base.
Schedule C receives a lower default because many service businesses have little
qualified property even when they have positive QBI.

## REIT, PTP, and BDC Income

The model does not draw REIT/PTP or BDC income independently for all records.
Instead it scales observed exposure bases:

| Output | Exposure source | Probability | Mean share of exposure |
| --- | --- | ---: | ---: |
| `qualified_reit_and_ptp_income` | `non_qualified_dividend_income` | 0.35 | 0.100 |
| `qualified_reit_and_ptp_income` | `partnership_s_corp_income` | 0.05 | 0.120 |
| `qualified_bdc_income` | `non_qualified_dividend_income` | 0.05 | 0.030 |

The dividend exposure uses non-qualified ordinary dividends because qualified
REIT dividends are reported separately from qualified dividends. The partnership
exposure is a proxy for PTP-related K-1 income. Amounts are clipped to be no
larger than the observed exposure base.

## Validation Targets

The current parameters are source-grounded priors. They should be revisited
when a full PUF regeneration is available. The most important validation checks
are:

- weighted totals for QBI, SSTB QBI, W-2 wages from qualified businesses, UBIA,
  qualified REIT/PTP income, and qualified BDC income;
- source-level shares of records with positive qualification flags;
- payroll-to-receipts and depreciation-to-receipts ratios against SOI industry
  tables;
- distributional sensitivity for high-income taxpayers, where W-2, UBIA, and
  SSTB phaseouts matter most.

## Public Sources

- IRS, [Instructions for Form 8995](https://www.irs.gov/instructions/i8995).
- IRS, [Instructions for Schedule E](https://www.irs.gov/instructions/i1040se).
- IRS SOI, [Individual income tax returns with small business income and losses](https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns-with-small-business-income-and-losses).
- IRS SOI, [Partnership statistics by sector or industry](https://www.irs.gov/statistics/soi-tax-stats-partnership-statistics-by-sector-or-industry).
- IRS SOI, [S corporation statistics](https://www.irs.gov/statistics/soi-tax-stats-s-corporation-statistics).
- Census Bureau, [Nonemployer Statistics](https://www.census.gov/programs-surveys/nonemployer-statistics.html).
- SBA Office of Advocacy, [Nonemployer demographic data project](https://advocacy.sba.gov/2019/09/19/advocacy-and-census-focus-on-the-smallest-small-businesses-with-a-new-nonemployer-demographic-data-project/).
