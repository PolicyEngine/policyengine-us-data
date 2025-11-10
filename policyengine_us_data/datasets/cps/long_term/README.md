# Long-Term Income Tax Revenue Projection Methodology
## Integrating Economic Uprating with Demographic Reweighting

### Executive Summary

This document outlines an innovative approach for projecting federal income tax revenue through 2100 that uniquely combines sophisticated economic microsimulation with demographic reweighting. By harmonizing PolicyEngine's state-of-the-art tax modeling with Social Security Administration demographic projections, we can isolate and quantify the fiscal impact of population aging while preserving the full complexity of the tax code.

---

## The Challenge

Projecting tax revenue over a 75-year horizon requires simultaneously modeling two distinct but interrelated dynamics:

**Economic Evolution**: How incomes, prices, and tax parameters change over time
- Wage growth and income distribution shifts
- Inflation affecting brackets and deductions
- Legislative changes and indexing rules
- Behavioral responses to tax policy

**Demographic Transformation**: How the population structure evolves
- Baby boom generation aging through retirement
- Declining birth rates reducing working-age population
- Increasing longevity extending retirement duration
- Shifting household composition patterns

Traditional approaches typically sacrifice either economic sophistication (using simplified tax calculations) or demographic realism (holding age distributions constant). Our methodology preserves both.

---

## Core Innovation

Our approach operates in two complementary stages:

### Stage 1: Economic Uprating

PolicyEngine's microsimulation engine projects each household's economic circumstances forward using:

**Sophisticated Income Modeling**

The system models 17 distinct income categories, each uprated according to its economic fundamentals:

*Primary Categories with Specific Projections:*
- Employment income (wages) - follows CBO wage growth projections
- Self-employment income - follows CBO business income projections
- Capital gains - follows CBO asset appreciation projections
- Interest income - follows CBO interest rate projections
- Dividend income - follows CBO corporate profit projections
- Pension income - follows CBO retirement income projections
- Social Security - follows SSA COLA projections (available through 2100)

*Secondary Categories with Derived Projections:*
- Business income (partnerships, S-corps) - follows self-employment patterns
- Farm income - follows agricultural economic indices
- Retirement distributions (401k, IRA) - follows pension uprating
- Unemployment compensation - follows labor market projections

*Minor Categories with AGI-Based Uprating:*
- Miscellaneous income, debt relief, and other small categories follow total Adjusted Gross Income growth as a reasonable proxy

This tiered approach maintains realistic income composition evolution while remaining computationally tractable. Behavioral responses including labor supply and capital gains realization elasticities further refine projections.

**Complete Tax Code Implementation**
- Progressive rate structures with all brackets and thresholds
- Complex interactions between provisions (AMT, capital gains rates, phase-outs)
- Full constellation of credits and deductions with eligibility rules
- Automatic indexing of parameters following current law

**Calibrated Projections**
- Individual microdata scaled to match aggregate national totals
- Maintains empirically-observed income distributions
- Preserves household heterogeneity and geographic variation

### Stage 2: Demographic Reweighting

We offer two calibration methods for adjusting household weights to match SSA projections:

**Method 1: Iterative Proportional Fitting (IPF)**
- Traditional raking approach using Kullback-Leibler divergence
- Iteratively adjusts weights to match marginal distributions
- Robust to specification and always produces non-negative weights
- Default method for backward compatibility

**Method 2: Generalized Regression (GREG) Calibration**
- Modern calibration using chi-squared distance minimization
- Enables simultaneous calibration to categorical AND continuous variables
- Direct solution via matrix operations (no iteration needed)
- Required for incorporating Social Security benefit constraints

**Age-Specific Targets**
- Single-year age groups from 0-85+
- Annual projections through 2100 from SSA Trustees Report
- Captures complete demographic transition including:
  - Working-age population share declining from 57.5% to 53.2%
  - Elderly population share rising from 18.6% to 31.0%

**Enhanced Calibration with Social Security Benefits** (GREG only)
- Incorporates OASDI cost projections from SSA Trustees Report Table VI.G9
- Ensures total Social Security benefits match official projections
- Values from CPI-indexed 2025 dollars converted to nominal using 2.4% inflation
- Creates consistency between demographic and fiscal projections

**Preservation of Economic Relationships**
- Both methods adjust only the frequency weights, not dollar amounts
- Maintains within-age income distributions
- Preserves household composition and filing status patterns
- Keeps all tax calculations and behavioral responses intact

---

## Methodological Advantages

### Dual Projection Capability

The approach produces two parallel projections:

**Economic Baseline**: PolicyEngine's standard projection with economic uprating but constant demographics
- Shows pure effect of income growth and inflation
- Maintains 2024 population age structure
- Isolates economic policy impacts

**Demographic-Adjusted**: Incorporates SSA population aging via reweighted households
- Captures shifting age composition effects
- Reflects changing dependency ratios
- Quantifies demographic fiscal pressure

The difference between these projections isolates the pure demographic effect on tax revenue.

### Preservation of Complexity

Unlike aggregate models, this approach maintains:

**Tax Code Nonlinearities**
- Progressive marginal rate jumps
- Cliff effects in credit eligibility
- Interaction effects between provisions
- State tax deductibility and other recursive calculations

**Household Heterogeneity**
- Income inequality within age groups
- Diverse household structures
- Geographic variation in incomes and taxes
- Full range of filing statuses and dependent arrangements

**Behavioral Responses**
- Labor supply adjustments to marginal tax rates
- Capital gains realization timing
- Tax-deferred savings decisions

---

## Data Foundation and Extensions Needed

### Current State

**Economic Projections**
- CBO revenue projections through 2035
- Inflation indices extended via constant growth rates
- Population totals from CBO demographic outlook (ending 2055)

**Demographic Data**
- SSA Trustees Report with age-specific projections to 2100
- Captures low/intermediate/high-cost scenarios
- Includes mortality, fertility, and immigration assumptions

**Social Security Benefit Projections** (New)
- SSA 2025 Trustees Report, Table VI.G9, Column C (OASDI Cost)
- Intermediate cost scenario in CPI-indexed 2025 dollars
- Converted to nominal using 2.4% annual inflation assumption
- Complete projections from 2025-2100 with interpolation for missing years

### Required Extensions for Full Implementation

**Population Calibration Update**
- Transition from CBO population totals (ending 2055) to SSA projections (through 2100)
- Ensures consistency with age-specific targets used in reweighting
- Affects all uprated variables through household weights

**Long-Term Economic Indices**
- Incorporate CBO Long-Term Budget Outlook inflation projections (through ~2054)
- Update CPI-U, Chained CPI, and CPI-W with long-term forecasts
- Apply consistent growth assumptions for 2054-2100 extension

**Income Category Projections Strategy**

*Through 2035:* Use existing CBO detailed projections by income source

*2035-2054:* Map to CBO Long-Term Budget Outlook indicators:
- Employment income → CBO wage growth projections
- Business/self-employment → CBO GDP growth rates
- Capital gains → Asset price growth (GDP + equity premium)
- Interest income → CBO interest rate projections
- Dividends → Corporate profit share of GDP
- Retirement income → Demographic-adjusted wage growth
- Social Security → SSA Trustees Report (already available to 2100)

*2054-2100:* Extend using constant real growth differentials:
- Maintain relative growth rates from 2053-2054
- Preserve compositional changes observed in CBO projections
- Apply consistent methodology across all categories

This practical approach balances analytical rigor with data availability, ensuring robust projections without requiring unrealistic precision in minor income categories.

These extensions would create a fully consistent projection framework spanning the entire 75-year horizon.

---

## Applications and Insights

This methodology enables unprecedented analysis of:

### Fiscal Sustainability
- Decompose revenue changes into economic vs. demographic components
- Quantify the "demographic dividend" as baby boomers remain in peak earning years
- Project the "demographic drag" as they transition to retirement

### Policy Design
- Evaluate reforms in context of future demographics
- Test robustness of proposals across the demographic transition
- Design policies that maintain generational equity

### Distributional Analysis
- Track tax burden shifts between age cohorts over time
- Analyze progressivity evolution under changing demographics
- Assess sustainability of age-targeted provisions

### Scenario Planning
- Model alternative demographic scenarios (immigration, fertility)
- Test sensitivity to economic growth assumptions
- Evaluate policy responses to demographic pressures

---

## Validation and Quality Metrics

### Calibration Accuracy
- Economic aggregates match IRS Statistics of Income within 2%
- Population totals exactly match SSA projections
- Age distributions preserved through reweighting process
- Social Security benefit totals match SSA Trustees Report projections (GREG with SS)

### Computational Metrics
- IPF convergence typically within 40-50 iterations
- GREG provides direct solution (no iteration required)
- Mean absolute weight adjustment: 5-7% across both methods
- Relative errors below 10⁻⁶ for age targets
- No negative weights when properly specified

### Method Comparison (2025 Projection)
| Method | Income Tax | Weight Adjustment | Convergence |
|--------|-----------|------------------|-------------|
| IPF (age only) | $2,101.2B | 5.5% | 41 iterations |
| GREG (age only) | $2,097.7B | 5.3% | Direct |
| GREG (age + SS) | $2,088.6B | 6.5% | Direct |

- Methods agree within 0.2% for age-only calibration
- Social Security constraint appropriately shifts weight toward retirees

### Robustness Checks
- IPF and GREG produce equivalent results for identical constraints
- Both methods validated against R's survey package (gold standard)
- Consistent with CBO long-term fiscal projections where comparable
- Aligned with Social Security Trustees Report on both demographics and benefits

---

## Technical Implementation Notes

### Usage

The projection system supports three operational modes:

```bash
# Traditional IPF approach (default)
python run_full_projection.py 2050

# GREG calibration with demographic constraints only
python run_full_projection.py 2050 --greg

# GREG with demographics + Social Security benefits
python run_full_projection.py 2050 --greg --use-ss
```

### Critical Implementation Lesson: Avoiding Multicollinearity

During GREG implementation, we encountered a subtle but critical issue that serves as an important lesson for calibration methods:

**The Problem**: Initial implementation included both:
- An intercept term (for total population control)
- All 86 age category indicators

This created perfect multicollinearity since the age categories sum to the total population, resulting in:
- Singular constraint matrix (non-invertible)
- Pathological solution with 25% of households receiving negative weights
- Weight values ranging from -$1.9M to +$4.0M
- Meaningless projections despite technically satisfying constraints

**The Solution**: Remove either:
- The intercept (use all age categories, total is implicit), or
- One age category (use intercept + 85 categories, last age is implicit)

This is the classic "dummy variable trap" from econometrics, demonstrating that even well-established methods require careful implementation. Both Python's samplics and R's survey package exhibited identical behavior with multicollinear constraints, confirming this is a fundamental mathematical issue, not an implementation bug.

### Theoretical Foundation

Both IPF and GREG are special cases of the Deville-Särndal (1992) calibration framework, minimizing distance from initial weights while meeting constraints:

- **IPF**: Minimizes Kullback-Leibler divergence, multiplicative adjustments
- **GREG**: Minimizes chi-squared distance, additive adjustments

For well-posed problems, both methods produce nearly identical results, as our validation confirms.

---

## Conclusion

This enhanced methodology represents a significant advance in long-term fiscal projection, uniquely combining:

1. **Economic sophistication** through PolicyEngine's complete tax code implementation
2. **Demographic realism** via SSA's authoritative population projections
3. **Methodological flexibility** with both IPF and GREG calibration options
4. **Fiscal consistency** by incorporating Social Security benefit projections
5. **Analytical power** to decompose and understand revenue dynamics

The addition of GREG calibration with continuous variable support marks a major enhancement, enabling:
- Simultaneous calibration to demographics AND benefit payments
- Future extensions for Medicare, labor force participation, and other continuous targets
- Validation that both methods produce consistent results for comparable constraints
- Greater confidence through cross-validation with R's survey package

By integrating the best available economic projections with official demographic and fiscal forecasts, this approach provides policymakers with an unprecedented tool for understanding and preparing for the fiscal challenges of an aging society.

The framework's modular design allows for continuous improvement as new projections become available, ensuring analyses remain grounded in the latest economic and demographic intelligence. With the planned extensions to incorporate long-term CBO economic projections and the successful integration of SSA benefit data through 2100, this methodology provides a comprehensive foundation for evidence-based fiscal policy in an era of demographic transformation.

---

*Document Version: 2.0*
*Methodology: Economic Uprating + Demographic Reweighting via IPF/GREG*
*Calibration Targets: Age Distribution + Social Security Benefits (optional)*
*Projection Horizon: 2025-2100*
*Last Updated: October 2024*