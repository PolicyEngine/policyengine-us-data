# Long-Term Income Tax Revenue Projection Methodology
## Integrating Economic Uprating with Demographic Reweighting

### Quick Start

Run projections using `run_household_projection.py`:

```bash
# Recommended: named profile with TOB benchmarked post-calibration
python run_household_projection.py 2100 --profile ss-payroll-tob --target-source trustees_2025_current_law --save-h5

# IPF with only age distribution constraints (faster, less accurate)
python run_household_projection.py 2050 --profile age-only

# GREG with age + Social Security only
python run_household_projection.py 2100 --profile ss
```

**Arguments:**
- `END_YEAR`: Target year for projection (default: 2035)
- `--profile`: Named calibration contract. Recommended over legacy flags.
- `--target-source`: Named long-term target source package.
- `--output-dir`: Output directory for generated H5 files and metadata sidecars.
- `--greg`: Use GREG calibration instead of IPF
- `--use-ss`: Include Social Security benefit totals as calibration target (requires `--greg`)
- `--use-payroll`: Include taxable payroll totals as calibration target (requires `--greg`)
- `--use-tob`: Include TOB (Taxation of Benefits) revenue as calibration target (requires `--greg`)
- `--save-h5`: Save year-specific .h5 files to `./projected_datasets/` directory

**Named profiles:**
- `age-only`: IPF age-only calibration
- `ss`: positive entropy calibration with age + Social Security
- `ss-payroll`: positive entropy calibration with age + Social Security + taxable payroll
- `ss-payroll-tob`: positive entropy calibration with age + Social Security + taxable payroll, with TOB benchmarked after calibration
- `ss-payroll-tob-h6`: positive entropy calibration with age + Social Security + taxable payroll + H6, with TOB benchmarked after calibration

**Validation contract:**
- Economic-targeted profiles no longer silently pretend an IPF fallback is equivalent to GREG.
- Named economic profiles must produce non-negative weights.
- Each generated H5 now gets a `YYYY.h5.metadata.json` sidecar with profile and calibration audit details.
- Each generated H5 sidecar now records the named long-term target source used for the build.
- Each output directory now gets a `calibration_manifest.json` file describing the
  profile/base dataset contract for the full artifact set.
- Profiles validate achieved constraint errors before writing output.

**Estimated runtime:** ~2 minutes/year without `--save-h5`, ~3 minutes/year with `--save-h5`

---

### Calibration Methods

**IPF (Iterative Proportional Fitting)**
- Adjusts weights to match age distribution only (86 categories: ages 0-85+)
- Fast and simple, but cannot enforce Social Security or payroll totals
- Converges iteratively (typically 20-40 iterations)

**Positive Entropy Calibration**
- Solves for strictly positive weights matching multiple constraints simultaneously
- Can enforce age distribution + Social Security benefits + taxable payroll
- Uses dual optimization to minimize divergence from baseline weights
- **Recommended** for publishable long-term projections

**GREG (Generalized Regression Estimator)**
- Legacy linear calibration path retained for explicit flag-based runs
- Can hit constraints exactly, but may produce negative weights in far-horizon years
- No longer the default for named economic calibration profiles

---

### Constraint Types

1. **Age Distribution** (always active)
   - 86 categories: ages 0-84 individually, 85+ aggregated
   - Source: SSA population projections (`SSPopJul_TR2024.csv`)

2. **Social Security Benefits** (`--use-ss`, GREG only)
   - Total OASDI benefit payments (nominal dollars)
   - Source: selected long-term target source package

3. **Taxable Payroll** (`--use-payroll`, GREG only)
   - W-2 wages capped at wage base + SE income within remaining cap room
   - Calculated as: `taxable_earnings_for_social_security` + `social_security_taxable_self_employment_income`
   - Source: selected long-term target source package

4. **TOB Revenue** (`--use-tob`, legacy hard-target mode only)
   - Taxation of Benefits revenue for OASDI and Medicare HI trust funds
   - OASDI: `tob_revenue_oasdi` (tier 1 taxation, 0-50% of benefits)
   - HI: `tob_revenue_medicare_hi` (tier 2 taxation, 50-85% of benefits)
   - Source: selected long-term target source package
   - Recommended usage: benchmark after calibration rather than use as a hard weight target

---

### Data Sources

**SSA 2025 OASDI Trustees Report**
- URL: https://www.ssa.gov/OACT/TR/2025/
- File: `SingleYearTRTables_TR2025.xlsx`
- Tables: IV.B2 (OASDI TOB % of taxable payroll), VI.G6 (taxable payroll in billions), VI.G9 (OASDI costs)

**CMS 2025 Medicare Trustees Report**
- URL: https://www.cms.gov/data-research/statistics-trends-and-reports/trustees-report-trust-funds
- File: `tr2025-tables-figures.zip` → CSV folder → "Medicare Sources of Non-Interest Income..."
- Column: Tax on Benefits (values in millions, 2024-2099)

**Local files** (in `policyengine_us_data/storage/`):
- `SSPopJul_TR2024.csv` - Population projections 2025-2100 by single year of age
- `long_term_target_sources/trustees_2025_current_law.csv` - explicit frozen Trustees/current-law package
- `long_term_target_sources/sources.json` - provenance metadata for named source packages
- `ASSUMPTION_COMPARISON.md` - side-by-side summary of our calibration assumptions versus Trustees/OACT

---

### Output Files

**Year-specific .h5 files** (when using `--save-h5`)
- Saved to: `./projected_datasets/YYYY.h5`
- Contains: All CPS microdata variables uprated to year YYYY with calibrated household weights
- Usage: Load directly with PolicyEngine: `Microsimulation(dataset="./projected_datasets/2050.h5")`

**Variables included:**
- All demographic variables (age, household composition, state, etc.)
- All income variables (wages, Social Security, capital income, etc.) - uprated to target year
- `household_weight` - calibrated to match age distribution and optional SS/payroll totals

---

### Files in this Directory

- **`run_household_projection.py`** - Main projection script (see Quick Start)
- **`calibration.py`** - IPF and GREG weight calibration implementations
- **`ssa_data.py`** - Load SSA population and named long-term target source projections
- **`projection_utils.py`** - Utility functions (age matrix builder, H5 file creator)
- **`extract_ssa_costs.py`** - One-time script to extract SSA data from Excel (already run)

---

### Methodology Overview

For each projection year (2025-2100):

1. **Load base microdata** - CPS 2024 Enhanced dataset
2. **Uprate variables** - PolicyEngine automatically uprates income, thresholds, etc. to target year
3. **Calculate values** - Income tax, Social Security, taxable payroll at household level
4. **Calibrate weights** - Adjust household weights to match SSA demographic/economic targets
5. **Benchmark TOB** - Compare modeled OASDI/HI TOB to the selected target source without forcing it into the weights
6. **Aggregate results** - Apply calibrated weights to calculate national totals

**Key innovation:** Household-level calculations avoid person→household aggregation issues, maintaining consistency across all variables.

---

### Validating Results

After generating calibrated datasets, verify calibration accuracy:

```python
from policyengine_us import Microsimulation

# Load calibrated dataset
sim = Microsimulation(dataset="./projected_datasets/2050.h5")

# Check Social Security benefits (should match target from social_security_aux.csv)
ss_total = sim.calculate("social_security").sum() / 1e9
print(f"Social Security 2050: ${ss_total:.1f}B")

# Check taxable payroll
payroll = (
    sim.calculate("taxable_earnings_for_social_security").sum() / 1e9 +
    sim.calculate("social_security_taxable_self_employment_income").sum() / 1e9
)
print(f"Taxable payroll 2050: ${payroll:.1f}B")

# Check population (should match SSA population projections)
pop = sim.calculate("people").sum() / 1e6
print(f"Population 2050: {pop:.1f}M")
```

**Expected 2050 values** (from SSA 2024 TR):
- Social Security: $4,802.5B
- Taxable payroll: $28,300.0B
- Population: ~389.5M

When using `--greg --use-ss --use-payroll`, calibration should achieve **< 0.1% error** on all constraints.
