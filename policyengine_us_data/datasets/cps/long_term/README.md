# Long-Term Income Tax Revenue Projection Methodology
## Integrating Economic Uprating with Demographic Reweighting

### Quick Start

Run projections using `run_household_projection.py`:

```bash
# Recommended: named profile with TOB benchmarked post-calibration
python run_household_projection.py 2100 --profile ss-payroll-tob --target-source trustees_2025_current_law --save-h5

# Experimental: donor-backed late-year support augmentation for tail-year runs
python run_household_projection.py 2075 2100 --profile ss-payroll-tob --target-source trustees_2025_current_law --support-augmentation-profile donor-backed-synthetic-v1 --support-augmentation-target-year 2100 --allow-validation-failures

# Experimental: role-based donor composites assembled into late-year support
python run_household_projection.py 2075 2100 --profile ss-payroll-tob --target-source trustees_2025_current_law --support-augmentation-profile donor-backed-composite-v1 --support-augmentation-target-year 2100 --allow-validation-failures

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
- `--support-augmentation-profile`: Experimental late-year support expansion mode. Currently supports `donor-backed-synthetic-v1` and `donor-backed-composite-v1`.
- `--support-augmentation-target-year`: Extreme year used to build the donor-backed supplement (defaults to `END_YEAR`).
- `--support-augmentation-start-year`: Earliest run year allowed for augmentation (defaults to `2075`).
- `--support-augmentation-top-n-targets`: Number of dominant synthetic target types to map back to real donors (default `20`).
- `--support-augmentation-donors-per-target`: Number of nearest real donor tax units per synthetic target (default `5`).
- `--support-augmentation-max-distance`: Maximum donor-match distance retained for cloning (default `3.0`).
- `--support-augmentation-clone-weight-scale`: Baseline weight multiplier applied to each donor-backed clone (default `0.1`).
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
- Experimental donor-backed augmentation is stamped into each year sidecar and the directory manifest via `support_augmentation`.
- Donor-backed runs now also write a shared `support_augmentation_report.json` artifact with per-clone provenance so late-year translation failures can be inspected directly.

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

**Donor-Backed Late-Year Support Augmentation**
- Experimental late-tail option for `2075+` runs
- Uses the `2100` synthetic-support prototype to identify dominant missing household types
- Maps those synthetic targets back to nearest real 2024 donor tax units
- Clones and perturbs the donor tax units to create a small augmented support without replacing the base CPS sample
- Intended to test whether donor-backed synthetic support improves late-year microsim feasibility without resorting to fully free synthetic records
- Current status: integrated into the runner and fully auditable in metadata, but still diagnostic. The first `2100` end-to-end run did not materially improve the late-tail calibration frontier or support-concentration metrics. After adding clone-provenance diagnostics and donor-specific inverse uprating, the realized clone households now match their intended `2100` age/SS/payroll targets closely; the remaining blocker is how much synthetic support is injected, not whether the actual-row translation works.

**Role-Based Donor Composites**
- Experimental structural extension of the donor-backed approach
- Recombines older-beneficiary donors, payroll-rich worker donors, and dependent structure into synthetic household candidates before assembling actual augmented rows
- In the synthetic support lab, this materially improves the `2100` exact-fit basis: exact entropy fit with `360` positive candidates, ESS about `95.8`, and top-10 weight share about `25.6%`
- The actual-row augmented dataset builder is now available in the runner as `donor-backed-composite-v1`
- Current runner status: the first full `2100` end-to-end run with the default composite supplement added `270` structural clones and modestly improved support concentration (`ESS 11.4 -> 13.3`, top-10 share `84.9% -> 79.4%`, lower TOB overshoot), but it did not yet move the main `SS + payroll` frontier off the current `-33.8% / -35.0%` late-tail bound

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

When donor-backed augmentation is enabled, step 1 uses the original 2024 CPS support and step 2 inserts a tagged late-year supplement derived from nearest real donors before the calibration loop begins. The underlying base dataset path remains unchanged in metadata; the augmentation details are recorded separately in `support_augmentation`, and the full augmentation build report is written once per output directory to `support_augmentation_report.json`.

To compare the intended clone targets with the realized output H5, run:

```bash
uv run python policyengine_us_data/datasets/cps/long_term/diagnose_support_augmentation_translation.py \
  ./projected_datasets/2100.h5 \
  --year 2100
```

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
