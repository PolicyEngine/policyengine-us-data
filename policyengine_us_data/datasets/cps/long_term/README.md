# Long-Term Income Tax Revenue Projection Methodology
## Integrating Economic Uprating with Demographic Reweighting

### Quick Start

Run projections using `run_household_projection.py`:

```bash
# Recommended: named profile with core-threshold tax assumption and TOB targeted
python run_household_projection.py 2100 --profile ss-payroll-tob --target-source trustees_2025_current_law --save-h5

# Experimental: donor-backed late-year support augmentation for tail-year runs
python run_household_projection.py 2075 2100 --profile ss-payroll-tob --target-source trustees_2025_current_law --support-augmentation-profile donor-backed-synthetic-v1 --support-augmentation-target-year 2100 --allow-validation-failures

# Experimental: role-based donor composites assembled into late-year support
python run_household_projection.py 2075 2100 --profile ss-payroll-tob --target-source trustees_2025_current_law --support-augmentation-profile donor-backed-composite-v1 --support-augmentation-target-year 2100 --allow-validation-failures

# Experimental: target-year blueprint calibration over donor-composite support
python run_household_projection.py 2100 2100 --profile ss-payroll-tob --target-source trustees_2025_current_law --support-augmentation-profile donor-backed-composite-v1 --support-augmentation-target-year 2100 --support-augmentation-blueprint-base-weight-scale 0.5 --allow-validation-failures --save-h5

# IPF with only age distribution constraints (faster, less accurate)
python run_household_projection.py 2050 --profile age-only

# GREG with age + Social Security only
python run_household_projection.py 2100 --profile ss

# Parallel year-level H5 construction with one subprocess per year
python run_household_projection_parallel.py \
  --years 2026-2035,2045,2049,2062,2063,2070 \
  --jobs 6 \
  --output-dir ./projected_datasets_parallel \
  --profile ss-payroll-tob \
  --target-source oact_2025_08_05_provisional
```

**Arguments:**
- `END_YEAR`: Target year for projection (default: 2035)
- `--profile`: Named calibration contract. Recommended over legacy flags.
- `--target-source`: Named long-term target source package.
- `--tax-assumption`: Long-run federal tax assumption. Defaults to `trustees-core-thresholds-v1`; use `current-law-literal` to opt out.
- `--output-dir`: Output directory for generated H5 files and metadata sidecars.
- `--support-augmentation-profile`: Experimental late-year support expansion mode. Currently supports `donor-backed-synthetic-v1` and `donor-backed-composite-v1`.
- `--support-augmentation-target-year`: Extreme year used to build the donor-backed supplement (defaults to `END_YEAR`).
- `--support-augmentation-align-to-run-year`: Rebuild the donor-backed supplement separately for each run year instead of reusing one target-year support snapshot.
- `--support-augmentation-start-year`: Earliest run year allowed for augmentation (defaults to `2075`).
- `--support-augmentation-top-n-targets`: Number of dominant synthetic target types to map back to real donors (default `20`).
- `--support-augmentation-donors-per-target`: Number of nearest real donor tax units per synthetic target (default `5`).
- `--support-augmentation-max-distance`: Maximum donor-match distance retained for cloning (default `3.0`).
- `--support-augmentation-clone-weight-scale`: Baseline weight multiplier applied to each donor-backed clone (default `0.1`).
- `--support-augmentation-blueprint-base-weight-scale`: When donor-composite augmentation is active at its target year, scales the original household priors before replacing clone priors with synthetic blueprint shares (default `0.5`).
- `--greg`: Use GREG calibration instead of IPF
- `--use-ss`: Include Social Security benefit totals as calibration target (requires `--greg`)
- `--use-payroll`: Include taxable payroll totals as calibration target (requires `--greg`)
- `--use-tob`: Include TOB (Taxation of Benefits) revenue as calibration target (requires `--greg`)
- `--save-h5`: Save year-specific .h5 files to `./projected_datasets/` directory

**Parallel wrapper:**
- `run_household_projection_parallel.py` runs one `run_household_projection.py YEAR YEAR ...` subprocess per year and merges the resulting H5 artifacts into one output directory.
- The wrapper forces `--save-h5` and controls `--output-dir` itself, so those flags should not be forwarded to the inner runner.
- Per-year stdout/stderr logs are written under `OUTPUT_DIR/.parallel_logs/`.

**Named profiles:**
- `age-only`: IPF age-only calibration
- `ss`: positive entropy calibration with age + Social Security
- `ss-payroll`: positive entropy calibration with age + Social Security + taxable payroll
- `ss-payroll-tob`: positive entropy calibration with age + Social Security + taxable payroll + TOB under the long-run core-threshold tax assumption
- `ss-payroll-tob-h6`: positive entropy calibration with age + Social Security + taxable payroll + TOB + H6 under the long-run core-threshold tax assumption

**Validation contract:**
- Economic-targeted profiles no longer silently pretend an IPF fallback is equivalent to GREG.
- Named economic profiles must produce non-negative weights.
- Each generated H5 now gets a `YYYY.h5.metadata.json` sidecar with profile and calibration audit details.
- Each generated H5 sidecar now records the named long-term target source used for the build.
- Each output directory now gets a `calibration_manifest.json` file describing the
  profile/base dataset contract for the full artifact set.
- Profiles validate achieved constraint errors before writing output.
- Experimental donor-backed augmentation is stamped into each year sidecar and the directory manifest via `support_augmentation`.
- The active long-run tax assumption is stamped into each year sidecar and the directory manifest via `tax_assumption`.
- Donor-backed runs now also write a shared `support_augmentation_report.json` artifact with per-clone provenance so late-year translation failures can be inspected directly.
- Long-run payroll calibration now guards against a flat Social Security wage base after 2035. If `policyengine-us` is missing the NAWI / payroll-cap extension, late-year payroll runs fail fast instead of silently mis-targeting taxable payroll.
- Trustees/OACT tax-side assumptions are documented in [ASSUMPTION_COMPARISON.md](./ASSUMPTION_COMPARISON.md). The active long-run baseline adopts a core-threshold bundle:
  - Social Security benefit-tax thresholds remain fixed in nominal dollars under Trustees current law.
  - Core ordinary federal thresholds are assumed to rise with average wages after the tenth projection year.
- The corrected apples-to-apples TOB share comparison is documented in
  [TOB_ALIGNMENT_NOTE.md](./TOB_ALIGNMENT_NOTE.md), with a small reproduction
  script in [compare_tob_shares.py](./compare_tob_shares.py).

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

**Long-Run Core-Threshold Tax Assumption**
- Default long-run tax assumption in this runner
- Keeps Social Security benefit-tax thresholds fixed
- Wage-indexes a core set of federal thresholds after `2034`:
  - ordinary income-tax brackets
  - standard deduction
  - aged/blind additional standard deduction
  - capital-gains thresholds
  - AMT thresholds / exemptions
- Intended as the best public Trustees approximation before TOB is hard-targeted again

**Donor-Backed Late-Year Support Augmentation**
- Experimental late-tail option for `2075+` runs
- Uses the `2100` synthetic-support prototype to identify dominant missing household types
- Maps those synthetic targets back to nearest real 2024 donor tax units
- Clones and perturbs the donor tax units to create a small augmented support without replacing the base CPS sample
- Intended to test whether donor-backed synthetic support improves late-year microsim feasibility without resorting to fully free synthetic records
- Current status: still diagnostic. The simple nearest-neighbor donor supplement does not materially improve the late-tail fit once the calibration uses SSA taxable payroll rather than uncapped wages.

**Role-Based Donor Composites**
- Experimental structural extension of the donor-backed approach
- Recombines older-beneficiary donors, payroll-rich worker donors, and dependent structure into synthetic household candidates before assembling actual augmented rows
- The actual-row augmented dataset builder is now available in the runner as `donor-backed-composite-v1`
- Current status:
  - Fixing the long-run payroll-cap bug in `policyengine-us` changed the picture materially. With the correct SSA wage base extended through `2100`, the donor-composite synthetic support is exact-feasible and dense at the archetype level.
  - The runner now supports a target-year calibration blueprint for donor-composite augmentation. At the augmentation target year, it can calibrate against the exact clone blueprints and synthetic prior shares while still auditing the realized rows.
  - In the current `2100` probe, that blueprint path gets actual age + SS + taxable payroll very close while keeping support quality in range: with `--support-augmentation-blueprint-base-weight-scale 0.5`, actual payroll miss is about `-0.86%`, ESS about `102.5`, top-10 weight share about `24.4%`, and top-100 share about `68.4%`.
  - The runner now also has a dynamic mode, `--support-augmentation-align-to-run-year`, that rebuilds donor-composite support for each run year and writes per-year augmentation reports.
  - This is still experimental. The blueprint path is now structurally capable of handling year-specific support, but the full `2075-2100` production sweep still needs runtime tuning and caching work.

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
   - Guardrail: the runner checks that the Social Security taxable earnings cap continues to rise after 2035. A flat cap indicates an invalid `policyengine-us` parameter baseline for long-run payroll work.

4. **TOB Revenue** (`ss-payroll-tob`, `ss-payroll-tob-h6`, or legacy `--use-tob`)
   - Taxation of Benefits revenue for OASDI and Medicare HI trust funds
   - OASDI: `tob_revenue_oasdi` (tier 1 taxation, 0-50% of benefits)
   - HI: `tob_revenue_medicare_hi` (tier 2 taxation, 50-85% of benefits)
   - Source: selected long-term target source package
   - In the current branch contract, TOB is hard-targeted under the long-run core-threshold tax assumption rather than under literal CPI-style current-law bracket indexing.
   - Primary-source note: Trustees do not appear to model long-run TOB by indexing the SS-specific `$25k/$32k/$0` and `$34k/$44k` thresholds. Those remain fixed under current law; the long-run divergence comes from broader tax-side assumptions, including ordinary bracket treatment after the tenth projection year.

---

### Data Sources

**SSA 2025 OASDI Trustees Report**
- URL: https://www.ssa.gov/OACT/TR/2025/
- File: `SingleYearTRTables_TR2025.xlsx`
- Tables: IV.B2 (OASDI TOB % of taxable payroll), VI.G6 (taxable payroll in billions), VI.G9 (OASDI costs)
- Program assumptions: [V.C.7](https://www.ssa.gov/oact/tr/2025/V_C_prog.html) documents fixed Social Security benefit-tax thresholds and the long-run ordinary-bracket assumption.

**CMS 2025 Medicare Trustees Report**
- URL: https://www.cms.gov/data-research/statistics-trends-and-reports/trustees-report-trust-funds
- File: `tr2025-tables-figures.zip` → CSV folder → "Medicare Sources of Non-Interest Income..."
- Column: Tax on Benefits (values in millions, 2024-2099)

**Local files** (in `policyengine_us_data/storage/`):
- `SSPopJul_TR2024.csv` - Population projections 2025-2100 by single year of age
- `long_term_target_sources/trustees_2025_current_law.csv` - explicit frozen Trustees/current-law package
- `long_term_target_sources/oact_2025_08_05_provisional.csv` - OACT-updated TOB package with provisional HI bridge
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
- **`build_long_term_target_sources.py`** - Rebuild named long-term target source packages
- **`projection_utils.py`** - Utility functions (age matrix builder, H5 file creator)
- **`extract_ssa_costs.py`** - One-time script to extract SSA data from Excel (already run)

---

### Methodology Overview

For each projection year (2025-2100):

1. **Load base microdata** - CPS 2024 Enhanced dataset
2. **Uprate variables** - PolicyEngine automatically uprates income, thresholds, etc. to target year
3. **Calculate values** - Income tax, Social Security, taxable payroll at household level
4. **Calibrate weights** - Adjust household weights to match SSA demographic/economic targets
5. **Target or benchmark TOB** - Under `ss-payroll-tob`, match modeled OASDI/HI TOB to the selected target source using the core-threshold tax assumption
6. **Aggregate results** - Apply calibrated weights to calculate national totals

When donor-backed augmentation is enabled, step 1 uses the original 2024 CPS support and step 2 inserts a tagged late-year supplement derived from nearest real donors before the calibration loop begins. The underlying base dataset path remains unchanged in metadata; the augmentation details are recorded separately in `support_augmentation`, and the full augmentation build report is written once per output directory to `support_augmentation_report.json`.

When donor-composite augmentation is enabled and the run year equals the augmentation target year, the runner can also replace the clone rows' calibration constraints with their exact synthetic blueprint values and use the synthetic-support solution as clone priors. The calibration audit still reports achieved constraints on the realized rows, so any blueprint-to-row translation gap remains visible in metadata.

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
