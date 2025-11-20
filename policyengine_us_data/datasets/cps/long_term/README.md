# Long-Term Income Tax Revenue Projection Methodology
## Integrating Economic Uprating with Demographic Reweighting

### Quick Start

Run projections using `run_household_projection.py`:

```bash
# Recommended: GREG with all three constraint types
python run_household_projection.py 2100 --greg --use-ss --use-payroll --save-h5

# IPF with only age distribution constraints (faster, less accurate)
python run_household_projection.py 2050

# GREG with age + Social Security only
python run_household_projection.py 2100 --greg --use-ss
```

**Arguments:**
- `END_YEAR`: Target year for projection (default: 2035)
- `--greg`: Use GREG calibration instead of IPF
- `--use-ss`: Include Social Security benefit totals as calibration target (requires `--greg`)
- `--use-payroll`: Include taxable payroll totals as calibration target (requires `--greg`)
- `--save-h5`: Save year-specific .h5 files to `./projected_datasets/` directory

**Estimated runtime:** ~2 minutes/year without `--save-h5`, ~3 minutes/year with `--save-h5`

---

### Calibration Methods

**IPF (Iterative Proportional Fitting)**
- Adjusts weights to match age distribution only (86 categories: ages 0-85+)
- Fast and simple, but cannot enforce Social Security or payroll totals
- Converges iteratively (typically 20-40 iterations)

**GREG (Generalized Regression Estimator)**
- Solves for weights matching multiple constraints simultaneously
- Can enforce age distribution + Social Security benefits + taxable payroll
- One-shot solution using `samplics` package
- **Recommended** for accurate long-term projections

---

### Constraint Types

1. **Age Distribution** (always active)
   - 86 categories: ages 0-84 individually, 85+ aggregated
   - Source: SSA population projections (`SSPopJul_TR2024.csv`)

2. **Social Security Benefits** (`--use-ss`, GREG only)
   - Total OASDI benefit payments (nominal dollars)
   - Source: SSA Trustee Report 2024 (`social_security_aux.csv`)

3. **Taxable Payroll** (`--use-payroll`, GREG only)
   - W-2 wages capped at wage base + SE income within remaining cap room
   - Calculated as: `taxable_earnings_for_social_security` + `social_security_taxable_self_employment_income`
   - Source: SSA Trustee Report 2024 (`social_security_aux.csv`)

---

### Data Sources

All data from **SSA 2024 Trustee Report**:

- `SSPopJul_TR2024.csv` - Population projections 2025-2100 by single year of age
- `social_security_aux.csv` - OASDI costs and taxable payroll projections 2025-2100
  - Extracted from `SingleYearTRTables_TR2025.xlsx` Table VI.G9 using `extract_ssa_costs.py`

Files located in: `policyengine_us_data/storage/`

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
- **`ssa_data.py`** - Load SSA population, benefit, and payroll projections
- **`projection_utils.py`** - Utility functions (age matrix builder, H5 file creator)
- **`extract_ssa_costs.py`** - One-time script to extract SSA data from Excel (already run)

---

### Methodology Overview

For each projection year (2025-2100):

1. **Load base microdata** - CPS 2024 Enhanced dataset
2. **Uprate variables** - PolicyEngine automatically uprates income, thresholds, etc. to target year
3. **Calculate values** - Income tax, Social Security, taxable payroll at household level
4. **Calibrate weights** - Adjust household weights to match SSA demographic/economic targets
5. **Aggregate results** - Apply calibrated weights to calculate national totals

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
