# Geo-Stacking Calibration

Creates state-level microsimulation datasets with Congressional District (CD) level calibration weights. Takes Current Population Survey (CPS) data, enriches it with Public Use File (PUF) income variables, applies L0 sparse calibration to match ~34k demographic and economic targets across 436 Congressional Districts, and produces optimized datasets for each US state.

**Key Achievement**: Reduces ~200k household dataset to ~13k households while maintaining statistical representativeness across all 436 CDs through sophisticated weight calibration.

## Quick Start

### Local Testing (100 epochs)
```bash
make data-geo
make calibration-package
make optimize-weights-local
make create-state-files
```

### Production (GCP, 4000 epochs)
```bash
make data-geo
make upload-calibration-package  # Note the date prefix shown
# Edit batch_pipeline/config.env with INPUT_PATH and OUTPUT_PATH
make optimize-weights-gcp
make download-weights-from-gcs
make create-state-files
make upload-state-files-to-gcs
```

## Pipeline Architecture

```
Phase 1: Data Preparation
├── CPS_2023_Full → Extended_CPS_2023 (288MB)
└── Extended_CPS_2023 → Stratified_CPS_2023 (28MB, ~13k households)

Phase 2: Calibration Package
├── Sparse Matrix (~34k targets × ~5.7M household-CD pairs)
├── Target Groups & Initial Weights
└── Upload → GCS://policyengine-calibration/DATE/inputs/

Phase 3: Weight Optimization (L0 Calibration)
├── Local: 100 epochs (testing) → ~0% sparsity
└── GCP: 4000 epochs (production) → ~87% sparsity

Phase 4: State Dataset Creation
├── Apply weights to stratified dataset
├── Create 51 state files + optional combined file
└── Upload → GCS & Hugging Face
```

## Conceptual Framework

### The Geo-Stacking Approach

The same household dataset is treated as existing in multiple geographic areas simultaneously, creating an "empirical superpopulation" where each household can represent itself in different locations with different weights.

**Matrix Structure:**
- **Rows = Targets** (calibration constraints)
- **Columns = Households × Geographic Areas**

This creates a "small n, large p" problem where household weights are the parameters we estimate.

**Sparsity Pattern Example (2 states):**
```
                     H1_CA  H2_CA  H3_CA  H1_TX  H2_TX  H3_TX
national_employment    X      X      X      X      X      X
CA_age_0_5            X      X      X      0      0      0
TX_age_0_5            0      0      0      X      X      X
```

### Hierarchical Target Selection

For each target concept:
1. If CD-level target exists → use it for that CD only
2. If no CD target but state target exists → use state target for all CDs in that state
3. If neither exists → use national target

For administrative data (SNAP, Medicaid), always prefer admin over survey data.

## Target Groups

Targets are grouped to ensure balanced optimization:

| Group Type | Count | Description |
|------------|-------|-------------|
| National | 30 | Hardcoded US-level targets (each singleton) |
| Age | 7,848 | 18 bins × 436 CDs |
| AGI Distribution | 3,924 | 9 brackets × 436 CDs |
| SNAP Household | 436 | CD-level counts |
| SNAP Cost | 51 | State-level administrative |
| Medicaid | 436 | CD-level enrollment |
| EITC | 1,744 | 4 categories × 436 CDs |
| IRS SOI | ~25k | Various tax variables by CD |

## Key Technical Details

### L0 Regularization

Creates truly sparse weights through stochastic gates:
- Gate formula: `gate = sigmoid(log_alpha/beta) * (zeta - gamma) + gamma`
- With default parameters, gates create exact zeros even with `lambda_l0=0`
- Production runs achieve ~87% sparsity (725k active from 5.7M weights)

### Relative Loss Function

Using `((y - y_pred) / (y + 1))^2`:
- Handles massive scale disparities (targets range from 178K to 385B)
- 10% error on $1B target = same penalty as 10% error on $100K target

### ID Allocation System

Each CD gets a 10,000 ID range to prevent collisions:
- Household IDs: `CD_index × 10,000` to `CD_index × 10,000 + 9,999`
- Person IDs: Add 5M offset to avoid household collision
- Max safe: ~49k per CD to stay under int32 overflow

### State-Dependent Variables

SNAP and other state-dependent variables require special handling:
- Matrix construction pre-calculates values for each state
- H5 creation reindexes entity IDs (same household in different CDs needs unique IDs)
- ID reindexing changes `random()` seeds, causing ~10-15% variance in random-dependent variables
- End-to-end tests use **aggregate tolerance** (~15%) rather than exact matching

### Cache Clearing for State Swaps

When setting `state_fips` to recalculate state-dependent benefits, cached variables must be cleared. This is subtle:

**What to clear** (variables that need recalculation):
- Variables with `formulas` (traditional calculated variables)
- Variables with `adds` (sum of other variables, e.g., `snap_unearned_income`)
- Variables with `subtracts` (difference of variables)

**What NOT to clear** (structural data from H5):
- ID variables: `person_id`, `household_id`, `tax_unit_id`, `spm_unit_id`, `family_id`, `marital_unit_id`
- These have formulas that generate sequential IDs (0, 1, 2, ...), but we need the original H5 values

**Why IDs matter**: PolicyEngine's `random()` function uses entity IDs as deterministic seeds:
```python
seed = abs(entity_id * 100 + count_random_calls)
```
If IDs are regenerated, random-dependent variables produce different results. Three variables use `random()`:
- `meets_ssi_resource_test` (SSI eligibility)
- `is_wic_at_nutritional_risk` (WIC eligibility)
- `would_claim_wic` (WIC takeup)

**Implementation** in `calibration_utils.py` (single source of truth):
```python
def get_calculated_variables(sim):
    exclude_ids = {'person_id', 'household_id', 'tax_unit_id',
                   'spm_unit_id', 'family_id', 'marital_unit_id'}
    return [name for name, var in sim.tax_benefit_system.variables.items()
            if (var.formulas or getattr(var, 'adds', None) or getattr(var, 'subtracts', None))
            and name not in exclude_ids]
```

**Why same-state households also get set_input + cache clear**: The matrix builder always creates a fresh simulation, sets `state_fips`, and clears the cache—even when a household stays in its original state. This seems redundant but is intentional:

1. **Consistency**: All matrix values are computed the same way, regardless of whether state changes
2. **Deterministic random()**: The `random()` function's seed includes `count_random_calls`. Clearing the cache resets this counter to 0, ensuring reproducible results. Without cache clearing, different calculation histories produce different random outcomes.
3. **Verification**: Tests can verify matrix values by replicating this exact procedure. Comparing against the original simulation (without cache clear) would show ~10-15% mismatches due to different random() counter states—not bugs, just different calculation paths.

```
Question 1: How are SSI/WIC related to SNAP?
The connection is through income calculation chains:
snap
  └── snap_gross_income
        └── snap_unearned_income (uses `adds`)
              └── ssi (SSI benefit amount)
                    └── is_ssi_eligible
                          └── meets_ssi_resource_test
                                └── random()  ← stochastic eligibility

SSI (Supplemental Security Income) counts as unearned income for SNAP. So:
- random() determines if someone "passes" SSI's resource test (since CPS lacks actual asset data)
- This affects ssi benefit amount
- Which feeds into snap_unearned_income
- Which affects final snap calculation

WIC doesn't directly affect SNAP, but shares similar random-dependent eligibility logic (is_wic_at_nutritional_risk, would_claim_wic).
Question 2: Why still 13.5% mismatches if we preserved IDs?

The key is the full seed formula:
seed = abs(entity_id * 100 + count_random_calls)
                                                                                        
We preserved entity_id by excluding ID variables from clearing. But count_random_calls tracks how many times random() has been called for that entity during the simulation

When we:                                                                                                                                                                          
1. Create a fresh simulation
2. Set state_fips
3. Clear calculated variables
4. Call calculate("snap")                                                                                                                                                         
                                           
The calculation order may differ from the original simulation's calculation order. Different traversal paths through the variable dependency graph → different
count_random_calls when meets_ssi_resource_test is reached → different seed → different random result.
```

## File Reference

### Core Scripts
| Script | Purpose |
|--------|---------|
| `create_stratified_cps.py` | Income-based stratification sampling |
| `create_calibration_package.py` | Build optimization inputs |
| `optimize_weights.py` | L0 weight optimization |
| `create_sparse_cd_stacked.py` | Apply weights, create state files |
| `sparse_matrix_builder.py` | Build sparse target matrix |
| `calibration_utils.py` | Helper functions, CD mappings |

### Data Files
| File | Purpose |
|------|---------|
| `policy_data.db` | SQLite with all calibration targets |
| `stratified_extended_cps_2023.h5` | Input dataset (~13k households) |
| `calibration_package.pkl` | Sparse matrix & metadata |
| `w_cd.npy` | Final calibration weights |

### Batch Pipeline
| File | Purpose |
|------|---------|
| `batch_pipeline/Dockerfile` | CUDA + PyTorch container |
| `batch_pipeline/submit_batch_job.sh` | Build, push, submit to GCP |
| `batch_pipeline/config.env` | GCP settings |

## Validation

### Matrix Cell Lookup

Use `household_tracer.py` to navigate the matrix:

```python
from household_tracer import HouseholdTracer
tracer = HouseholdTracer(targets_df, matrix, household_mapping, cd_geoids, sim)

# Find where a household appears
positions = tracer.get_household_column_positions(household_id=565)

# Look up any cell
cell_info = tracer.lookup_matrix_cell(row_idx=10, col_idx=500)
```

### Key Validation Findings

1. **Tax Unit vs Household**: AGI constraints apply at tax unit level. A 5-person household with 3 people in a qualifying tax unit shows matrix value 3.0 (correct).

2. **Hierarchical Consistency**: Targets sum correctly from CD → State → National levels.

3. **SNAP Behavior**: May use reported values from dataset (not formulas), so state changes may not affect SNAP.

## Troubleshooting

### "CD exceeded 10k household allocation"
Weight vector has wrong dimensions or 0% sparsity. Check sparsity is ~87% for production.

### Memory Issues
- Local: Reduce batch size or use GCP
- State file creation: Use `--include-full-dataset` only with 32GB+ RAM

### GCP Job Fails
1. Check paths in `config.env`
2. Run `gcloud auth configure-docker`
3. Verify input file exists in GCS

## Known Issues

### CD-County Mappings
Only 10 CDs have real county proportions. Remaining CDs use state's most populous county. Fix requires Census geographic relationship files.

### Variables Excluded from Calibration
Certain high-error variables are excluded (rental income, various tax deductions). See `calibration_utils.py` for the full list.

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Stratified sampling | 93% size reduction while preserving income distribution |
| L0 regularization | Creates exact zeros for truly sparse weights |
| 10k ID ranges | Prevents int32 overflow in PolicyEngine |
| Group-wise loss | Prevents histogram variables from dominating |
| Relative loss | Handles 6 orders of magnitude in target scales |
