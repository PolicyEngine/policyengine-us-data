# PolicyEngine US Data Pipeline Documentation

## Overview

The PolicyEngine US data pipeline integrates Census surveys (CPS, ACS), IRS tax data (PUF, SOI), and Federal Reserve wealth data (SCF) to create a comprehensive microsimulation dataset. The pipeline produces three progressively enhanced dataset levels:
1. **CPS**: Base demographic layer from Census
2. **Extended CPS**: CPS + PUF-imputed financial variables
3. **Enhanced CPS**: Extended CPS + calibrated weights to match official statistics

## The Complete Pipeline Architecture

```bash
# Full pipeline in execution order
make download                # Download private IRS data from HuggingFace
make database                # Build calibration targets database
make data                    # Run complete pipeline:
  ├── python policyengine_us_data/utils/uprating.py
  ├── python policyengine_us_data/datasets/acs/acs.py
  ├── python policyengine_us_data/datasets/cps/cps.py
  ├── python policyengine_us_data/datasets/puf/irs_puf.py
  ├── python policyengine_us_data/datasets/puf/puf.py
  ├── python policyengine_us_data/datasets/cps/extended_cps.py
  ├── python policyengine_us_data/datasets/cps/enhanced_cps.py
  └── python policyengine_us_data/datasets/cps/small_enhanced_cps.py
make upload                  # Upload completed datasets to cloud storage
```

## Critical Pipeline Dependencies

### Hidden Dependencies

1. **PUF always requires CPS_2021**: The PUF generation hardcodes CPS_2021 for pension contribution imputation, regardless of target year. This creates a permanent dependency on 2021 data.

2. **PUF_2021 is the base for all future years**: Unlike going back to 2015, years 2022+ start from PUF_2021 and apply uprating. This makes PUF_2021 a critical checkpoint.

3. **Pre-trained models are cached**: SIPP tip model (tips.pkl) and SCF relationships are trained once and reused. These are not part of the main pipeline execution.

4. **Database targets are required for Enhanced CPS**: The calibration targets database must be populated before running Enhanced CPS generation.

## Private Data Management

### Download Prerequisites
The pipeline requires private IRS data downloaded from HuggingFace:
- `puf_2015.csv`: IRS Public Use File base data
- `demographics_2015.csv`: Demographic supplement
- `soi.csv`: Statistics of Income aggregates
- `np2023_d5_mid.csv`: Census population projections

Access controlled via `HUGGING_FACE_TOKEN` environment variable.

### Upload Distribution
Completed datasets are uploaded to:
- **HuggingFace**: Public access at `policyengine/policyengine-us-data`
- **Google Cloud Storage**: `policyengine-us-data` bucket

Uploaded files include:
- `enhanced_cps_2024.h5` (sparse version)
- `dense_enhanced_cps_2024.h5` (full weights)
- `small_enhanced_cps_2024.h5` (1,000 household sample)
- `pooled_3_year_cps_2023.h5` (combined 2021-2023)
- `policy_data.db` (calibration targets database)

## The Three-Stage Dataset Hierarchy

### Stage 1: CPS (Base Demographics)
**What it provides**:
- Household structure and demographics
- Basic income variables
- Geographic distribution
- Raw survey weights

**Transformations applied**:
1. Immigration status via ASEC-UA algorithm (targeting 13M undocumented)
2. Rent imputed from ACS-trained model
3. Tips from pre-trained SIPP model (loaded from tips.pkl)
4. Wealth/auto loans from SCF via QRF imputation

### Stage 2: Extended CPS (Financial Imputation)
**The Statistical Fusion Process**:
1. Train QRF models on PUF's 70+ financial variables
2. Learn relationships between demographics and finances
3. Apply patterns to CPS households
4. Result: CPS demographics + PUF-learned financial distributions

**Variables Imputed**:
- Income types: wages, capital gains, dividends, pensions
- Deductions: mortgage interest, charitable, state/local taxes
- Credits: EITC-relevant amounts, child care expenses
- Business income: partnership, S-corp, farm, rental

### Stage 3: Enhanced CPS (Calibrated Weights)
**The Calibration Process**:
Enhanced CPS reweights Extended CPS households to match official statistics through sophisticated optimization.

**Calibration Targets**:
- **IRS SOI Statistics**: Income distributions by AGI bracket, state, filing status
- **Hard-coded totals**: Medical expenses, child support, property tax, rent
- **National/State balance**: Separate normalization for national vs state targets

**Two Optimization Approaches**:

1. **Dense Optimization** (Standard gradient descent):
   - All households receive adjusted weights
   - Smooth weight distribution
   - Better for small-area estimates

2. **Sparse Optimization** (L0 regularization via HardConcrete gates):
   - Many households get zero weight
   - Fewer non-zero weights but higher values
   - More computationally efficient for large-scale simulations
   - Uses temperature and initialization parameters to control sparsity

The sparse version is the default distributed dataset, with dense available as `dense_enhanced_cps_2024.h5`.

## Dataset Variants

### Pooled CPS
Combines multiple years for increased sample size:
- **Pooled_3_Year_CPS_2023**: Merges CPS 2021, 2022, 2023
- Maintains year indicators for time-series analysis
- Larger sample for state-level estimates

### Small Enhanced CPS
Two reduction methods for development/testing:

1. **Random Sampling**: 1,000 households randomly selected
2. **Sparse Selection**: Uses L0 regularization results

Benefits:
- Fast iteration during development
- Unit testing microsimulation changes
- Reduced memory footprint (100MB vs 16GB)

## The Two-Phase Uprating System

### 2021 is a Methodology Boundary

The system uses completely different uprating approaches before and after 2021:

#### Phase 1: SOI Historical (2015 → 2021)
- Function: `uprate_puf()` in `datasets/puf/uprate_puf.py`
- Data source: IRS Statistics of Income actuals
- Method: Variable-specific growth from SOI aggregates
- Population adjustment: Divides by population growth for per-capita rates
- Special cases: Itemized deductions fixed at 2% annual growth

#### Phase 2: Parameter Projection (2021 → Future)  
- Function: `create_policyengine_uprating_factors_table()`
- Data source: PolicyEngine parameters (CBO, Census projections)
- Method: Indexed growth factors (2020 = 1.0)
- Coverage: 131+ variables with consistent methodology
- Any year >= 2021 can be generated this way

### Why This Matters

The 2021 boundary means:
- Historical accuracy for 2015-2021 using actual IRS data
- Forward flexibility for 2022+ using economic projections
- PUF_2021 must exist before creating any future year
- Changing pre-2021 methodology requires modifying SOI-based code

## How Data Sources Actually Connect

### ACS: Model Training Only
ACS_2022 doesn't contribute data to the final dataset. Instead:
- Trains a QRF model relating demographics to rent/property tax
- Model learns patterns like "income X in state Y → rent Z"
- These relationships apply across years (why 2022 works for 2023+)
- Located in `add_rent()` function in CPS generation

### CPS: The Demographic Foundation
Foundation for all subsequent processing with four imputation layers.

### PUF: Tax Detail Layer
**Critical Processing Steps**:
1. Uprating (two-phase system described above)
2. QBI simulation (W-2 wages, UBIA for Section 199A)
3. Demographics imputation for records missing age/gender
4. **Pension contributions learned from CPS_2021** (hardcoded dependency)

**The QBI Simulation**: Since PUF lacks Section 199A details, the system:
- Simulates W-2 wages paid by businesses
- Estimates unadjusted basis of qualified property
- Assigns SSTB (specified service trade or business) status
- Based on parameters in `qbi_assumptions.yaml`

## Technical Implementation Details

### Memory Management
- ExtendedCPS QRF imputation: ~16GB RAM peak
- Processing 70+ variables sequentially to manage memory
- Batch processing with configurable batch sizes
- HDF5 format for efficient storage/access

### Performance Optimization
- **Parallel processing**: Tool calls run concurrently where possible
- **Caching**: Pre-trained models cached to disk
- **Sparse storage**: Default distribution uses sparse weights
- **Incremental generation**: Can generate specific years without full rebuild

### Error Recovery
- **Checkpoint saves**: Each major stage saves to disk
- **Resumable pipeline**: Can restart from last successful stage
- **Validation checks**: After each stage to catch issues early
- **Fallback options**: Dense weights if sparse optimization fails

## CI/CD Integration

### GitHub Actions Workflow
Triggered on:
- Push to main branch
- Pull requests
- Manual dispatch

Pipeline stages:
1. **Lint**: Code quality checks
2. **Test**: 
   - Basic tests (every PR)
   - Full suite with data build (main branch only)
3. **Publish**: PyPI release on version bump

### Test Modes
- **Standard**: Unit tests only
- **Full Suite** (`full_suite: true`):
  - Downloads private data
  - Builds calibration database
  - Generates all datasets
  - Uploads to cloud storage

### Environment Requirements
- **Secrets**:
  - `HUGGING_FACE_TOKEN`: Private data access
  - `POLICYENGINE_US_DATA_GITHUB_TOKEN`: Cross-repo operations
- **GCP Authentication**: Workload identity for uploads
- **TEST_LITE**: Reduces processing for non-production runs

## Data Validation Checkpoints

### After CPS Generation
- Immigration status populations (13M undocumented target)
- Household structure integrity
- Geographic distribution
- Weight normalization

### After PUF Processing  
- QBI component reasonableness
- Pension contribution distributions
- Demographic completeness
- Tax variable consistency

### After Extended CPS
- Financial variable distributions vs PUF
- Preservation of CPS demographics
- Total income aggregates
- Imputation quality metrics

### After Enhanced CPS
- Target achievement rates (>95% for key variables)
- Weight distribution statistics
- State-level calibration quality
- Sparsity metrics (for sparse version)

## Creating Datasets for Arbitrary Years

### Creating Any Year >= 2021

You can create any year >= 2021 by defining a class:

```python
class PUF_2023(PUF):
    name = "puf_2023"
    time_period = 2023
    file_path = STORAGE_FOLDER / "puf_2023.h5"

PUF_2023().generate()  # Automatically uprates from PUF_2021
```

### Why Only 2015, 2021, 2024 Are Pre-Built

- **2015**: IRS PUF base year (original data)
- **2021**: Methodology pivot + calibration year
- **2024**: Current year for policy analysis

The infrastructure supports any year 2021-2034 (extent of uprating parameters).

### The Cascade Effect

Creating ExtendedCPS_2023 requires:
1. CPS_2023 (or uprated from CPS_2023 if no raw data)
2. PUF_2023 (uprated from PUF_2021)
3. ACS_2022 (already suitable, relationships stable)
4. SCF_2022 (wealth patterns applicable)

Creating EnhancedCPS_2023 additionally requires:
5. ExtendedCPS_2023 (from above)
6. Calibration targets database (SOI + other sources)

## Understanding the Web of Dependencies

```
uprating_factors.csv ──────────────────┐
                                       ↓
ACS_2022 → [rent model] ────────→ CPS_2023 → ExtendedCPS_2023 → EnhancedCPS_2023
                                      ↑              ↑                  ↑
CPS_2021 → [pension model] ──────────┘              │                  │
    ↓                                                │                  │
PUF_2015 → PUF_2021 → PUF_2023 ─────────────────────┘                  │
             ↑                                                          │
        [SOI data]                                                      │
                                                                        │
calibration_targets.db ─────────────────────────────────────────────────┘
```

This web means:
- Can't generate PUF without CPS_2021 existing
- Can't generate ExtendedCPS without both CPS and PUF
- Can't generate EnhancedCPS without ExtendedCPS and targets database
- Can't uprate PUF_2022+ without PUF_2021
- But CAN reuse ACS_2022 for multiple years

## Reproducibility Considerations

### Ensuring Consistent Results
- **Random seeds**: Set via `set_seeds()` function
- **Model versioning**: Pre-trained models include version tags
- **Parameter freezing**: Uprating factors fixed at generation time
- **Data hashing**: Input files verified via checksums

### Sources of Variation
- **Optimization convergence**: Different hardware may converge differently
- **Floating point precision**: GPU vs CPU differences
- **Library versions**: Especially torch, scikit-learn
- **Calibration targets**: Updates to SOI data affect results

## Glossary

- **QRF**: Quantile Random Forest - preserves distributions during imputation
- **SOI**: Statistics of Income - IRS published aggregates
- **QBI**: Qualified Business Income (Section 199A deduction)
- **UBIA**: Unadjusted Basis Immediately After Acquisition
- **SSTB**: Specified Service Trade or Business
- **ASEC-UA**: Algorithm for imputing undocumented status in CPS
- **HardConcrete**: Differentiable gate for L0 regularization
- **L0 Regularization**: Penalty on number of non-zero weights
- **Dense weights**: All households have positive weights
- **Sparse weights**: Many households have zero weight