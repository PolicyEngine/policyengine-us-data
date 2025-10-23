# Congressional District Geo-Stacking Calibration Pipeline

## Executive Summary

This pipeline creates state-level microsimulation datasets with Congressional District (CD) level calibration weights. It takes the Current Population Survey (CPS) data, enriches it with Public Use File (PUF) income variables, applies L0 sparse calibration to match 34,089 demographic and economic targets across 436 Congressional Districts, and produces optimized datasets for each US state.

**Key Achievement**: Reduces ~200k household dataset to ~13k households while maintaining statistical representativeness across all 436 CDs through sophisticated weight calibration.

## Prerequisites

### Required Software
- Python 3.9+ with `policyengine-us` environment
- Google Cloud SDK (`gcloud`, `gsutil`)
- Docker (for GCP batch jobs)
- CUDA-capable GPU (optional, for local GPU runs)
- Make

### Required Python Packages
```bash
pip install policyengine-us policyengine-us-data
pip install torch scipy h5py sqlalchemy pandas numpy
# L0 package should be available in ~/devl/L0 or installed separately
```

### GCP Credentials
```bash
# Authenticate for GCP
gcloud auth login
gcloud auth configure-docker

# Set project (if not default)
gcloud config set project policyengine-research
```

### Environment Setup
```bash
# From repo root
cd policyengine_us_data/datasets/cps/geo_stacking_calibration/

# For GCP batch jobs, check config
cat batch_pipeline/config.env
```

## Quick Start

### Complete Pipeline (Local + GCP)
```bash
# 1. Generate base datasets
make data-geo

# 2. Create and upload calibration package
make upload-calibration-package
# Note the date prefix shown (e.g., 2025-10-22-1721)

# 3. Update GCP config with the date prefix
# Edit batch_pipeline/config.env:
#   INPUT_PATH=2025-10-22-1721/inputs
#   OUTPUT_PATH=2025-10-22-1721/outputs

# 4. Run optimization on GCP (4000 epochs)
make optimize-weights-gcp
# Monitor with: ./batch_pipeline/monitor_batch_job.sh <job-name>

# 5. Download optimized weights
make download-weights-from-gcs
# Enter the date prefix when prompted

# 6. Create state datasets
make create-state-files

# 7. Upload to GCS
make upload-state-files-to-gcs
```

### Local Testing Only (100 epochs)
```bash
make data-geo
make calibration-package
make optimize-weights-local  # CPU/GPU local, 100 epochs only
make create-state-files
```

## Pipeline Architecture

```
Phase 1: Data Preparation
├── CPS_2023_Full → Extended_CPS_2023 (288MB)
└── Extended_CPS_2023 → Stratified_CPS_2023 (28MB, ~13k households)

Phase 2: Calibration Package
├── Sparse Matrix (24,484 targets × 5.7M household-CD pairs)
├── Target Groups & Initial Weights
└── Upload → GCS://policyengine-calibration/DATE/inputs/

Phase 3: Weight Optimization (L0 Calibration)
├── Local: 100 epochs (testing) → ~0% sparsity
└── GCP: 4000 epochs (production) → ~87% sparsity

Phase 4: State Dataset Creation
├── Apply weights to stratified dataset
├── Create 51 state files + 1 combined file
└── Upload → GCS & Hugging Face
```

## Detailed Pipeline Phases

### Phase 1: Data Preparation

**Purpose**: Create a stratified sample that maintains income distribution while reducing computational load.

**Makefile Target**: `make data-geo`

**Key Scripts**:
- `policyengine_us_data/datasets/cps/cps.py` - Generates CPS_2023_Full when `GEO_STACKING=true`
- `policyengine_us_data/datasets/puf/puf.py` - Generates PUF_2023 when `GEO_STACKING=true`
- `policyengine_us_data/datasets/cps/extended_cps.py` - Imputes PUF variables when `GEO_STACKING_MODE=true`
- `create_stratified_cps.py` - Creates stratified sample

**Outputs**:
- `policyengine_us_data/storage/extended_cps_2023.h5` (288MB, ~200k households)
- `policyengine_us_data/storage/stratified_extended_cps_2023.h5` (28MB, ~13k households)

**Stratification Strategy**:
- Keeps ALL top 1% income households
- Progressively samples lower income strata
- Target: 10,000 total households (actually gets ~13k)

### Phase 2: Calibration Package Creation

**Purpose**: Build sparse matrix and prepare optimization inputs.

**Makefile Targets**:
- `make calibration-package` (local only)
- `make upload-calibration-package` (local + GCS upload)

**Key Script**: `create_calibration_package.py`

**Arguments**:
```bash
--db-path policyengine_us_data/storage/policy_data.db
--dataset-uri policyengine_us_data/storage/stratified_extended_cps_2023.h5
--mode Stratified  # Options: Test, Stratified, Full
--gcs-bucket policyengine-calibration  # For upload
--gcs-date 2025-10-22-1721  # Auto-generated timestamp
```

**Outputs**:
- Local: `policyengine_us_data/storage/calibration/calibration_package.pkl` (1.2GB)
- GCS: `gs://policyengine-calibration/DATE/inputs/calibration_package.pkl`

**Package Contents**:
- `X_sparse`: Sparse matrix (24,484 targets × 5,706,804 household-CD pairs)
- `targets_df`: Target values from database
- `initial_weights`: Starting weights per household-CD
- `keep_probs`: Sampling probabilities for L0
- `household_id_mapping`: Original household IDs
- `target_groups`: Grouping for hierarchical calibration

### Phase 3: Weight Optimization

**Purpose**: Find optimal weights that minimize prediction error while maintaining sparsity.

**Makefile Targets**:
- `make optimize-weights-local` - Quick test, 100 epochs, CPU
- `make optimize-weights-gcp` - Production, 4000 epochs, GPU

**Key Scripts**:
- Local: `optimize_weights.py`
- GCP: `batch_pipeline/optimize_weights.py`

**Configuration** (`batch_pipeline/config.env`):
```env
TOTAL_EPOCHS=4000
BETA=0.35          # L0 temperature parameter
LAMBDA_L0=5e-7     # L0 sparsity regularization
LAMBDA_L2=5e-9     # L2 weight regularization
LR=0.1             # Learning rate
GPU_TYPE=nvidia-tesla-p100
```

**Outputs**:
- `w_cd.npy` - Canonical weights file (22MB)
- `w_cd_TIMESTAMP.npy` - Timestamped backup
- `cd_sparsity_history_TIMESTAMP.csv` - Sparsity progression

**Expected Results**:
- 100 epochs: ~0% sparsity (all weights active)
- 4000 epochs: ~87% sparsity (~725k active from 5.7M)

### Phase 4: State Dataset Creation

**Purpose**: Apply calibrated weights to create state-level datasets.

**Makefile Target**: `make create-state-files`

**Key Script**: `create_sparse_cd_stacked.py`

**How to Run Directly** (with Python module syntax):
```bash
python -m policyengine_us_data.datasets.cps.geo_stacking_calibration.create_sparse_cd_stacked \
  --weights-path policyengine_us_data/storage/calibration/w_cd.npy \
  --dataset-path policyengine_us_data/storage/stratified_extended_cps_2023.h5 \
  --db-path policyengine_us_data/storage/policy_data.db \
  --output-dir policyengine_us_data/storage/cd_states
```

**Outputs** (in `policyengine_us_data/storage/cd_states/`):
- 51 state files: `AL.h5`, `AK.h5`, ..., `WY.h5`
- 1 combined file: `cd_calibration.h5`
- Mapping CSVs: `STATE_household_mapping.csv` for tracing

**Processing Details**:
- Filters households by non-zero weights per CD
- Reindexes IDs using 10k ranges per CD to avoid overflow
- Updates geographic variables (state, CD, county)
- Preserves household structure (tax units, SPM units)

## File Reference

### Configuration Files
| File | Purpose |
|------|---------|
| `batch_pipeline/config.env` | GCP batch job settings |
| `cd_county_mappings.json` | CD to county proportion mappings |
| `Makefile` | All pipeline targets (lines 78-142) |

### Core Scripts
| Script | Purpose |
|--------|---------|
| `create_stratified_cps.py` | Income-based stratification sampling |
| `create_calibration_package.py` | Build optimization inputs |
| `optimize_weights.py` | L0 weight optimization |
| `create_sparse_cd_stacked.py` | Apply weights, create state files |
| `metrics_matrix_geo_stacking_sparse.py` | Build sparse target matrix |
| `calibration_utils.py` | Helper functions, CD mappings |

### Database & Data
| File | Purpose |
|------|---------|
| `policy_data.db` | SQLite with all calibration targets |
| `stratified_extended_cps_2023.h5` | Input dataset (~13k households) |
| `calibration_package.pkl` | Sparse matrix & metadata |
| `w_cd.npy` | Final calibration weights |

### Batch Pipeline Files
| File | Purpose |
|------|---------|
| `batch_pipeline/Dockerfile` | CUDA + PyTorch container |
| `batch_pipeline/submit_batch_job.sh` | Build, push, submit to GCP |
| `batch_pipeline/monitor_batch_job.sh` | Track job progress |
| `batch_pipeline/run_batch_job.sh` | Runs inside container |

## Environment Variables

### For Data Generation
- `GEO_STACKING=true` - Generate geographic-specific CPS/PUF files
- `GEO_STACKING_MODE=true` - Enable extended CPS creation
- `TEST_LITE=true` - Use smaller test datasets (optional)

### For GCP Batch
Set in `batch_pipeline/config.env`:
- `PROJECT_ID` - GCP project
- `BUCKET_NAME` - GCS bucket (policyengine-calibration)
- `INPUT_PATH` - Input location in bucket
- `OUTPUT_PATH` - Output location in bucket
- `TOTAL_EPOCHS` - Training iterations
- `GPU_TYPE` - nvidia-tesla-p100

## Common Operations

### Check Dataset Dimensions
```python
import h5py
import numpy as np

with h5py.File('policyengine_us_data/storage/stratified_extended_cps_2023.h5', 'r') as f:
    households = f['household_id']['2023'][:]
    print(f"Households: {len(np.unique(households)):,}")
```

### Verify Weight Sparsity
```python
import numpy as np
w = np.load('policyengine_us_data/storage/calibration/w_cd.npy')
sparsity = 100 * (1 - np.sum(w > 0) / w.shape[0])
print(f"Sparsity: {sparsity:.2f}%")
print(f"Active weights: {np.sum(w > 0):,} of {w.shape[0]:,}")
```

### Monitor GCP Job
```bash
# Get job status
gcloud batch jobs describe <job-name> --location=us-central1

# Stream logs
gcloud logging read "resource.type=batch.googleapis.com/Job AND resource.labels.job_id=<job-name>" --limit=50

# Or use helper script
./batch_pipeline/monitor_batch_job.sh <job-name>
```

### Upload to Hugging Face
```bash
# Automatic on push to main via GitHub Actions
# Manual upload:
python policyengine_us_data/storage/upload_completed_datasets.py
```

## Troubleshooting

### "CD exceeded 10k household allocation"
**Problem**: Weight vector has wrong dimensions or 0% sparsity.
**Solution**:
1. Check weight sparsity (should be ~87% for production)
2. Re-download from GCS: `make download-weights-from-gcs`
3. Delete old w_cd.npy before downloading

### "FileNotFoundError" when running create_sparse_cd_stacked.py
**Problem**: Relative paths don't resolve with module imports.
**Solution**: Use `-m` flag:
```bash
python -m policyengine_us_data.datasets.cps.geo_stacking_calibration.create_sparse_cd_stacked
```

### "cd_county_mappings.json not found"
**Problem**: Script looking in wrong directory.
**Solution**: Already fixed in code to use script's parent directory. Warning is non-fatal.

### GCP Job Fails
**Common Causes**:
1. Wrong paths in config.env
2. Docker authentication: `gcloud auth configure-docker`
3. Insufficient GPU quota
4. Input file not in GCS

### Memory Issues
**For local runs**: Reduce batch size or use GCP
**For GCP**: Increase `MEMORY_MIB` in config.env (default: 32768)

## Architecture Decisions

### Why Stratified Sampling?
- Full extended CPS: ~200k households × 436 CDs = 87M pairs
- Stratified: ~13k households × 436 CDs = 5.7M pairs (93% reduction)
- Preserves income distribution critical for tax policy analysis

### Why L0 Regularization?
- Creates truly sparse weights (exact zeros, not near-zeros)
- Reduces storage and computation for production use
- 87% sparsity = only 725k active weights from 5.7M

### Why 10k ID Ranges per CD?
- Prevents int32 overflow when IDs multiplied by 100
- Allows unique identification across geographic stacking
- Simple mapping: CD index × 10,000

### Why Separate Package Creation?
- Calibration package (1.2GB) created once, used many times
- Allows experimentation with optimization parameters
- Enables GCP/local switching without regenerating data

## Future Improvements

### High Priority
1. **Fix CD-County Mappings** (PROJECT_STATUS.md:256-271)
   - Currently uses crude state-level defaults
   - Should use Census geographic relationship files
   - Only 10 CDs have accurate county proportions

2. **Automate GCS Path Updates**
   - Currently manual edit of config.env
   - Could parse from upload output

### Medium Priority
1. **Add validation checks**
   - Verify targets sum correctly across hierarchies
   - Check weight convergence metrics
   - Validate geographic assignments

2. **Optimize memory usage**
   - Stream processing for large states
   - Chunked matrix operations

3. **Add resume capability**
   - Save checkpoint weights during optimization
   - Allow restart from epoch N

### Low Priority
1. **Parallelize state file creation**
   - Currently sequential (takes ~1 hour)
   - Could process states in parallel

2. **Add data lineage tracking**
   - Version control for calibration runs
   - Metadata for reproducibility

## Support Files

- `PROJECT_STATUS.md` - Detailed project history and issues
- `GEO_STACKING_TECHNICAL.md` - Deep technical documentation
- `README.md` - Quick overview

## Contact

For questions about:
- Pipeline operations: Check this document first
- Technical details: See GEO_STACKING_TECHNICAL.md
- Known issues: See PROJECT_STATUS.md
- L0 package: Check ~/devl/L0/README.md