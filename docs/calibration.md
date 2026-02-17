# Calibration Pipeline User's Manual

The unified calibration pipeline reweights cloned CPS records to match administrative targets using L0-regularized optimization. This guide covers the three main workflows: full pipeline, build-then-fit, and fitting from a saved package.

## Quick Start

```bash
# Full pipeline (build matrix + fit weights):
make calibrate

# Build matrix only (save package for later fitting):
make calibrate-build
```

## Architecture Overview

The pipeline has two expensive phases:

1. **Matrix build** (~30 min with PUF): Clone CPS records, assign geography, optionally PUF-impute, compute all target variable values, assemble a sparse calibration matrix.
2. **Weight fitting** (~5-20 min on GPU): L0-regularized optimization to find household weights that reproduce administrative targets.

The calibration package checkpoint lets you run phase 1 once and iterate on phase 2 with different hyperparameters or target selections---without rebuilding.

## Workflows

### 1. Single-pass (default)

Build the matrix and fit weights in one run:

```bash
python -m policyengine_us_data.calibration.unified_calibration \
  --puf-dataset policyengine_us_data/storage/puf_2024.h5 \
  --target-config policyengine_us_data/calibration/target_config.yaml \
  --epochs 200 \
  --device cuda
```

Output:
- `storage/calibration/unified_weights.npy` --- calibrated weight vector
- `storage/calibration/unified_diagnostics.csv` --- per-target error report
- `storage/calibration/unified_run_config.json` --- full run configuration

### 2. Build-then-fit (recommended for iteration)

**Step 1: Build the matrix and save a package.**

```bash
python -m policyengine_us_data.calibration.unified_calibration \
  --puf-dataset policyengine_us_data/storage/puf_2024.h5 \
  --target-config policyengine_us_data/calibration/target_config.yaml \
  --build-only
```

This saves `storage/calibration/calibration_package.pkl` (default location). Use `--package-output` to specify a different path.

**Step 2: Fit weights from the package (fast, repeatable).**

```bash
python -m policyengine_us_data.calibration.unified_calibration \
  --package-path storage/calibration/calibration_package.pkl \
  --epochs 500 \
  --lambda-l0 1e-8 \
  --beta 0.65 \
  --lambda-l2 1e-8 \
  --device cuda
```

You can re-run Step 2 as many times as you want with different hyperparameters. The expensive matrix build only happens once.

### 3. Re-filtering a saved package

A saved package contains **all** targets from the database (before target config filtering). You can apply a different target config at fit time:

```bash
python -m policyengine_us_data.calibration.unified_calibration \
  --package-path storage/calibration/calibration_package.pkl \
  --target-config my_custom_config.yaml \
  --epochs 200
```

This lets you experiment with which targets to include without rebuilding the matrix.

### 4. Running on Modal (GPU cloud)

```bash
modal run modal_app/remote_calibration_runner.py \
  --branch puf-impute-fix-530 \
  --gpu A10 \
  --epochs 500 \
  --target-config policyengine_us_data/calibration/target_config.yaml \
  --beta 0.65
```

The target config YAML is read from the cloned repo inside the container, so it must be committed to the branch you specify.

### 5. Portable fitting (Kaggle, Colab, etc.)

Transfer the package file to any environment with `scipy`, `numpy`, `pandas`, `torch`, and `l0-python` installed:

```python
from policyengine_us_data.calibration.unified_calibration import (
    load_calibration_package,
    apply_target_config,
    fit_l0_weights,
)

package = load_calibration_package("calibration_package.pkl")
targets_df = package["targets_df"]
X_sparse = package["X_sparse"]

weights = fit_l0_weights(
    X_sparse=X_sparse,
    targets=targets_df["value"].values,
    lambda_l0=1e-8,
    epochs=500,
    device="cuda",
    beta=0.65,
    lambda_l2=1e-8,
)
```

## Target Config

The target config controls which targets reach the optimizer. It uses a YAML exclusion list:

```yaml
exclude:
  - variable: rent
    geo_level: national
  - variable: eitc
    geo_level: district
  - variable: snap
    geo_level: state
    domain_variable: snap   # optional: further narrow the match
```

Each rule drops rows from the calibration matrix where **all** specified fields match. Unrecognized variables silently match nothing.

### Fields

| Field | Required | Values | Description |
|---|---|---|---|
| `variable` | Yes | Any variable name in `target_overview` | The calibration target variable |
| `geo_level` | Yes | `national`, `state`, `district` | Geographic aggregation level |
| `domain_variable` | No | Any domain variable in `target_overview` | Narrows match to a specific domain |

### Default config

The checked-in config at `policyengine_us_data/calibration/target_config.yaml` reproduces the junkyard notebook's 22 excluded target groups. It drops:

- **13 national-level variables**: alimony, charitable deduction, child support, interest deduction, medical expense deduction, net worth, person count, real estate taxes, rent, social security dependents/survivors
- **9 district-level variables**: ACA PTC, EITC, income tax before credits, medical expense deduction, net capital gains, rental income, tax unit count, partnership/S-corp income, taxable social security

Applying this config reduces targets from ~37K to ~21K, matching the junkyard's target selection.

### Writing a custom config

To experiment, copy the default and edit:

```bash
cp policyengine_us_data/calibration/target_config.yaml my_config.yaml
# Edit my_config.yaml to add/remove exclusion rules
python -m policyengine_us_data.calibration.unified_calibration \
  --package-path storage/calibration/calibration_package.pkl \
  --target-config my_config.yaml \
  --epochs 200
```

To see what variables and geo_levels are available in the database:

```sql
SELECT DISTINCT variable, geo_level
FROM target_overview
ORDER BY variable, geo_level;
```

## CLI Reference

### Core flags

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `storage/stratified_extended_cps_2024.h5` | Path to CPS h5 file |
| `--db-path` | `storage/calibration/policy_data.db` | Path to target database |
| `--output` | `storage/calibration/unified_weights.npy` | Weight output path |
| `--puf-dataset` | None | Path to PUF h5 (enables PUF cloning) |
| `--preset` | `local` | L0 preset: `local` (1e-8) or `national` (1e-4) |
| `--lambda-l0` | None | Custom L0 penalty (overrides `--preset`) |
| `--epochs` | 100 | Training epochs |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--n-clones` | 10 | Number of dataset clones |
| `--seed` | 42 | Random seed for geography assignment |

### Target selection

| Flag | Default | Description |
|---|---|---|
| `--target-config` | None | Path to YAML exclusion config |
| `--domain-variables` | None | Comma-separated domain filter (SQL-level) |
| `--hierarchical-domains` | None | Domains for hierarchical uprating |

### Checkpoint flags

| Flag | Default | Description |
|---|---|---|
| `--build-only` | False | Build matrix, save package, skip fitting |
| `--package-path` | None | Load pre-built package (skip matrix build) |
| `--package-output` | Auto (when `--build-only`) | Where to save package |

### Hyperparameter flags

| Flag | Default | Junkyard value | Description |
|---|---|---|---|
| `--beta` | 0.35 | 0.65 | L0 gate temperature (higher = softer gates) |
| `--lambda-l2` | 1e-12 | 1e-8 | L2 regularization on weights |
| `--learning-rate` | 0.15 | 0.15 | Optimizer learning rate |

### Skip flags

| Flag | Description |
|---|---|
| `--skip-puf` | Skip PUF clone + QRF imputation |
| `--skip-source-impute` | Skip ACS/SIPP/SCF re-imputation |
| `--skip-takeup-rerandomize` | Skip takeup re-randomization |

## Calibration Package Format

The package is a pickled Python dict:

```python
{
    "X_sparse": scipy.sparse.csr_matrix,  # (n_targets, n_records)
    "targets_df": pd.DataFrame,           # target metadata + values
    "target_names": list[str],            # human-readable names
    "metadata": {
        "dataset_path": str,
        "db_path": str,
        "n_clones": int,
        "n_records": int,
        "seed": int,
        "created_at": str,       # ISO timestamp
        "target_config": dict,   # config used at build time
    },
}
```

The `targets_df` DataFrame has columns: `variable`, `geo_level`, `geographic_id`, `domain_variable`, `value`, and others from the database.

## Hyperparameter Tuning Guide

The three key hyperparameters control the tradeoff between target accuracy and sparsity:

- **`beta`** (L0 gate temperature): Controls how sharply the L0 gates open/close. Higher values (0.5--0.8) give softer decisions and more exploration early in training. Lower values (0.2--0.4) give harder on/off decisions.

- **`lambda_l0`** (via `--preset` or `--lambda-l0`): Controls how many records survive. `1e-8` (local preset) keeps millions of records for local-area analysis. `1e-4` (national preset) keeps ~50K for the web app.

- **`lambda_l2`**: Regularizes weight magnitudes. Larger values (1e-8) prevent any single record from having extreme weight. Smaller values (1e-12) allow more weight concentration.

### Suggested starting points

For **local-area calibration** (millions of records):
```bash
--lambda-l0 1e-8 --beta 0.65 --lambda-l2 1e-8 --epochs 500
```

For **national web app** (~50K records):
```bash
--lambda-l0 1e-4 --beta 0.35 --lambda-l2 1e-12 --epochs 200
```

## Makefile Targets

| Target | Description |
|---|---|
| `make calibrate` | Full pipeline with PUF and target config |
| `make calibrate-build` | Build-only mode (saves package, no fitting) |
