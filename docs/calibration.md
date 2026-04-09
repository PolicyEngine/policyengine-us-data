# Calibration Pipeline User's Manual

The unified calibration pipeline reweights cloned CPS records to match administrative targets using
L0-regularized optimization. This guide covers the main workflows: lightweight build-then-fit, full
pipeline with PUF, and fitting from a saved package.

This is the current production calibration path. The older national-only Enhanced CPS path
(`make data-legacy`) remains in the repo for legacy reproduction and uses a separate `EnhancedCPS` /
`build_loss_matrix()` flow.

## Quick Start

```bash
# Build matrix only from stratified CPS (no PUF, no re-imputation):
python -m policyengine_us_data.calibration.unified_calibration \
  --target-config policyengine_us_data/calibration/target_config.yaml \
  --skip-source-impute \
  --skip-takeup-rerandomize \
  --build-only

# Fit weights from a saved package:
python -m policyengine_us_data.calibration.unified_calibration \
  --package-path storage/calibration/calibration_package.pkl \
  --epochs 500 --device cuda

# Full pipeline with PUF (build + fit in one shot):
make calibrate
```

## Architecture Overview

The pipeline has two phases:

1. **Matrix build**: Clone CPS records, assign geography, compute all target variable values,
   assemble a sparse calibration matrix. Optionally includes PUF cloning (doubles record count) and
   source re-imputation.
1. **Weight fitting** (~5-20 min on GPU): L0-regularized optimization to find household weights that
   reproduce administrative targets.

The calibration package checkpoint lets you run phase 1 once and iterate on phase 2 with different
hyperparameters or target selections---without rebuilding.

### Prerequisites

The matrix build requires two inputs from the data pipeline:

- **Stratified CPS** (`storage/stratified_extended_cps_2025.h5`): ~12K households, built by
  `make data`. This is the base dataset that gets cloned.
- **Target database** (`storage/calibration/policy_data.db`): Administrative targets, built by
  `make database`.

Both must exist before running calibration. The stratified CPS already contains all CPS variables
needed for calibration; PUF cloning and source re-imputation are optional enhancements that happen
at calibration time.

## Workflows

### 1. Lightweight build-then-fit (recommended for iteration)

Build the matrix from the stratified CPS without PUF cloning or re-imputation. This is the fastest
way to get a calibration package for experimentation.

**Step 1: Build the matrix (~12K base records x 430 clones = ~5.2M columns).**

```bash
python -m policyengine_us_data.calibration.unified_calibration \
  --target-config policyengine_us_data/calibration/target_config.yaml \
  --skip-source-impute \
  --skip-takeup-rerandomize \
  --build-only
```

This saves `storage/calibration/calibration_package.pkl` (default location). Use `--package-output`
to specify a different path.

**Step 2: Fit weights from the package (fast, repeatable).**

```bash
python -m policyengine_us_data.calibration.unified_calibration \
  --package-path storage/calibration/calibration_package.pkl \
  --epochs 1000 \
  --lambda-l0 1e-8 \
  --beta 0.65 \
  --lambda-l2 1e-8 \
  --device cuda
```

You can re-run Step 2 as many times as you want with different hyperparameters. The expensive matrix
build only happens once.

### 2. Full pipeline with PUF

Adding `--puf-dataset` doubles the record count (~24K base records x 430 clones = ~10.3M columns) by
creating PUF-imputed copies of every CPS record. This also triggers source re-imputation unless
skipped.

**Single-pass (build + fit):**

```bash
python -m policyengine_us_data.calibration.unified_calibration \
  --puf-dataset policyengine_us_data/storage/puf_2024.h5 \
  --target-config policyengine_us_data/calibration/target_config.yaml \
  --epochs 200 \
  --device cuda
```

Or equivalently: `make calibrate`

Output:

- `storage/calibration/calibration_weights.npy` --- calibrated weight vector
- `storage/calibration/unified_diagnostics.csv` --- per-target error report
- `storage/calibration/unified_run_config.json` --- full run configuration

**Build-only (save package for later fitting):**

```bash
python -m policyengine_us_data.calibration.unified_calibration \
  --puf-dataset policyengine_us_data/storage/puf_2024.h5 \
  --target-config policyengine_us_data/calibration/target_config.yaml \
  --build-only
```

Or equivalently: `make calibrate-build`

This saves `storage/calibration/calibration_package.pkl` (default location). Use `--package-output`
to specify a different path.

Then fit from the package using the same Step 2 command from Workflow 1.

### 3. Re-filtering a saved package

A saved package contains **all** targets from the database (before target config filtering). You can
apply a different target config at fit time:

```bash
python -m policyengine_us_data.calibration.unified_calibration \
  --package-path storage/calibration/calibration_package.pkl \
  --target-config my_custom_config.yaml \
  --epochs 200
```

This lets you experiment with which targets to include without rebuilding the matrix.

### 4. Running on Modal (GPU cloud)

**From a pre-built package** (recommended):

Use `--package-path` to point at a local `.pkl` file. The runner automatically uploads it to the
Modal Volume and then fits from it on the GPU, avoiding the function argument size limit.

```bash
modal run modal_app/remote_calibration_runner.py \
  --package-path policyengine_us_data/storage/calibration/calibration_package.pkl \
  --branch calibration-pipeline-improvements \
  --gpu T4 \
  --epochs 1000 \
  --beta 0.65 \
  --lambda-l0 1e-8 \
  --lambda-l2 1e-8
```

If a package already exists on the volume from a previous upload, you can also use
`--prebuilt-matrices` to fit directly without re-uploading.

**Full pipeline** (builds matrix from scratch on Modal):

```bash
modal run modal_app/remote_calibration_runner.py \
  --branch calibration-pipeline-improvements \
  --gpu T4 \
  --epochs 1000 \
  --beta 0.65 \
  --lambda-l0 1e-8 \
  --lambda-l2 1e-8 \
  --target-config policyengine_us_data/calibration/target_config.yaml
```

The target config YAML is read from the cloned repo inside the container, so it must be committed to
the branch you specify.

### 5. Portable fitting (Kaggle, Colab, etc.)

Transfer the package file to any environment with `scipy`, `numpy`, `pandas`, `torch`, and
`l0-python` installed:

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

Each rule drops rows from the calibration matrix where **all** specified fields match. Unrecognized
variables silently match nothing.

### Fields

| Field             | Required | Values                                   | Description                        |
| ----------------- | -------- | ---------------------------------------- | ---------------------------------- |
| `variable`        | Yes      | Any variable name in `target_overview`   | The calibration target variable    |
| `geo_level`       | Yes      | `national`, `state`, `district`          | Geographic aggregation level       |
| `domain_variable` | No       | Any domain variable in `target_overview` | Narrows match to a specific domain |

### Default config

The checked-in config at `policyengine_us_data/calibration/target_config.yaml` reproduces the
junkyard notebook's 22 excluded target groups. It drops:

- **13 national-level variables**: alimony, charitable deduction, child support, interest deduction,
  medical expense deduction, net worth, person count, real estate taxes, rent, social security
  dependents/survivors
- **9 district-level variables**: ACA PTC, EITC, income tax before credits, medical expense
  deduction, net capital gains, rental income, tax unit count, partnership/S-corp income, taxable
  social security

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

| Flag             | Default                                       | Description                                    |
| ---------------- | --------------------------------------------- | ---------------------------------------------- |
| `--dataset`      | `storage/stratified_extended_cps_2025.h5`     | Path to CPS h5 file                            |
| `--db-path`      | `storage/calibration/policy_data.db`          | Path to target database                        |
| `--output`       | `storage/calibration/calibration_weights.npy` | Weight output path                             |
| `--puf-dataset`  | None                                          | Path to PUF h5 (enables PUF cloning)           |
| `--preset`       | `local`                                       | L0 preset: `local` (1e-8) or `national` (1e-4) |
| `--lambda-l0`    | None                                          | Custom L0 penalty (overrides `--preset`)       |
| `--epochs`       | 100                                           | Training epochs                                |
| `--device`       | `cpu`                                         | `cpu` or `cuda`                                |
| `--n-clones`     | 430                                           | Number of dataset clones                       |
| `--seed`         | 42                                            | Random seed for geography assignment           |
| `--national`     | False                                         | Use national preset (λ_L0=1e-4, ~50K records)  |
| `--workers`      | 1                                             | Parallel workers for per-state precomputation  |
| `--county-level` | False                                         | Include county-level targets (slower)          |

### Target selection

| Flag                     | Default | Description                               |
| ------------------------ | ------- | ----------------------------------------- |
| `--target-config`        | None    | Path to YAML exclusion config             |
| `--domain-variables`     | None    | Comma-separated domain filter (SQL-level) |
| `--hierarchical-domains` | None    | Domains for hierarchical uprating         |

### Checkpoint flags

| Flag               | Default                    | Description                                                                            |
| ------------------ | -------------------------- | -------------------------------------------------------------------------------------- |
| `--build-only`     | False                      | Build matrix, save package, skip fitting                                               |
| `--package-path`   | None                       | Load pre-built package (uploads to Modal volume automatically when using Modal runner) |
| `--package-output` | Auto (when `--build-only`) | Where to save package                                                                  |

### Hyperparameter flags

| Flag              | Default | Junkyard value | Description                                 |
| ----------------- | ------- | -------------- | ------------------------------------------- |
| `--beta`          | 0.35    | 0.65           | L0 gate temperature (higher = softer gates) |
| `--lambda-l2`     | 1e-12   | 1e-8           | L2 regularization on weights                |
| `--learning-rate` | 0.15    | 0.15           | Optimizer learning rate                     |

### Skip flags

| Flag                        | Description                     |
| --------------------------- | ------------------------------- |
| `--skip-puf`                | Skip PUF clone + QRF imputation |
| `--skip-source-impute`      | Skip ACS/SIPP/SCF re-imputation |
| `--skip-takeup-rerandomize` | Skip takeup re-randomization    |

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

The `targets_df` DataFrame has columns: `variable`, `geo_level`, `geographic_id`, `domain_variable`,
`value`, and others from the database.

## Validating a Package

Before uploading a package to Modal, validate it:

```bash
# Default package location
python -m policyengine_us_data.calibration.validate_package

# Specific package
python -m policyengine_us_data.calibration.validate_package path/to/calibration_package.pkl

# Strict mode: fail if any target has row_sum/target < 1%
python -m policyengine_us_data.calibration.validate_package --strict
```

Exit codes: **0** = pass, **1** = impossible targets, **2** = strict ratio failures.

Validation also runs automatically after `--build-only`.

## Hyperparameter Tuning Guide

The three key hyperparameters control the tradeoff between target accuracy and sparsity:

- **`beta`** (L0 gate temperature): Controls how sharply the L0 gates open/close. Higher values
  (0.5--0.8) give softer decisions and more exploration early in training. Lower values (0.2--0.4)
  give harder on/off decisions.

- **`lambda_l0`** (via `--preset` or `--lambda-l0`): Controls how many records survive. `1e-8`
  (local preset) keeps millions of records for local-area analysis. `1e-4` (national preset) keeps
  ~50K for the web app.

- **`lambda_l2`**: Regularizes weight magnitudes. Larger values (1e-8) prevent any single record
  from having extreme weight. Smaller values (1e-12) allow more weight concentration.

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

| Target                          | Description                                        |
| ------------------------------- | -------------------------------------------------- |
| `make calibrate`                | Full local pipeline with target config             |
| `make calibrate-build`          | Build-only mode (saves package, no fitting)        |
| `make build-matrices`           | Build calibration matrices on Modal (CPU)          |
| `make calibrate-modal`          | Fit county-level weights on Modal GPU              |
| `make calibrate-modal-national` | Fit national weights on Modal GPU (T4)             |
| `make calibrate-both`           | Run county + national fits in parallel             |
| `make stage-h5s`                | Build state/district/city H5s on Modal             |
| `make stage-national-h5`        | Build national US.h5 on Modal                      |
| `make stage-all-h5s`            | Run both staging jobs in parallel                  |
| `make promote`                  | Promote staged files to versioned HF paths         |
| `make pipeline`                 | Print sequential steps for full pipeline           |
| `make validate-staging`         | Validate staged H5s against targets (states only)  |
| `make validate-staging-full`    | Validate staged H5s (states + districts)           |
| `make upload-validation`        | Push validation_results.csv to HF                  |
| `make check-staging`            | Smoke test: sum key variables across all state H5s |
| `make check-sanity`             | Quick structural integrity check on one state      |
| `make upload-calibration`       | Upload weights, blocks, and logs to HF             |

## Takeup Rerandomization

The calibration pipeline uses two independent code paths to compute the same target variables:

1. **Matrix builder** (`UnifiedMatrixBuilder.build_matrix`): Computes a sparse calibration matrix
   $X$ where each row is a target and each column is a cloned household. The optimizer finds weights
   $w$ that minimize $|Xw - t|$ (target values).

1. **Stacked builder** (`create_sparse_cd_stacked_dataset`): Produces the `.h5` files that users
   load in `Microsimulation`. It reconstructs each congressional district by combining base CPS
   records with calibrated weights and block-level geography.

For the calibration to be meaningful, **both paths must produce identical values** for every target
variable. If the matrix builder computes $X\_{snap,NC} \\cdot w = $5.2B$ but the stacked NC.h5 file
yields `sim.calculate("snap") * household_weight = $4.8B`, then the optimizer's solution does not
actually match the target.

### The problem with takeup variables

Variables like `snap`, `aca_ptc`, `ssi`, and `medicaid` depend on **takeup draws** — random
Bernoulli samples that determine whether an eligible household actually claims the benefit. By
default, PolicyEngine draws these at simulation time using Python's built-in `hash()`, which is
randomized per process.

This means loading the same H5 file in two different processes can produce different SNAP totals,
even with the same weights. Worse, the matrix builder runs in process A while the stacked builder
runs in process B, so their draws can diverge.

### The solution: block-level seeding

Both paths call `seeded_rng(variable_name, salt=f"{block_geoid}:{household_id}")` to generate
deterministic takeup draws. This ensures:

- The same household at the same block always gets the same draw
- Draws are stable across processes (no dependency on `hash()`)
- Draws are stable when aggregating to any geography (state, CD, county)

All takeup variables in `SIMPLE_TAKEUP_VARS` (in `utils/takeup.py`) receive block-seeded draws in
the H5 builder, including `would_file_taxes_voluntarily`. The calibration matrix uses
`TAKEUP_AFFECTED_TARGETS` to identify which *target* variables need takeup-adjusted rows, but the H5
builder applies draws to all `SIMPLE_TAKEUP_VARS` so that every takeup variable gets proper
block-seeded values.

The `--skip-takeup-rerandomize` flag disables this rerandomization for faster iteration when you
only care about non-takeup variables. Do not use it for production calibrations.

## Block-Level Seeding

Each cloned household is assigned to a Census block (15-digit GEOID) during the
`assign_random_geography` step. The first 2 digits are the state FIPS code, which determines the
household's takeup rates (since benefit eligibility rules are state-specific).

### Mechanism

```python
rng = seeded_rng(variable_name, salt=f"{block_geoid}:{household_id}")
draw = rng.random()
takes_up = draw < takeup_rate[state_fips]
```

The `seeded_rng` function uses `_stable_string_hash` — a deterministic hash that does not depend on
Python's `PYTHONHASHSEED`. This is critical because Python's built-in `hash()` is randomized per
process by default (since Python 3.3).

### Why block (not CD or state)?

Blocks are the finest Census geography. A household's block assignment stays the same regardless of
how blocks are aggregated — the same household-block-draw triple produces the same result whether
you are building an H5 for a state, a congressional district, or a county. This means:

- State H5s and district H5s are consistent (no draw drift)
- Future county-level H5s will also be consistent
- Re-running the pipeline with different area selections yields the same per-household values

### Inactive records

When converting to stacked format, households that are not assigned to a given CD get zero weight.
These inactive records must receive an empty string `""` as their block GEOID, not a real block. If
they received real blocks, they would inflate the entity count `n` passed to the RNG, shifting the
draw positions for active entities and breaking the $X \\cdot w$ consistency invariant.

## The $X \\cdot w$ Consistency Invariant

### Formal statement

For every target variable $v$ and geography $g$:

$$X\_{v,g} \\cdot w = \\sum\_{i \\in g} \\text{sim.calculate}(v)\_i \\times w_i$$

where the left side comes from the matrix builder and the right side comes from loading the stacked
H5 and running `Microsimulation.calculate()`.

### Why it matters

This invariant is what makes calibration meaningful. Without it, the optimizer's solution (which
minimizes $|Xw - t|$) does not actually produce a dataset that matches the targets. The weights
would be "correct" in the matrix builder's view but produce different totals in the H5 files that
users actually load.

### Known sources of drift

1. **Mismatched takeup draws**: The matrix builder and stacked builder use different RNG states.
   Solved by block-level seeding (see above).

1. **Different block assignments**: The stacked format uses first-clone-wins for multi-clone-same-CD
   records. With ~11M blocks and 3-10 clones, collision rate is ~0.7-10% of records. In practice,
   the residual mismatch is negligible.

1. **Inactive records in RNG calls**: If inactive records (w=0) receive real block GEOIDs, they
   inflate the entity count for that block's RNG call, shifting draw positions. Solved by using `""`
   for inactive blocks.

1. **Entity ordering**: Both paths must iterate over entities in the same order
   (`sim.calculate("{entity}_id", map_to=entity)`). NumPy boolean masking preserves order, so
   `draws[i]` maps to the same entity in both paths.

### Testing

The `test_xw_consistency.py` test (`pytest -m slow`) verifies this invariant end-to-end:

1. Load base dataset, create geography with uniform weights
1. Build $X$ with the matrix builder (including takeup rerandomization)
1. Convert weights to stacked format
1. Build stacked H5 for selected CDs
1. Compare $X \\cdot w$ vs `sim.calculate() * household_weight` — assert ratio within 1%

## Post-Calibration Gating Workflow

After the pipeline stages H5 files to HuggingFace, two manual review gates determine whether to
promote to production.

### Gate 1: Review calibration fit

Load `calibration_log.csv` in the microcalibrate dashboard. This file contains the $X \\cdot w$
values from the matrix builder for every target at every epoch.

**What to check:**

- Loss curve converges (no divergence in later epochs)
- No individual target groups diverging while others improve
- Final loss is comparable to or better than the previous production run

If fit is poor, re-calibrate with different hyperparameters (learning rate, lambda_l0, beta,
epochs).

### Gate 2: Review simulation quality

```bash
make validate-staging          # states only (~30 min)
make validate-staging-full     # states + districts (~3 hrs)
make upload-validation         # push CSV to HF
```

This produces `validation_results.csv` with `sim.calculate()` values for every target. Load it in
the dashboard's Combined tab alongside `calibration_log.csv`.

**What to check:**

- `CalibrationVsSimComparison` shows the gap between $X \\cdot w$ and `sim.calculate()` values
- No large regressions vs the previous production run
- Sanity check column has no FAIL entries

### Promote

If both gates pass:

- Run the "Promote Local Area H5 Files" GitHub workflow, OR
- Manually copy staged files to the production paths in the HF repo

### Structural pre-flight

For a quick structural check without loading the full database:

```bash
make check-sanity              # one state, ~2 min
```

This runs weight non-negativity, entity ID uniqueness, NaN/Inf detection, person-household mapping,
boolean takeup validation, and per-household range checks.
