# Pipeline internals — developer reference

Internal notebooks for the policyengine-us-data calibration pipeline. Not published in the Jupyter
Book. Use these when debugging a wrong aggregate, understanding an implementation choice, or
extending the pipeline.

______________________________________________________________________

## Notebooks

| Notebook                                                                                                             | Stages                                     | Required files / inputs                                                     |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------- |
| [`data_build_internals.ipynb`](data_build_internals.ipynb)                                                           | Stage 1: build_datasets                    | donor QRF cells need ACS/SIPP/SCF files                                     |
| [`calibration_package_internals.ipynb`](calibration_package_internals.ipynb)                                         | Stage 2: build_package                     | Part 1 uses a toy sparse matrix; Parts 2–5 use static excerpts or toy demos |
| [`optimization_and_local_dataset_assembly_internals.ipynb`](optimization_and_local_dataset_assembly_internals.ipynb) | Stages 3–4: fit_weights, publish_and_stage | L0 toy run; diagnostic cells need a completed run's CSV output              |

### Which notebook to open

**Wrong value in an individual record** → `data_build_internals.ipynb` The record value is set in
Stage 1 and never changed by calibration. The problem is in clone creation, source imputation, or
PUF imputation.

**Wrong weighted aggregate despite correct record values** → `calibration_package_internals.ipynb`
The calibration matrix determines which records contribute to which targets. Check matrix assembly,
domain constraints, and takeup randomization.

**Calibration converged but aggregate still off, or H5 values unexpected** →
`local_dataset_assembly_internals.ipynb` The optimizer may have failed to match a target, or the
weight expansion step is applying incorrect geographic filtering. Check L0 diagnostics and weight
expansion.

______________________________________________________________________

## Pipeline orchestration reference

The pipeline runs on [Modal](https://modal.com) via `modal_app/pipeline.py`. It chains five steps
under a single **run ID**, with resume support and per-step checkpointing.

### Run ID format

```
{version}_{sha[:8]}_{timestamp}
```

Example: `1.23.0_a3f1b2c4_20260315_142037`

- `version`: package version from `pyproject.toml` at the baked image
- `sha[:8]`: first 8 characters of the branch tip SHA at orchestrator start
- `timestamp`: UTC datetime in `YYYYMMDD_HHMMSS`

The SHA is pinned at orchestrator start. If the branch moves mid-run, intermediate artifacts may
come from different commits — the pipeline warns but does not abort.

### Step dependency graph

```
Step 1: build_datasets      → produces source_imputed_*.h5, policy_data.db
           ↓
Step 2: build_package       → produces calibration_package.pkl (the calibration matrix)
           ↓
Step 3: fit_weights         → regional and national fits run in parallel
           ↓                  produces calibration_weights.npy
Step 4: publish_and_stage   → builds H5 files per area, validates, stages to HuggingFace
           ↓
Step 5: promote             → moves staged H5s to production (no new computation)
```

Steps 3 regional and national fits spawn concurrently (`regional_handle.spawn()` /
`national_handle.spawn()`). The orchestrator waits for both before advancing to Step 4.

Default hyperparameters passed in `run_pipeline()`:

- Regional: `beta=0.65`, `lambda_l0=1e-7`, `lambda_l2=1e-8`, 1,000 epochs, T4 GPU
- National: `beta=0.65`, `lambda_l0=1e-4`, `lambda_l2=1e-12`, 4,000 epochs, T4 GPU

### Modal volumes

Two Modal volumes back the pipeline:

| Volume name          | Mount path  | Purpose                                          |
| -------------------- | ----------- | ------------------------------------------------ |
| `pipeline-artifacts` | `/pipeline` | Run metadata, calibration artifacts, diagnostics |
| `local-area-staging` | `/staging`  | Intermediate H5 files during publish step        |

Directory layout inside `pipeline-artifacts`:

```
/pipeline/
  runs/
    {run_id}/
      meta.json               ← run metadata (status, step timings, validation summary)
      diagnostics/
        calibration_log.csv
        unified_diagnostics.csv
        unified_run_config.json
        national_calibration_log.csv
        national_unified_diagnostics.csv
        validation_results.csv
        national_validation.txt
  artifacts/
    {run_id}/
      calibration_package.pkl
      calibration_weights.npy
      national_calibration_weights.npy
      source_imputed_*.h5
      policy_data.db
```

### `meta.json` structure

```json
{
  "run_id": "1.23.0_a3f1b2c4_20260315_142037",
  "branch": "main",
  "sha": "a3f1b2c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0",
  "version": "1.23.0",
  "start_time": "2026-03-15T14:20:37+00:00",
  "status": "running | completed | failed | promoted",
  "step_timings": {
    "build_datasets": {
      "start": "2026-03-15T14:20:40+00:00",
      "end": "2026-03-15T16:45:12+00:00",
      "duration_s": 8672.1,
      "status": "completed"
    },
    "build_package": { "...": "..." },
    "fit_weights":   { "...": "..." },
    "publish":       { "...": "..." },
    "validation": {
      "total_targets": 3842,
      "sanity_failures": 12,
      "mean_rel_abs_error": 0.0231,
      "worst_areas": [...]
    }
  },
  "error": null
}
```

### Resume logic

The orchestrator auto-resumes if it finds a run with the same `branch` + `sha` and
`status == "running"` in the pipeline volume. Resume skips any step whose
`step_timings[step]["status"] == "completed"`.

If the branch has moved since the run started (SHA mismatch), the orchestrator raises a
`RuntimeError` and requires starting a fresh run.

To force a resume of a specific run:

```bash
modal run --detach modal_app/pipeline.py::main \
    --action run --resume-run-id 1.23.0_a3f1b2c4_20260315_142037
```

To start fresh (ignore resumable runs):

```bash
modal run --detach modal_app/pipeline.py::main \
    --action run --branch main
```

### HuggingFace artifact paths

All artifacts land in `policyengine/policyengine-us-data` (model repo) under the `staging/` prefix
until promoted.

| Artifact            | HF path (staging)                                               | HF path (production after promote) |
| ------------------- | --------------------------------------------------------------- | ---------------------------------- |
| source_imputed H5s  | `staging/calibration/source_imputed_*.h5`                       | `calibration/source_imputed_*.h5`  |
| policy_data.db      | `staging/calibration/policy_data.db`                            | `calibration/policy_data.db`       |
| Calibration log     | `calibration/runs/{run_id}/diagnostics/calibration_log.csv`     | — (never promoted)                 |
| Unified diagnostics | `calibration/runs/{run_id}/diagnostics/unified_diagnostics.csv` | — (never promoted)                 |
| Validation results  | `calibration/runs/{run_id}/diagnostics/validation_results.csv`  | — (never promoted)                 |
| Local area H5s      | `staging/` (area-specific paths)                                | final dataset paths                |

Diagnostics are never promoted — they remain under `calibration/runs/{run_id}/` permanently.

To fetch a diagnostic file from a known run ID:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="policyengine/policyengine-us-data",
    repo_type="model",
    filename=f"calibration/runs/{run_id}/diagnostics/unified_diagnostics.csv",
)
```

### Checking pipeline status

```bash
modal run modal_app/pipeline.py::main --action status
```

This reads `meta.json` for all runs in the pipeline volume and prints step completion status and
timings.

### Promoting a completed run

```bash
modal run modal_app/pipeline.py::main \
    --action promote --run-id 1.23.0_a3f1b2c4_20260315_142037
```

Promote moves staged H5s to their production paths on HuggingFace. It does not re-run any
computation. After promotion, the run's `status` in `meta.json` changes to `"promoted"`.

______________________________________________________________________

## File reference

> **Note:** This reference reflects the codebase as of the time of writing. File responsibilities
> may shift as the pipeline evolves — use this as a starting point, then read the file to confirm.

### `policyengine_us_data/calibration/`

| File                           | Purpose                                                                                                                                                                                                                                                         |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `unified_calibration.py`       | Main calibration entry point: clones CPS, assigns geography, builds matrix, runs L0 optimizer, saves weights. Start here for the end-to-end flow.                                                                                                               |
| `unified_matrix_builder.py`    | Builds the sparse calibration matrix. Per-state simulation, clone loop, domain constraints, takeup re-randomization, COO assembly.                                                                                                                              |
| `clone_and_assign.py`          | Clones CPS records N times, assigns each clone a random census block with no-CD-collision constraint and AGI-conditional routing.                                                                                                                               |
| `block_assignment.py`          | Per-CD block assignment and geographic variable derivation (county, tract, CBSA, SLDU, SLDL, place, PUMA, VTD, ZCTA) from block GEOIDs.                                                                                                                         |
| `county_assignment.py`         | Legacy/fallback: assigns counties within CDs using P(county \| CD). Only called by `block_assignment.py::_generate_fallback_blocks()` when a CD is missing from the pre-computed block distribution (primarily in tests). Not used in production pipeline runs. |
| `puf_impute.py`                | PUF cloning: doubles the dataset, imputes 70+ tax variables via sequential QRF, reconciles Social Security sub-components.                                                                                                                                      |
| `source_impute.py`             | Re-imputes housing, asset, and labor-market variables from ACS, SIPP, ORG, and SCF donor surveys using QRF.                                                                                                                                                     |
| `create_source_imputed_cps.py` | Standalone script that runs `source_impute.py` on the stratified extended CPS to produce the dataset used by calibration.                                                                                                                                       |
| `create_stratified_cps.py`     | Creates a stratified CPS sample preserving all high-income households while maintaining low-income diversity.                                                                                                                                                   |
| `publish_local_area.py`        | Builds per-area H5 files (states, districts, cities) from calibrated weights. Weight expansion, entity cloning, geography override, SPM recalculation, takeup draws.                                                                                            |
| `calibration_utils.py`         | Shared utilities: state mappings, SPM threshold calculation, geographic adjustment factors, target group functions, initial weight computation.                                                                                                                 |
| `target_config.yaml`           | Include rules that gate which DB targets enter calibration (applied post-matrix-build). The training config.                                                                                                                                                    |
| `target_config_full.yaml`      | Broader include rules used for validation — includes targets not in the training set for holdout evaluation.                                                                                                                                                    |
| `validate_staging.py`          | Validates built H5 files by running `sim.calculate()` and comparing weighted aggregates against DB targets. Produces `validation_results.csv`.                                                                                                                  |
| `validate_national_h5.py`      | Validates the national `US.h5` against known national totals and runs structural sanity checks.                                                                                                                                                                 |
| `validate_package.py`          | Validates a calibration package (matrix + targets) before uploading to Modal — checks structure, achievability, and provenance.                                                                                                                                 |
| `sanity_checks.py`             | Structural integrity checks on H5 files: weights, monetary variable ranges, takeup booleans, entity ID consistency.                                                                                                                                             |
| `check_staging_sums.py`        | Standalone CLI utility (not part of the automated pipeline): sums key variables across all 51 state H5 files and compares to national references. Run manually via `make check-staging` or `python -m ...`.                                                     |
| `promote_local_h5s.py`         | Standalone CLI utility (not part of the automated pipeline): promotes locally-built H5 files to production via HuggingFace staging and GCS upload. Used for manual local builds outside Modal.                                                                  |

### `modal_app/`

| File                           | Purpose                                                                                                                                                         |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pipeline.py`                  | End-to-end pipeline orchestrator: chains dataset build → matrix build → weight fitting → H5 publish → promote. Manages run IDs, resume, and diagnostics upload. |
| `data_build.py`                | Modal app for Stage 1: parallel dataset building (CPS extraction, PUF cloning, source imputation) with checkpoint persistence.                                  |
| `remote_calibration_runner.py` | Modal app for Stages 2–3: builds calibration package and/or runs L0 optimizer on GPU. Supports `build_package` and `fit_from_package` workflows.                |
| `local_area.py`                | Modal app for Stage 4: parallel H5 building with distributed worker coordination, LPT scheduling, and validation aggregation.                                   |
| `worker_script.py`             | Subprocess worker called by `local_area.py` to build individual H5 files. Runs in a separate process to avoid import conflicts.                                 |
| `images.py`                    | Defines pre-baked Modal container images with source code, dependencies, and Git metadata for reproducibility.                                                  |
| `resilience.py`                | Retry and resume utilities for Modal workflows (exponential backoff, idempotent step execution).                                                                |

### `policyengine_us_data/db/`

| File                           | Purpose                                                                                                                    |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `create_database_tables.py`    | Defines SQLModel schema for `policy_data.db` (targets, strata, constraints, metadata). Entry point for `make database`.    |
| `create_initial_strata.py`     | Seeds the strata table with geographic and domain strata from census and administrative boundaries.                        |
| `create_field_valid_values.py` | Populates the `field_valid_values` table with valid operations, active flags, periods, and policyengine-us variable names. |
| `etl_age.py`                   | Loads age-bin population targets (Census) into `policy_data.db`.                                                           |
| `etl_irs_soi.py`               | Loads IRS SOI district-level tax targets (AGI, credits, deductions) into `policy_data.db`.                                 |
| `etl_snap.py`                  | Loads SNAP household count and benefit targets (USDA) into `policy_data.db`.                                               |
| `etl_medicaid.py`              | Loads Medicaid enrollment targets into `policy_data.db`.                                                                   |
| `etl_national_targets.py`      | Loads national-level calibration targets into `policy_data.db`.                                                            |
| `etl_pregnancy.py`             | Loads state-level birth count targets from CDC VSRR and female population from Census ACS.                                 |
| `etl_state_income_tax.py`      | Loads state income tax collection targets from Census Bureau STC survey.                                                   |
| `validate_database.py`         | Post-build validation of `policy_data.db`: checks target completeness, value ranges, and cross-table consistency.          |
| `validate_hierarchy.py`        | Validates parent-child strata hierarchy: geographic and age strata relationships.                                          |
