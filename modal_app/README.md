# Modal App for GPU Weight Fitting

Run calibration weight fitting on Modal's cloud GPUs.

## Prerequisites

- [Modal](https://modal.com/) account and CLI installed (`pip install modal`)
- `modal token new` to authenticate
- HuggingFace token stored as Modal secret named `huggingface-token`

## Usage

```bash
modal run modal_app/remote_calibration_runner.py --branch <branch> --epochs <n> --gpu <type>
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--branch` | `main` | Git branch to clone and run |
| `--epochs` | `200` | Number of training epochs |
| `--gpu` | `T4` | GPU type: `T4`, `A10`, `A100-40GB`, `A100-80GB`, `H100` |
| `--output` | `calibration_weights.npy` | Local path for weights file |
| `--log-output` | `unified_diagnostics.csv` | Local path for diagnostics log |
| `--log-freq` | (none) | Log every N epochs to `calibration_log.csv` |
| `--push-results` | `False` | Upload weights, blocks, and logs to HuggingFace |
| `--trigger-publish` | `False` | Fire `repository_dispatch` to trigger the Publish workflow |
| `--target-config` | (none) | Target configuration name |
| `--beta` | (none) | L0 relaxation parameter |
| `--lambda-l0` | (none) | L0 penalty weight |
| `--lambda-l2` | (none) | L2 penalty weight |
| `--learning-rate` | (none) | Optimizer learning rate |
| `--package-path` | (none) | Local path to a pre-built calibration package (uploads to Modal volume, then fits) |
| `--prebuilt-matrices` | `False` | Fit from pre-built package on Modal volume |
| `--full-pipeline` | `False` | Force full rebuild even if a package exists on the volume |
| `--county-level` | `False` | Include county-level targets |
| `--workers` | `1` | Number of parallel workers |

### Examples

**Two-step workflow (recommended):**

Step 1 — Build the X matrix on CPU (no GPU cost, 10h timeout):
```bash
modal run modal_app/remote_calibration_runner.py::build_package \
  --branch main
```

Step 2 — Fit weights from the pre-built package on GPU:
```bash
modal run modal_app/remote_calibration_runner.py::main \
  --branch main --epochs 200 --gpu A100-80GB \
  --prebuilt-matrices --push-results
```

**Full pipeline (single step, requires enough timeout for matrix build + fit):**
```bash
modal run modal_app/remote_calibration_runner.py::main \
  --branch main --epochs 200 --gpu A100-80GB \
  --full-pipeline --push-results
```

Fit, push, and trigger the publish workflow:
```bash
modal run modal_app/remote_calibration_runner.py::main \
  --gpu A100-80GB --epochs 200 \
  --prebuilt-matrices --push-results --trigger-publish
```

## Output Files

Every run produces these local files (whichever the calibration script emits):

- **calibration_weights.npy** — Fitted household weights
- **unified_diagnostics.csv** — Final per-target diagnostics
- **calibration_log.csv** — Per-target metrics across epochs (requires `--log-freq`)
- **unified_run_config.json** — Run configuration and summary stats

## Artifact Upload to HuggingFace

The `--push-results` flag uploads all artifacts to HuggingFace in a single
atomic commit after writing them locally:

| Local file | HF path |
|------------|---------|
| `calibration_weights.npy` | `calibration/calibration_weights.npy` |
| `calibration_log.csv` | `calibration/logs/calibration_log.csv` |
| `unified_diagnostics.csv` | `calibration/logs/unified_diagnostics.csv` |
| `unified_run_config.json` | `calibration/logs/unified_run_config.json` |

Each upload overwrites the previous files. HF git history provides implicit
versioning — browse past commits to see earlier runs.

## Triggering the Publish Workflow

The `--trigger-publish` flag fires a `repository_dispatch` event
(`calibration-updated`) on GitHub, which starts the "Publish Local Area H5
Files" workflow. Requires `GITHUB_TOKEN` or
`POLICYENGINE_US_DATA_GITHUB_TOKEN` set locally.

### Downloading logs

```python
from policyengine_us_data.utils.huggingface import download_calibration_logs

paths = download_calibration_logs("/tmp/cal_logs")
# {"calibration_log": Path(...), "diagnostics": Path(...), "config": Path(...)}
```

Pass `version="<commit-or-tag>"` to download from a specific HF revision.

### Viewing logs in the microcalibrate dashboard

The [microcalibration dashboard](https://github.com/PolicyEngine/microcalibrate)
has a **Hugging Face** tab that loads `calibration_log.csv` directly from HF:

1. Open the dashboard
2. Click the **Hugging Face** tab
3. Defaults are pre-filled — click **Load**
4. Change the **Revision** field to load from a specific HF commit or tag

## Important Notes

- **Keep your connection open** — Modal needs to stay connected to download
  results. Don't close your laptop or let it sleep until you see the local
  "Weights saved to:" message.
- Modal clones from GitHub, so local changes must be pushed before they
  take effect.
- `--push-results` requires the `HUGGING_FACE_TOKEN` environment variable
  to be set locally (not just as a Modal secret).
- `--trigger-publish` requires `GITHUB_TOKEN` or
  `POLICYENGINE_US_DATA_GITHUB_TOKEN` set locally.

## Full Pipeline Reference

The calibration pipeline has six stages. Each can be run locally, via Modal CLI, or via GitHub Actions.

### Stage 1: Build data

Produces `stratified_extended_cps_2024.h5` from raw CPS/PUF/ACS inputs.

| Method | Command |
|--------|---------|
| **Local** | `make data` |
| **Modal (CI)** | `modal run modal_app/data_build.py --branch=<branch>` |
| **GitHub Actions** | Automatic on merge to `main` via `code_changes.yaml` → `reusable_test.yaml` (with `full_suite: true`). Also triggered by `pr_code_changes.yaml` on PRs. |

Notes:
- `make data` stops at `create_stratified_cps.py`. Use `make data-legacy` to also build `enhanced_cps.py` and `small_enhanced_cps.py`.
- `data_build.py` (CI) always builds the full suite including enhanced_cps.

### Stage 2: Upload inputs to HuggingFace

Pushes the dataset and (optionally) database to HF so Modal can download them.

| Artifact | Command |
|----------|---------|
| Dataset | `make upload-dataset` |
| Database | `make upload-database` |

The database is relatively stable; only re-upload after `make database` or `make database-refresh`.

### Stage 3: Build calibration matrices

Downloads dataset + database from HF, builds the X matrix, saves to Modal volume. CPU-only, no GPU cost.

| Method | Command |
|--------|---------|
| **Local** | `make calibrate-build` |
| **Modal CLI** | `make build-matrices BRANCH=<branch>` (aka `modal run modal_app/remote_calibration_runner.py::build_package --branch=<branch>`) |

### Stage 4: Fit calibration weights

Loads pre-built matrices from Modal volume, fits L0-regularized weights on GPU.

| Method | Command |
|--------|---------|
| **Local (CPU)** | `make calibrate` |
| **Modal CLI** | `make calibrate-modal BRANCH=<branch> GPU=<gpu> EPOCHS=<n>` |

`make calibrate-modal` passes `--prebuilt-matrices --push-results` automatically.

Full example:
```
modal run modal_app/remote_calibration_runner.py::main \
  --branch calibration-pipeline-improvements \
  --gpu T4 --epochs 1000 \
  --beta 0.65 --lambda-l0 1e-6 --lambda-l2 1e-8 \
  --log-freq 500 \
  --target-config policyengine_us_data/calibration/target_config.yaml \
  --prebuilt-matrices --push-results
```

**Safety check**: If a pre-built package exists on the volume and you don't pass `--prebuilt-matrices` or `--full-pipeline`, the runner refuses to proceed and tells you which flag to add. This prevents accidentally rebuilding from scratch.

Artifacts uploaded to HF by `--push-results`:

| Local file | HF path |
|------------|---------|
| `calibration_weights.npy` | `calibration/calibration_weights.npy` |
| `calibration_log.csv` | `calibration/logs/calibration_log.csv` |
| `unified_diagnostics.csv` | `calibration/logs/unified_diagnostics.csv` |
| `unified_run_config.json` | `calibration/logs/unified_run_config.json` |

### Stage 5: Build and stage local area H5 files

Downloads weights + dataset + database from HF, builds state/district/city H5 files.

| Method | Command |
|--------|---------|
| **Local** | `python policyengine_us_data/calibration/publish_local_area.py --rerandomize-takeup` |
| **Modal CLI** | `make stage-h5s BRANCH=<branch>` (aka `modal run modal_app/local_area.py --branch=<branch> --num-workers=8`) |
| **GitHub Actions** | "Publish Local Area H5 Files" workflow — manual trigger via `workflow_dispatch`, or automatic via `repository_dispatch` (`--trigger-publish` flag), or on code push to `main` touching `calibration/` or `modal_app/`. |

This stages H5s to HF `staging/` paths. It does NOT promote to production or GCS.

### Stage 6: Promote (manual gate)

Moves files from HF staging to production paths and uploads to GCS.

| Method | Command |
|--------|---------|
| **Modal CLI** | `modal run modal_app/local_area.py::main_promote --version=<version>` |
| **GitHub Actions** | "Promote Local Area H5 Files" workflow — manual `workflow_dispatch` only. Requires `version` input. |

### One-command pipeline

For the common case (local data build → Modal calibration → Modal staging):

```
make pipeline GPU=T4 EPOCHS=1000 BRANCH=calibration-pipeline-improvements
```

This chains: `data` → `upload-dataset` → `build-matrices` → `calibrate-modal` → `stage-h5s`.
