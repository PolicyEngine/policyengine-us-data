# Modal Apps

Modal apps for running calibration and local area publishing on Modal's cloud infrastructure.

## Prerequisites

- [Modal](https://modal.com/) account and CLI installed (`pip install modal`)
- `modal token new` to authenticate
- HuggingFace token stored as Modal secret named `huggingface-token`
- GCP credentials stored as Modal secret named `gcp-credentials`

## Local Area Publishing (`local_area.py`)

Builds state, district, and city H5 files in parallel, then uploads them to GCS and HuggingFace.

### Two-Phase Workflow

Publishing is split into **build+stage** and **promote** to prevent silent failures:

1. **Build + Stage** (`main` entrypoint): Builds all H5 files, uploads to GCS (production) and HuggingFace (`staging/` folder). Does NOT overwrite HuggingFace production files.
2. **Promote** (`main_promote` entrypoint): Copies files from `staging/` to production paths on HuggingFace, then cleans up staging. Raises an error if the promote commit is a no-op (e.g., staging files identical to production).

### GitHub Actions Workflows

- **Publish Local Area H5 Files** (`local_area_publish.yaml`): Runs build+stage. Triggered by pushes to `local_area_calibration/`, `repository_dispatch`, or manual dispatch.
- **Promote Local Area H5 Files** (`local_area_promote.yaml`): Manual dispatch only. Requires the version string from the build output.

### Manual Usage

```bash
# Build + stage
modal run modal_app/local_area.py::main --branch=main --num-workers=8

# Promote (after verifying staging looks correct)
modal run modal_app/local_area.py::main_promote --version=1.62.0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--branch` | `main` | Git branch to clone and run |
| `--num-workers` | `8` | Number of parallel build workers |
| `--skip-upload` | `false` | Build only, skip GCS/HF upload |
| `--version` | (required for promote) | Version to promote |

### Important Notes

- Every build clears the Modal volume cache and rebuilds from scratch (no stale data reuse).
- Calibration inputs (weights, dataset, database) are re-downloaded each run.
- Modal clones from GitHub, so local changes must be pushed before they take effect.

## GPU Weight Fitting (`remote_calibration_runner.py`)

Run calibration weight fitting on Modal's cloud GPUs.

### Usage

```bash
modal run modal_app/remote_calibration_runner.py --branch <branch> --epochs <n> --gpu <type>
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--branch` | `main` | Git branch to clone and run |
| `--epochs` | `200` | Number of training epochs |
| `--gpu` | `T4` | GPU type: `T4`, `A10`, `A100-40GB`, `A100-80GB`, `H100` |
| `--output` | `calibration_weights.npy` | Local path for weights file |
| `--log-output` | `calibration_log.csv` | Local path for calibration log |

### Example

```bash
modal run modal_app/remote_calibration_runner.py --branch health-insurance-premiums --epochs 100 --gpu T4
```

### Output Files

- **calibration_weights.npy** - Fitted household weights
- **calibration_log.csv** - Per-target performance metrics across epochs

### Changing Hyperparameters

Hyperparameters are in `policyengine_us_data/datasets/cps/local_area_calibration/fit_calibration_weights.py`:

```python
BETA = 0.35
GAMMA = -0.1
ZETA = 1.1
INIT_KEEP_PROB = 0.999
LOG_WEIGHT_JITTER_SD = 0.05
LOG_ALPHA_JITTER_SD = 0.01
LAMBDA_L0 = 1e-8
LAMBDA_L2 = 1e-8
LEARNING_RATE = 0.15
```

To change them:
1. Edit `fit_calibration_weights.py`
2. Commit and push to your branch
3. Re-run the Modal command with that branch

### Important Notes

- **Keep your connection open** - Modal needs to stay connected to download results. Don't close your laptop or let it sleep until you see the local "Weights saved to:" and "Calibration log saved to:" messages.
- Modal clones from GitHub, so local changes must be pushed before they take effect.
