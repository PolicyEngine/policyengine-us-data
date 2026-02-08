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
| `--log-output` | `calibration_log.csv` | Local path for calibration log |

### Example

```bash
modal run modal_app/remote_calibration_runner.py --branch health-insurance-premiums --epochs 100 --gpu T4
```

## Output Files

- **calibration_weights.npy** - Fitted household weights
- **calibration_log.csv** - Per-target performance metrics across epochs (target_name, estimate, target, epoch, error, rel_error, abs_error, rel_abs_error, loss)

## Changing Hyperparameters

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

## Important Notes

- **Keep your connection open** - Modal needs to stay connected to download results. Don't close your laptop or let it sleep until you see the local "Weights saved to:" and "Calibration log saved to:" messages.
- Modal clones from GitHub, so local changes must be pushed before they take effect.
