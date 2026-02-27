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
| `--upload` | `False` | Upload weights, blocks, and logs to HuggingFace |
| `--upload-logs` | `False` | Alias for `--upload` (backwards compat) |
| `--trigger-publish` | `False` | Fire `repository_dispatch` to trigger the Publish workflow |
| `--target-config` | (none) | Target configuration name |
| `--beta` | (none) | L0 relaxation parameter |
| `--lambda-l0` | (none) | L0 penalty weight |
| `--lambda-l2` | (none) | L2 penalty weight |
| `--learning-rate` | (none) | Optimizer learning rate |
| `--package-path` | (none) | Local path to a pre-built calibration package |
| `--package-volume` | `False` | Use package from Modal volume instead |
| `--county-level` | `False` | Include county-level targets |
| `--workers` | `1` | Number of parallel workers |

### Examples

Fit weights and upload everything to HF:
```bash
modal run modal_app/remote_calibration_runner.py \
  --branch main --epochs 200 --gpu A100-80GB --upload
```

Fit, upload, and trigger the publish workflow:
```bash
modal run modal_app/remote_calibration_runner.py \
  --gpu A100-80GB --epochs 200 --upload --trigger-publish
```

## Output Files

Every run produces these local files (whichever the calibration script emits):

- **calibration_weights.npy** — Fitted household weights
- **unified_diagnostics.csv** — Final per-target diagnostics
- **calibration_log.csv** — Per-target metrics across epochs (requires `--log-freq`)
- **unified_run_config.json** — Run configuration and summary stats
- **stacked_blocks.npy** — Census block assignments for stacked records

## Artifact Upload to HuggingFace

The `--upload` flag uploads all artifacts to HuggingFace in a single atomic
commit after writing them locally:

| Local file | HF path |
|------------|---------|
| `calibration_weights.npy` | `calibration/calibration_weights.npy` |
| `stacked_blocks.npy` | `calibration/stacked_blocks.npy` |
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
- `--upload` requires the `HUGGING_FACE_TOKEN` environment variable
  to be set locally (not just as a Modal secret).
- `--trigger-publish` requires `GITHUB_TOKEN` or
  `POLICYENGINE_US_DATA_GITHUB_TOKEN` set locally.
