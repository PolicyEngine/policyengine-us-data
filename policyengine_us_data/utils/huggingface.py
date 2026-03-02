from huggingface_hub import hf_hub_download, login, HfApi, CommitOperationAdd
import os

TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
if not TOKEN:
    raise ValueError(
        "Required environment variable 'HUGGING_FACE_TOKEN' is not set. "
        "This token is needed to download files from Hugging Face Hub. "
        "Please set the HUGGING_FACE_TOKEN environment variable."
    )


def download(
    repo: str, repo_filename: str, local_folder: str, version: str = None
):

    hf_hub_download(
        repo_id=repo,
        repo_type="model",
        filename=repo_filename,
        local_dir=local_folder,
        revision=version,
        token=TOKEN,
    )


def upload(local_file_path: str, repo: str, repo_file_path: str):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=repo_file_path,
        repo_id=repo,
        repo_type="model",
        token=TOKEN,
    )


def download_calibration_inputs(
    output_dir: str,
    repo: str = "policyengine/policyengine-us-data",
    version: str = None,
) -> dict:
    """
    Download calibration inputs from Hugging Face.

    Args:
        output_dir: Local directory to download files to
        repo: Hugging Face repository ID
        version: Optional revision (commit, tag, or branch)

    Returns:
        dict with keys 'weights', 'dataset', 'database' mapping to local paths
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = {
        "weights": "calibration/calibration_weights.npy",
        "dataset": "calibration/stratified_extended_cps.h5",
        "database": "calibration/policy_data.db",
    }

    paths = {}
    for key, hf_path in files.items():
        hf_hub_download(
            repo_id=repo,
            filename=hf_path,
            local_dir=str(output_path),
            repo_type="model",
            revision=version,
            token=TOKEN,
        )
        # hf_hub_download preserves directory structure
        local_path = output_path / hf_path
        paths[key] = local_path
        print(f"Downloaded {hf_path} to {local_path}")

    optional_files = {
        "blocks": "calibration/stacked_blocks.npy",
        "source_imputed_dataset": (
            "calibration/" "source_imputed_stratified_extended_cps.h5"
        ),
    }
    for key, hf_path in optional_files.items():
        try:
            hf_hub_download(
                repo_id=repo,
                filename=hf_path,
                local_dir=str(output_path),
                repo_type="model",
                revision=version,
                token=TOKEN,
            )
            local_path = output_path / hf_path
            paths[key] = local_path
            print(f"Downloaded {hf_path} to {local_path}")
        except Exception as e:
            print(f"Skipping optional {hf_path}: {e}")

    return paths


def download_calibration_logs(
    output_dir: str,
    repo: str = "policyengine/policyengine-us-data",
    version: str = None,
) -> dict:
    """
    Download calibration logs from Hugging Face.

    Args:
        output_dir: Local directory to download files to
        repo: Hugging Face repository ID
        version: Optional revision (commit, tag, or branch)

    Returns:
        dict mapping artifact names to local paths
        (only includes files that exist on HF)
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = {
        "calibration_log": "calibration/logs/calibration_log.csv",
        "diagnostics": "calibration/logs/unified_diagnostics.csv",
        "config": "calibration/logs/unified_run_config.json",
    }

    paths = {}
    for key, hf_path in files.items():
        try:
            hf_hub_download(
                repo_id=repo,
                filename=hf_path,
                local_dir=str(output_path),
                repo_type="model",
                revision=version,
                token=TOKEN,
            )
            local_path = output_path / hf_path
            paths[key] = local_path
            print(f"Downloaded {hf_path} to {local_path}")
        except Exception as e:
            print(f"Skipping {hf_path}: {e}")

    return paths


def upload_calibration_artifacts(
    weights_path: str = None,
    blocks_path: str = None,
    log_dir: str = None,
    repo: str = "policyengine/policyengine-us-data",
) -> list:
    """Upload calibration artifacts to HuggingFace in a single commit.

    Args:
        weights_path: Path to calibration_weights.npy
        blocks_path: Path to stacked_blocks.npy
        log_dir: Directory containing log files
            (calibration_log.csv, unified_diagnostics.csv,
             unified_run_config.json)
        repo: HuggingFace repository ID

    Returns:
        List of uploaded HF paths
    """
    operations = []

    if weights_path and os.path.exists(weights_path):
        operations.append(
            CommitOperationAdd(
                path_in_repo="calibration/calibration_weights.npy",
                path_or_fileobj=weights_path,
            )
        )

    if blocks_path and os.path.exists(blocks_path):
        operations.append(
            CommitOperationAdd(
                path_in_repo="calibration/stacked_blocks.npy",
                path_or_fileobj=blocks_path,
            )
        )

    if log_dir:
        log_files = {
            "calibration_log.csv": ("calibration/logs/calibration_log.csv"),
            "unified_diagnostics.csv": (
                "calibration/logs/unified_diagnostics.csv"
            ),
            "unified_run_config.json": (
                "calibration/logs/unified_run_config.json"
            ),
            "validation_results.csv": (
                "calibration/logs/validation_results.csv"
            ),
        }
        for filename, hf_path in log_files.items():
            local_path = os.path.join(log_dir, filename)
            if os.path.exists(local_path):
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=hf_path,
                        path_or_fileobj=local_path,
                    )
                )

    if not operations:
        print("No calibration artifacts to upload.")
        return []

    api = HfApi()
    api.create_commit(
        token=TOKEN,
        repo_id=repo,
        operations=operations,
        repo_type="model",
        commit_message=(f"Upload {len(operations)} calibration artifact(s)"),
    )

    uploaded = [op.path_in_repo for op in operations]
    print(f"Uploaded to HuggingFace: {uploaded}")
    return uploaded
