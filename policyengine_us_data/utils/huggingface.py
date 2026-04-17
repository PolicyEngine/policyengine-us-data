"""Thin wrappers over ``huggingface_hub`` for PolicyEngine US data.

Token handling
--------------
Import-time used to hard-``raise`` if ``HUGGING_FACE_TOKEN`` was not
set, which blocked every non-HF workflow that happens to import this
module (docs builds, lightweight CI checks, fully local calibration
per issue #591, plus transitive imports via ``raw_cache`` and
``datasets.sipp.sipp``).

The token is only strictly required for *uploads*. Downloads from a
public repo work without a token, and gated / private downloads fall
back to whatever the user has configured via
``huggingface_hub``'s own credential store.

This module therefore reads the token lazily and raises only from
:func:`upload` and :func:`upload_calibration_artifacts` (the two
functions that genuinely need auth). :func:`download` passes the
token through when present and lets ``huggingface_hub`` handle
auth otherwise.
"""

from huggingface_hub import hf_hub_download, HfApi, CommitOperationAdd
import os


def get_token() -> str | None:
    """Return the HF token from env, or ``None`` if unset.

    Downloads from public repos still work when this returns ``None``.
    """
    return os.environ.get("HUGGING_FACE_TOKEN")


def _require_token(action: str) -> str:
    """Fetch the token or raise with a clear message.

    Called from upload paths that genuinely cannot proceed without a
    token. ``action`` is included in the message so the failure
    points at the operation that needed auth.
    """
    token = get_token()
    if not token:
        raise ValueError(
            "Required environment variable 'HUGGING_FACE_TOKEN' is not set. "
            f"This token is needed to {action}. "
            "Please set the HUGGING_FACE_TOKEN environment variable."
        )
    return token


def download(repo: str, repo_filename: str, local_folder: str, version: str = None):
    hf_hub_download(
        repo_id=repo,
        repo_type="model",
        filename=repo_filename,
        local_dir=local_folder,
        revision=version,
        token=get_token(),
    )


def upload(local_file_path: str, repo: str, repo_file_path: str):
    token = _require_token("upload files to Hugging Face Hub")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=repo_file_path,
        repo_id=repo,
        repo_type="model",
        token=token,
    )


def download_calibration_inputs(
    output_dir: str,
    repo: str = "policyengine/policyengine-us-data",
    version: str = None,
    prefix: str = "",
) -> dict:
    """
    Download calibration inputs from Hugging Face.

    Args:
        output_dir: Local directory to download files to
        repo: Hugging Face repository ID
        version: Optional revision (commit, tag, or branch)
        prefix: Filename prefix for weights/blocks
            (e.g. "national_")

    Returns:
        dict with keys including weights, geography, dataset, and database
        for any artifacts that exist remotely
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    token = get_token()

    # Core inputs needed by both calibration and local area pipelines
    files = {
        "dataset": ("calibration/source_imputed_stratified_extended_cps.h5"),
    }

    paths = {}
    for key, hf_path in files.items():
        hf_hub_download(
            repo_id=repo,
            filename=hf_path,
            local_dir=str(output_path),
            repo_type="model",
            revision=version,
            token=token,
        )
        local_path = output_path / hf_path
        paths[key] = local_path
        print(f"Downloaded {hf_path} to {local_path}")

    # Calibration outputs — required by local area pipeline,
    # but won't exist yet when running calibration from scratch
    optional_files = {
        "weights": f"calibration/{prefix}calibration_weights.npy",
        "geography": f"calibration/{prefix}geography_assignment.npz",
        "checkpoint": f"calibration/{prefix}calibration_weights.checkpoint.pt",
        "run_config": (f"calibration/{prefix}unified_run_config.json"),
    }
    for key, hf_path in optional_files.items():
        try:
            hf_hub_download(
                repo_id=repo,
                filename=hf_path,
                local_dir=str(output_path),
                repo_type="model",
                revision=version,
                token=token,
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

    token = get_token()

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
                token=token,
            )
            local_path = output_path / hf_path
            paths[key] = local_path
            print(f"Downloaded {hf_path} to {local_path}")
        except Exception as e:
            print(f"Skipping {hf_path}: {e}")

    return paths


def upload_calibration_artifacts(
    weights_path: str = None,
    geography_path: str = None,
    checkpoint_path: str = None,
    log_dir: str = None,
    repo: str = "policyengine/policyengine-us-data",
    prefix: str = "",
) -> list:
    """Upload calibration artifacts to HuggingFace in a single commit.

    Args:
        weights_path: Path to calibration_weights.npy
        log_dir: Directory containing log files
            (calibration_log.csv, unified_diagnostics.csv,
             unified_run_config.json)
        repo: HuggingFace repository ID
        prefix: Filename prefix for HF paths (e.g. "national_")

    Returns:
        List of uploaded HF paths
    """
    operations = []

    if weights_path and os.path.exists(weights_path):
        operations.append(
            CommitOperationAdd(
                path_in_repo=(f"calibration/{prefix}calibration_weights.npy"),
                path_or_fileobj=weights_path,
            )
        )

    if geography_path and os.path.exists(geography_path):
        operations.append(
            CommitOperationAdd(
                path_in_repo=(f"calibration/{prefix}geography_assignment.npz"),
                path_or_fileobj=geography_path,
            )
        )

    if checkpoint_path and os.path.exists(checkpoint_path):
        operations.append(
            CommitOperationAdd(
                path_in_repo=(f"calibration/{prefix}calibration_weights.checkpoint.pt"),
                path_or_fileobj=checkpoint_path,
            )
        )

    if log_dir:
        # Upload run config to calibration/ root for artifact validation
        run_config_local = os.path.join(log_dir, f"{prefix}unified_run_config.json")
        if os.path.exists(run_config_local):
            operations.append(
                CommitOperationAdd(
                    path_in_repo=(f"calibration/{prefix}unified_run_config.json"),
                    path_or_fileobj=run_config_local,
                )
            )

        log_files = {
            f"{prefix}calibration_log.csv": (
                f"calibration/logs/{prefix}calibration_log.csv"
            ),
            f"{prefix}unified_diagnostics.csv": (
                f"calibration/logs/{prefix}unified_diagnostics.csv"
            ),
            f"{prefix}unified_run_config.json": (
                f"calibration/logs/{prefix}unified_run_config.json"
            ),
            f"{prefix}validation_results.csv": (
                f"calibration/logs/{prefix}validation_results.csv"
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

    token = _require_token("upload calibration artifacts to Hugging Face Hub")
    api = HfApi()
    api.create_commit(
        token=token,
        repo_id=repo,
        operations=operations,
        repo_type="model",
        commit_message=(f"Upload {len(operations)} calibration artifact(s)"),
    )

    uploaded = [op.path_in_repo for op in operations]
    print(f"Uploaded to HuggingFace: {uploaded}")
    return uploaded
