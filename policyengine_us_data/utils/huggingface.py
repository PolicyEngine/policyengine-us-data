from huggingface_hub import hf_hub_download, login, HfApi
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
        "weights": "calibration/w_district_calibration.npy",
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

    return paths
