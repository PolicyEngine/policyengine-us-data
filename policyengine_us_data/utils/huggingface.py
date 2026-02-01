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
