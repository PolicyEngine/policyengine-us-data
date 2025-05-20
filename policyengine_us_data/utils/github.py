import os
import requests
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time

auth_headers = {
    "Authorization": f"token {os.environ.get('POLICYENGINE_US_DATA_GITHUB_TOKEN')}",
}


def get_asset_url(
    org: str, repo: str, release_tag: str, file_name: str
) -> str:
    url = f"https://api.github.com/repos/{org}/{repo}/releases/tags/{release_tag}"
    response = requests.get(url, headers=auth_headers)
    if response.status_code != 200:
        raise ValueError(
            f"Invalid response code {response.status_code} for url {url}."
        )
    assets = response.json()["assets"]
    for asset in assets:
        if asset["name"] == file_name:
            return asset["url"]
    else:
        raise ValueError(
            f"File {file_name} not found in release {release_tag} of {org}/{repo}."
        )


def get_release_id(org: str, repo: str, release_tag: str) -> int:
    url = f"https://api.github.com/repos/{org}/{repo}/releases/tags/{release_tag}"
    response = requests.get(url, headers=auth_headers)
    if response.status_code == 404:
        raise ValueError(f"Release {release_tag} not found in {org}/{repo}.")
    elif response.status_code != 200:
        raise ValueError(
            f"Invalid response code {response.status_code} for url {url}."
        )
    return response.json()["id"]


def get_all_assets(org: str, repo: str, release_id: int) -> list:
    url = f"https://api.github.com/repos/{org}/{repo}/releases/{release_id}/assets"
    response = requests.get(url, headers=auth_headers)
    if response.status_code != 200:
        raise ValueError(
            f"Invalid response code {response.status_code} for url {url}."
        )
    return response.json()


def get_asset_id(
    org: str, repo: str, release_id: int, file_name: str
) -> int | None:

    # Get all assets in the release (schema: array of JSON objects)
    assets: dict = get_all_assets(org, repo, release_id)

    # Iterate over to see if the file is already released
    for asset in assets:
        if asset["name"] == file_name:
            return asset["id"]

    return None


def delete_asset(org: str, repo: str, asset_id: int):
    url = (
        f"https://api.github.com/repos/{org}/{repo}/releases/assets/{asset_id}"
    )
    headers = {
        "Accept": "application/vnd.github.v3+json",
        **auth_headers,
    }

    response = requests.delete(url, headers=headers)
    if response.status_code != 204:
        raise ValueError(
            f"Invalid response code {response.status_code} for url {url}."
        )


def download(
    org: str, repo: str, release_tag: str, file_name: str, file_path: str
) -> bytes:

    url = get_asset_url(org, repo, release_tag, file_name)

    try:

        response = requests.get(
            url,
            stream=True,
            headers={
                "Accept": "application/octet-stream",
                **auth_headers,
            },
        )

        file_size = int(response.headers.get("Content-Length", 0))

        with open(file_path, "wb") as f:
            with tqdm(
                total=file_size, unit="B", unit_scale=True, unit_divisor=1024
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    pbar.update(len(chunk))

    except Exception as e:
        raise ValueError(f"Failed to download file: {str(e)}")


def create_session_with_retries():
    session = requests.Session()
    retries = Retry(
        total=5, backoff_factor=1, status_forcelist=[502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def upload(
    org: str, repo: str, release_tag: str, file_name: str, file_path: str
) -> bytes:

    # Pull release ID
    release_id = get_release_id(org, repo, release_tag)

    # Fetch asset ID if the file is already released, else None
    asset_id = get_asset_id(org, repo, release_id, file_name)

    try:

        temp_file_path = "asset_fallback.tmp"

        if asset_id is not None:
            # If the asset already exists, download it. There's unfortunately
            # no native transaction feature in GitHub releases, so we'll download
            # in case our subsequent delete-upload fails

            print(
                f"Asset {file_name} already exists in release {release_tag}. Skipping."
            )

            return

        # Now, upload the asset
        print(f"Uploading {file_name} to release {release_tag}...")
        create_asset(org, repo, release_id, file_name, file_path)

        # If the upload was successful, delete the temporary file
        if os.path.exists(temp_file_path):
            print(f"Deleting backup file...")
            os.remove(temp_file_path)

    except Exception as e:
        print(f"Error uploading {file_name}: {str(e)}")

        if os.path.exists(temp_file_path):
            print(f"Restoring backup file...")
            create_asset(org, repo, release_id, file_name, temp_file_path)
        raise e


def create_asset(
    org: str, repo: str, release_id: int, file_name: str, file_path: str
):

    url = f"https://uploads.github.com/repos/{org}/{repo}/releases/{release_id}/assets?name={file_name}"

    file_size = os.path.getsize(file_path)
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/octet-stream",
        "Content-Length": str(file_size),
        **auth_headers,
    }

    session = create_session_with_retries()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as f:
                with tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    wrapped_file = CallbackIOWrapper(pbar.update, f, "read")
                    response = session.post(
                        url,
                        headers=headers,
                        data=wrapped_file,
                        timeout=300,  # 5 minutes timeout
                    )

            if response.status_code == 201:
                return response.json()
            else:
                print(
                    f"Attempt {attempt + 1} failed with status code {response.status_code}. Response: {response.text}"
                )

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")

        if attempt < max_retries - 1:
            wait_time = (
                attempt + 1
            ) * 60  # Wait 1 minute, then 2 minutes, then 3 minutes
            print(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)

    raise ValueError(f"Failed to upload file after {max_retries} attempts.")


def set_pr_auto_review_comment(text: str):
    # On a pull request, set a review comment with the given text.

    pr_number = os.environ["GITHUB_PR_NUMBER"]

    url = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/pulls/{pr_number}/reviews"

    response = requests.post(
        url,
        headers=auth_headers,
        json={
            "body": text,
            "event": "COMMENT",
        },
    )

    if response.status_code != 200:
        raise ValueError(
            f"Invalid response code {response.status_code} for url {url}. Received: {response.text}"
        )
