import os
import requests
from tqdm import tqdm
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
    if response.status_code != 200:
        raise ValueError(
            f"Invalid response code {response.status_code} for url {url}."
        )
    return response.json()["id"]


def download(
    org: str, repo: str, release_tag: str, file_name: str, file_path: str
) -> bytes:

    url = get_asset_url(org, repo, release_tag, file_name)

    response = requests.get(
        url,
        headers={
            "Accept": "application/octet-stream",
            **auth_headers,
        },
    )

    if response.status_code != 200:
        raise ValueError(
            f"Invalid response code {response.status_code} for url {url}."
        )

    with open(file_path, "wb") as f:
        f.write(response.content)


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
    release_id = get_release_id(org, repo, release_tag)
    url = f"https://uploads.github.com/repos/{org}/{repo}/releases/{release_id}/assets?name={file_name}"

    file_size = os.path.getsize(file_path)
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/octet-stream",
        **auth_headers,
    }

    session = create_session_with_retries()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as f:
                with tqdm(total=file_size, unit="B", unit_scale=True) as pbar:
                    response = session.post(
                        url,
                        headers=headers,
                        data=f,
                        stream=True,
                        hooks=dict(
                            response=lambda r, *args, **kwargs: pbar.update(
                                len(r.content)
                            )
                        ),
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
