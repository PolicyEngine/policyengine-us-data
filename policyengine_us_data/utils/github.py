import os
import requests

auth_headers = {
    "Authorization": f"token {os.environ.get('POLICYENGINE_US_DATA_GITHUB_TOKEN')}",
}
print(auth_headers["Authorization"][:10] + "***")


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


def upload(
    org: str, repo: str, release_tag: str, file_name: str, file_path: str
) -> bytes:
    release_id = get_release_id(org, repo, release_tag)
    url = f"https://uploads.github.com/repos/{org}/{repo}/releases/{release_id}/assets?name={file_name}"
    response = requests.post(
        url,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/octet-stream",
            **auth_headers,
        },
        data=open(file_path, "rb").read(),
    )

    if response.status_code != 201:
        raise ValueError(
            f"Invalid response code {response.status_code} for url {url}. Received: {response.text}"
        )

    return response.json()


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
