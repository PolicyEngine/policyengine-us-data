"""Download build prerequisites that are not vendored in the package."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.huggingface import get_token

HF_REPO_TYPE = "model"
PRIVATE_PUF_REPO = "policyengine/irs-soi-puf"
GEOGRAPHY_REPO = "policyengine/policyengine-us-data"
GEOGRAPHY_REVISION = "afe8d64cd1b66d35a5d6be11abe12bbc72b2e44b"


@dataclass(frozen=True)
class PrerequisiteArtifact:
    """Hugging Face artifact required before running the data build."""

    repo: str
    path_in_repo: str
    local_filename: str
    revision: str | None = None
    sha256: str | None = None


PREREQUISITE_ARTIFACTS = (
    PrerequisiteArtifact(
        repo=PRIVATE_PUF_REPO,
        path_in_repo="puf_2015.csv",
        local_filename="puf_2015.csv",
    ),
    PrerequisiteArtifact(
        repo=PRIVATE_PUF_REPO,
        path_in_repo="demographics_2015.csv",
        local_filename="demographics_2015.csv",
    ),
    PrerequisiteArtifact(
        repo=PRIVATE_PUF_REPO,
        path_in_repo="np2023_d5_mid.csv",
        local_filename="np2023_d5_mid.csv",
    ),
    PrerequisiteArtifact(
        repo=GEOGRAPHY_REPO,
        path_in_repo="prerequisites/geography/block_cd_distributions.csv.gz",
        local_filename="block_cd_distributions.csv.gz",
        revision=GEOGRAPHY_REVISION,
        sha256="0932ddbf95f454ddcf299d4aa8e3d6919ded9c401e2e7d2cc769466f7fade9bd",
    ),
    PrerequisiteArtifact(
        repo=GEOGRAPHY_REPO,
        path_in_repo="prerequisites/geography/block_crosswalk.csv.gz",
        local_filename="block_crosswalk.csv.gz",
        revision=GEOGRAPHY_REVISION,
        sha256="cb729f21ef59ea44c0f49aa3c2369f884419765a2c6bd32dc18857952cb8ed4f",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_prerequisite(
    artifact: PrerequisiteArtifact,
    *,
    storage_folder: Path = STORAGE_FOLDER,
) -> Path:
    """Download one prerequisite artifact and return its local storage path."""
    source_path = Path(
        hf_hub_download(
            repo_id=artifact.repo,
            repo_type=HF_REPO_TYPE,
            filename=artifact.path_in_repo,
            revision=artifact.revision,
            token=get_token(),
        )
    )
    destination = storage_folder / artifact.local_filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source_path.resolve() != destination.resolve():
        shutil.copyfile(source_path, destination)

    if artifact.sha256 is not None:
        actual = _sha256(destination)
        if actual != artifact.sha256:
            raise ValueError(
                f"Downloaded {artifact.path_in_repo} from {artifact.repo} "
                f"with SHA256 {actual}, expected {artifact.sha256}."
            )

    return destination


def download_prerequisites(
    artifacts: tuple[PrerequisiteArtifact, ...] = PREREQUISITE_ARTIFACTS,
    *,
    storage_folder: Path = STORAGE_FOLDER,
) -> dict[str, Path]:
    """Download all build prerequisites into ``storage_folder``."""
    return {
        artifact.local_filename: download_prerequisite(
            artifact,
            storage_folder=storage_folder,
        )
        for artifact in artifacts
    }


def main() -> None:
    download_prerequisites()


if __name__ == "__main__":
    main()
