import hashlib
from pathlib import Path

import pytest

from policyengine_us_data.storage import download_prerequisites


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_download_prerequisite_copies_artifact_and_checks_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = b"expected prerequisite content"
    cached_path = tmp_path / "hf-cache" / "remote.csv.gz"
    cached_path.parent.mkdir()
    cached_path.write_bytes(content)
    storage_folder = tmp_path / "storage"
    artifact = download_prerequisites.PrerequisiteArtifact(
        repo="policyengine/example",
        path_in_repo="prerequisites/example.csv.gz",
        local_filename="example.csv.gz",
        revision="abc123",
        sha256=_sha256(content),
    )
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return cached_path

    monkeypatch.setattr(
        download_prerequisites,
        "hf_hub_download",
        fake_hf_hub_download,
    )
    monkeypatch.setattr(download_prerequisites, "get_token", lambda: "token")

    destination = download_prerequisites.download_prerequisite(
        artifact,
        storage_folder=storage_folder,
    )

    assert destination == storage_folder / "example.csv.gz"
    assert destination.read_bytes() == content
    assert calls == [
        {
            "repo_id": "policyengine/example",
            "repo_type": "model",
            "filename": "prerequisites/example.csv.gz",
            "revision": "abc123",
            "token": "token",
        }
    ]


def test_download_prerequisite_rejects_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cached_path = tmp_path / "hf-cache" / "remote.csv.gz"
    cached_path.parent.mkdir()
    cached_path.write_bytes(b"actual content")
    artifact = download_prerequisites.PrerequisiteArtifact(
        repo="policyengine/example",
        path_in_repo="prerequisites/example.csv.gz",
        local_filename="example.csv.gz",
        sha256=_sha256(b"different content"),
    )

    monkeypatch.setattr(
        download_prerequisites,
        "hf_hub_download",
        lambda **kwargs: cached_path,
    )

    with pytest.raises(ValueError, match="SHA256"):
        download_prerequisites.download_prerequisite(
            artifact,
            storage_folder=tmp_path / "storage",
        )


def test_geography_prerequisites_are_pinned_and_hash_checked() -> None:
    geography_artifacts = [
        artifact
        for artifact in download_prerequisites.PREREQUISITE_ARTIFACTS
        if artifact.repo == download_prerequisites.GEOGRAPHY_REPO
    ]

    assert len(geography_artifacts) == 2
    for artifact in geography_artifacts:
        assert artifact.revision == download_prerequisites.GEOGRAPHY_REVISION
        assert artifact.path_in_repo.startswith("prerequisites/geography/")
        assert artifact.sha256 is not None
        assert len(artifact.sha256) == 64
