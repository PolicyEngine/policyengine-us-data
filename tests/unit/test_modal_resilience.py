import json

import pytest

from modal_app.resilience import (
    ensure_resume_sha_compatible,
    reconcile_version_dir_fingerprint,
)


def test_resume_requires_same_sha():
    with pytest.raises(RuntimeError, match="Start a fresh run instead"):
        ensure_resume_sha_compatible(
            branch="fix/pipeline-resilience",
            run_sha="0123456789abcdef",
            current_sha="fedcba9876543210",
        )


def test_resume_allows_same_sha():
    ensure_resume_sha_compatible(
        branch="fix/pipeline-resilience",
        run_sha="0123456789abcdef",
        current_sha="0123456789abcdef",
    )


def test_reconcile_version_dir_resumes_matching_fingerprint(tmp_path):
    version_dir = tmp_path / "1.2.3"
    version_dir.mkdir()
    (version_dir / "states").mkdir()
    (version_dir / "states" / "CA.h5").write_text("h5")
    (version_dir / "fingerprint.json").write_text(json.dumps({"fingerprint": "abc123"}))

    action = reconcile_version_dir_fingerprint(version_dir, "abc123")

    assert action == "resume"
    assert (version_dir / "states" / "CA.h5").exists()
    assert json.loads((version_dir / "fingerprint.json").read_text()) == {
        "fingerprint": "abc123"
    }


def test_reconcile_version_dir_rejects_changed_fingerprint_with_h5s(tmp_path):
    version_dir = tmp_path / "1.2.3"
    version_dir.mkdir()
    (version_dir / "states").mkdir()
    (version_dir / "states" / "CA.h5").write_text("stale")
    (version_dir / "fingerprint.json").write_text(json.dumps({"fingerprint": "oldfp"}))

    with pytest.raises(RuntimeError, match="Fingerprint mismatch"):
        reconcile_version_dir_fingerprint(version_dir, "newfp")

    assert (version_dir / "states" / "CA.h5").exists()
    assert json.loads((version_dir / "fingerprint.json").read_text()) == {
        "fingerprint": "oldfp"
    }


def test_reconcile_version_dir_rejects_missing_fingerprint_with_h5s(tmp_path):
    version_dir = tmp_path / "1.2.3"
    version_dir.mkdir()
    (version_dir / "states").mkdir()
    (version_dir / "states" / "CA.h5").write_text("stale")

    with pytest.raises(RuntimeError, match="Missing fingerprint metadata"):
        reconcile_version_dir_fingerprint(version_dir, "newfp")

    assert (version_dir / "states" / "CA.h5").exists()
    assert not (version_dir / "fingerprint.json").exists()


def test_reconcile_version_dir_clears_empty_stale_directory(tmp_path):
    version_dir = tmp_path / "1.2.3"
    version_dir.mkdir()
    (version_dir / "scratch.txt").write_text("stale")
    (version_dir / "fingerprint.json").write_text(json.dumps({"fingerprint": "oldfp"}))

    action = reconcile_version_dir_fingerprint(version_dir, "newfp")

    assert action == "initialized"
    assert not (version_dir / "scratch.txt").exists()
    assert json.loads((version_dir / "fingerprint.json").read_text()) == {
        "fingerprint": "newfp"
    }
