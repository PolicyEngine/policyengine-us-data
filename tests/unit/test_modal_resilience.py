import json

import pytest

from modal_app.resilience import (
    ensure_resume_sha_compatible,
    reconcile_run_dir_fingerprint,
)


def test_resume_requires_same_sha():
    with pytest.raises(RuntimeError, match="Start a fresh run instead"):
        ensure_resume_sha_compatible(
            branch="fix/pipeline-resilience",
            run_sha="0123456789abcdef",
            current_sha="fedcba9876543210",
        )


def test_resume_allows_same_sha():
    result = ensure_resume_sha_compatible(
        branch="fix/pipeline-resilience",
        run_sha="0123456789abcdef",
        current_sha="0123456789abcdef",
    )
    assert result is True


def test_resume_force_allows_mismatched_sha():
    result = ensure_resume_sha_compatible(
        branch="fix/pipeline-resilience",
        run_sha="0123456789abcdef",
        current_sha="fedcba9876543210",
        force=True,
    )
    assert result is False


def test_resume_force_with_matching_sha():
    result = ensure_resume_sha_compatible(
        branch="fix/pipeline-resilience",
        run_sha="0123456789abcdef",
        current_sha="0123456789abcdef",
        force=True,
    )
    assert result is True


def test_reconcile_run_dir_resumes_matching_fingerprint(tmp_path):
    run_dir = tmp_path / "1.2.3_abc12345_20260407_120000"
    run_dir.mkdir()
    (run_dir / "states").mkdir()
    (run_dir / "states" / "CA.h5").write_text("h5")
    (run_dir / "fingerprint.json").write_text(json.dumps({"fingerprint": "abc123"}))

    action = reconcile_run_dir_fingerprint(run_dir, "abc123")

    assert action == "resume"
    assert (run_dir / "states" / "CA.h5").exists()
    assert json.loads((run_dir / "fingerprint.json").read_text()) == {
        "fingerprint": "abc123"
    }


def test_reconcile_run_dir_rejects_changed_fingerprint_with_h5s(tmp_path):
    run_dir = tmp_path / "1.2.3_abc12345_20260407_120000"
    run_dir.mkdir()
    (run_dir / "states").mkdir()
    (run_dir / "states" / "CA.h5").write_text("stale")
    (run_dir / "fingerprint.json").write_text(json.dumps({"fingerprint": "oldfp"}))

    with pytest.raises(RuntimeError, match="Fingerprint mismatch"):
        reconcile_run_dir_fingerprint(run_dir, "newfp")

    assert (run_dir / "states" / "CA.h5").exists()
    assert json.loads((run_dir / "fingerprint.json").read_text()) == {
        "fingerprint": "oldfp"
    }


def test_reconcile_run_dir_rejects_missing_fingerprint_with_h5s(tmp_path):
    run_dir = tmp_path / "1.2.3_abc12345_20260407_120000"
    run_dir.mkdir()
    (run_dir / "states").mkdir()
    (run_dir / "states" / "CA.h5").write_text("stale")

    with pytest.raises(RuntimeError, match="Missing fingerprint metadata"):
        reconcile_run_dir_fingerprint(run_dir, "newfp")

    assert (run_dir / "states" / "CA.h5").exists()
    assert not (run_dir / "fingerprint.json").exists()


def test_reconcile_run_dir_clears_empty_stale_directory(tmp_path):
    run_dir = tmp_path / "1.2.3_abc12345_20260407_120000"
    run_dir.mkdir()
    (run_dir / "scratch.txt").write_text("stale")
    (run_dir / "fingerprint.json").write_text(json.dumps({"fingerprint": "oldfp"}))

    action = reconcile_run_dir_fingerprint(run_dir, "newfp")

    assert action == "initialized"
    assert not (run_dir / "scratch.txt").exists()
    assert json.loads((run_dir / "fingerprint.json").read_text()) == {
        "fingerprint": "newfp"
    }
