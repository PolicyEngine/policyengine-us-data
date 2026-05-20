from pathlib import Path

from policyengine_us_data.build_datasets import (
    CheckpointReuseSummary,
    CheckpointStore,
)


def _store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(root=tmp_path, branch="stage-1", commit_sha="abc123")


def test_checkpoint_store_scopes_paths_by_branch_and_commit(tmp_path):
    store = _store(tmp_path)

    assert store.checkpoint_path("policyengine_us_data/storage/cps_2024.h5") == (
        tmp_path / "stage-1" / "abc123" / "cps_2024.h5"
    )


def test_checkpoint_store_requires_all_outputs_for_restore(tmp_path, monkeypatch):
    store = _store(tmp_path)
    checkpoint = store.checkpoint_path("a.txt")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("cached")
    monkeypatch.chdir(tmp_path)

    assert store.restore_all_outputs(("a.txt", "b.txt")) is False
    assert not (tmp_path / "a.txt").exists()

    store.checkpoint_path("b.txt").write_text("cached-b")
    assert store.restore_all_outputs(("a.txt", "b.txt")) is True
    assert (tmp_path / "a.txt").read_text() == "cached"
    assert (tmp_path / "b.txt").read_text() == "cached-b"


def test_missing_or_empty_checkpoint_invalidates_reuse(tmp_path):
    store = _store(tmp_path)
    empty = store.checkpoint_path("empty.txt")
    empty.parent.mkdir(parents=True)
    empty.write_text("")

    missing_decision = store.decision_for("missing.txt")
    empty_decision = store.decision_for("empty.txt")

    assert missing_decision.action == "recompute"
    assert missing_decision.reason == "missing"
    assert empty_decision.action == "recompute"
    assert empty_decision.reason == "empty"
    assert store.all_outputs_reusable(("missing.txt", "empty.txt")) is False


def test_checkpoint_summary_matches_prior_counters(tmp_path):
    store = _store(tmp_path)
    checkpoint = store.checkpoint_path("valid.txt")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("cached")
    decisions = store.decisions_for(("valid.txt", "missing.txt"))

    reused_summary = CheckpointReuseSummary.from_decisions(
        decisions,
        recomputed=False,
    )
    recomputed_summary = CheckpointReuseSummary.from_decisions(
        decisions,
        recomputed=True,
    )

    assert reused_summary.to_dict() == {
        "expected_outputs": 2,
        "valid_reused_outputs": 1,
        "recomputed_outputs": 0,
        "invalid_outputs": 1,
    }
    assert recomputed_summary.to_dict() == {
        "expected_outputs": 2,
        "valid_reused_outputs": 0,
        "recomputed_outputs": 2,
        "invalid_outputs": 1,
    }


def test_checkpoint_cleanup_removes_only_branch_scope(tmp_path):
    store = _store(tmp_path)
    (tmp_path / "stage-1" / "abc123").mkdir(parents=True)
    (tmp_path / "other-branch" / "abc123").mkdir(parents=True)

    assert store.cleanup_branch() is True

    assert not (tmp_path / "stage-1").exists()
    assert (tmp_path / "other-branch" / "abc123").exists()


def test_checkpoint_cleanup_removes_only_other_commits(tmp_path):
    store = _store(tmp_path)
    current = tmp_path / "stage-1" / "abc123"
    stale = tmp_path / "stage-1" / "old"
    current.mkdir(parents=True)
    stale.mkdir(parents=True)

    removed = store.cleanup_other_commits()

    assert removed == (stale,)
    assert current.exists()
    assert not stale.exists()
