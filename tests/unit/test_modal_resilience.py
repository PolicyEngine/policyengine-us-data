import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import pytest


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_resilience_module(monkeypatch):
    package = types.ModuleType("policyengine_us_data")
    package.__path__ = []
    calibration_package = types.ModuleType("policyengine_us_data.calibration")
    calibration_package.__path__ = []
    local_h5_package = types.ModuleType("policyengine_us_data.calibration.local_h5")
    local_h5_package.__path__ = []

    monkeypatch.setitem(sys.modules, "policyengine_us_data", package)
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration",
        calibration_package,
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5",
        local_h5_package,
    )

    @dataclass(frozen=True)
    class FakeFingerprintRecord:
        schema_version: str
        algorithm: str
        digest: str
        components: object | None = None
        inputs: dict = None

        def to_dict(self):
            return {
                "fingerprint": self.digest,
                "digest": self.digest,
                "schema_version": self.schema_version,
                "algorithm": self.algorithm,
            }

    class FakeFingerprintService:
        ALGORITHM = "sha256-truncated-16"

        def legacy_record(self, digest: str):
            return FakeFingerprintRecord(
                schema_version="legacy",
                algorithm=self.ALGORITHM,
                digest=str(digest),
            )

        def write_record(self, path, record):
            Path(path).write_text(json.dumps(record.to_dict()))

        def read_record(self, path):
            payload = json.loads(Path(path).read_text())
            return FakeFingerprintRecord(
                schema_version=payload.get("schema_version", "legacy"),
                algorithm=payload.get("algorithm", self.ALGORITHM),
                digest=payload.get("digest") or payload.get("fingerprint"),
            )

        def matches(self, stored, current):
            return stored.digest == current.digest

    fingerprinting = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.fingerprinting"
    )
    fingerprinting.FingerprintRecord = FakeFingerprintRecord
    fingerprinting.FingerprintService = FakeFingerprintService
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.fingerprinting",
        fingerprinting,
    )

    module_path = Path(__file__).resolve().parents[2] / "modal_app" / "resilience.py"
    module = _load_module("resilience_under_test", module_path)
    return module, FakeFingerprintRecord


def test_resume_requires_same_sha(monkeypatch):
    resilience, _ = _load_resilience_module(monkeypatch)

    with pytest.raises(RuntimeError, match="Start a fresh run instead"):
        resilience.ensure_resume_sha_compatible(
            branch="fix/pipeline-resilience",
            run_sha="0123456789abcdef",
            current_sha="fedcba9876543210",
        )


def test_resume_allows_same_sha(monkeypatch):
    resilience, _ = _load_resilience_module(monkeypatch)

    result = resilience.ensure_resume_sha_compatible(
        branch="fix/pipeline-resilience",
        run_sha="0123456789abcdef",
        current_sha="0123456789abcdef",
    )
    assert result is True


def test_resume_force_allows_mismatched_sha(monkeypatch):
    resilience, _ = _load_resilience_module(monkeypatch)

    result = resilience.ensure_resume_sha_compatible(
        branch="fix/pipeline-resilience",
        run_sha="0123456789abcdef",
        current_sha="fedcba9876543210",
        force=True,
    )
    assert result is False


def test_resume_force_with_matching_sha(monkeypatch):
    resilience, _ = _load_resilience_module(monkeypatch)

    result = resilience.ensure_resume_sha_compatible(
        branch="fix/pipeline-resilience",
        run_sha="0123456789abcdef",
        current_sha="0123456789abcdef",
        force=True,
    )
    assert result is True


def test_reconcile_run_dir_resumes_matching_legacy_fingerprint(monkeypatch, tmp_path):
    resilience, _ = _load_resilience_module(monkeypatch)
    run_dir = tmp_path / "1.2.3_abc12345_20260407_120000"
    run_dir.mkdir()
    (run_dir / "states").mkdir()
    (run_dir / "states" / "CA.h5").write_text("h5")
    (run_dir / "fingerprint.json").write_text(json.dumps({"fingerprint": "abc123"}))

    action = resilience.reconcile_run_dir_fingerprint(run_dir, "abc123")

    assert action == "resume"
    assert (run_dir / "states" / "CA.h5").exists()
    assert json.loads((run_dir / "fingerprint.json").read_text()) == {
        "fingerprint": "abc123"
    }


def test_reconcile_run_dir_rejects_changed_fingerprint_with_h5s(monkeypatch, tmp_path):
    resilience, _ = _load_resilience_module(monkeypatch)
    run_dir = tmp_path / "1.2.3_abc12345_20260407_120000"
    run_dir.mkdir()
    (run_dir / "states").mkdir()
    (run_dir / "states" / "CA.h5").write_text("stale")
    (run_dir / "fingerprint.json").write_text(json.dumps({"fingerprint": "oldfp"}))

    with pytest.raises(RuntimeError, match="Fingerprint mismatch"):
        resilience.reconcile_run_dir_fingerprint(run_dir, "newfp")

    assert (run_dir / "states" / "CA.h5").exists()
    assert json.loads((run_dir / "fingerprint.json").read_text()) == {
        "fingerprint": "oldfp"
    }


def test_reconcile_run_dir_rejects_missing_fingerprint_with_h5s(monkeypatch, tmp_path):
    resilience, _ = _load_resilience_module(monkeypatch)
    run_dir = tmp_path / "1.2.3_abc12345_20260407_120000"
    run_dir.mkdir()
    (run_dir / "states").mkdir()
    (run_dir / "states" / "CA.h5").write_text("stale")

    with pytest.raises(RuntimeError, match="Missing fingerprint metadata"):
        resilience.reconcile_run_dir_fingerprint(run_dir, "newfp")

    assert (run_dir / "states" / "CA.h5").exists()
    assert not (run_dir / "fingerprint.json").exists()


def test_reconcile_run_dir_clears_empty_stale_directory(monkeypatch, tmp_path):
    resilience, _ = _load_resilience_module(monkeypatch)
    run_dir = tmp_path / "1.2.3_abc12345_20260407_120000"
    run_dir.mkdir()
    (run_dir / "scratch.txt").write_text("stale")
    (run_dir / "fingerprint.json").write_text(json.dumps({"fingerprint": "oldfp"}))

    action = resilience.reconcile_run_dir_fingerprint(run_dir, "newfp")

    assert action == "initialized"
    assert not (run_dir / "scratch.txt").exists()
    stored = json.loads((run_dir / "fingerprint.json").read_text())
    assert stored["fingerprint"] == "newfp"
    assert stored["digest"] == "newfp"
    assert stored["schema_version"] == "legacy"
    assert stored["algorithm"] == "sha256-truncated-16"


def test_reconcile_run_dir_accepts_rich_record_object(monkeypatch, tmp_path):
    resilience, FakeFingerprintRecord = _load_resilience_module(monkeypatch)
    run_dir = tmp_path / "1.2.3_abc12345_20260407_120000"

    action = resilience.reconcile_run_dir_fingerprint(
        run_dir,
        FakeFingerprintRecord(
            schema_version="local_h5_publish_v1",
            algorithm="sha256-truncated-16",
            digest="rich1234",
        ),
    )

    assert action == "initialized"
    stored = json.loads((run_dir / "fingerprint.json").read_text())
    assert stored["fingerprint"] == "rich1234"
    assert stored["digest"] == "rich1234"
    assert stored["schema_version"] == "local_h5_publish_v1"
