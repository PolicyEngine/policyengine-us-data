import importlib.util
import pickle
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np


def _module_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[3].joinpath(*parts)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_package_hierarchy(monkeypatch):
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

    clone_module = types.ModuleType("policyengine_us_data.calibration.clone_and_assign")

    @dataclass(frozen=True)
    class FakeGeographyAssignment:
        block_geoid: np.ndarray
        cd_geoid: np.ndarray
        county_fips: np.ndarray
        state_fips: np.ndarray
        n_records: int
        n_clones: int

    def fake_assign_random_geography(*, n_records, n_clones, seed):
        total = n_records * n_clones
        return FakeGeographyAssignment(
            block_geoid=np.asarray(["990010000000001"] * total, dtype=str),
            cd_geoid=np.asarray(["9901"] * total, dtype=str),
            county_fips=np.asarray(["99001"] * total, dtype=str),
            state_fips=np.asarray([99] * total, dtype=np.int64),
            n_records=n_records,
            n_clones=n_clones,
        )

    clone_module.GeographyAssignment = FakeGeographyAssignment
    clone_module.assign_random_geography = fake_assign_random_geography
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.clone_and_assign",
        clone_module,
    )

    package_geography = _load_module(
        "policyengine_us_data.calibration.local_h5.package_geography",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "package_geography.py",
        ),
    )
    fingerprinting = _load_module(
        "policyengine_us_data.calibration.local_h5.fingerprinting",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "fingerprinting.py",
        ),
    )
    return package_geography, fingerprinting


def _write_bytes(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _write_weights(path: Path, values: np.ndarray) -> Path:
    np.save(path, values)
    return path


def _write_package(path: Path, *, geography: dict, metadata: dict | None = None) -> Path:
    with open(path, "wb") as f:
        pickle.dump(
            {
                "geography": geography,
                "metadata": metadata or {},
                "X_sparse": types.SimpleNamespace(shape=(1, 1)),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return path


def _sample_geography(*, suffix: str) -> dict:
    return {
        "block_geoid": np.asarray(
            [f"06001000100100{suffix}", f"36061000100100{suffix}"],
            dtype=str,
        ),
        "cd_geoid": np.asarray(["601", "1208"], dtype=str),
        "county_fips": np.asarray(["06001", "36061"], dtype=str),
        "state_fips": np.asarray([6, 36], dtype=np.int64),
        "n_records": 2,
        "n_clones": 1,
    }


def test_create_publish_fingerprint_round_trips_record(monkeypatch, tmp_path):
    _, fingerprinting = _install_fake_package_hierarchy(monkeypatch)
    FingerprintService = fingerprinting.FingerprintService

    weights_path = _write_weights(
        tmp_path / "weights.npy",
        np.asarray([1.0, 2.0], dtype=np.float64),
    )
    dataset_path = _write_bytes(tmp_path / "dataset.h5", b"dataset-one")
    package_path = _write_package(
        tmp_path / "package.pkl",
        geography=_sample_geography(suffix="1"),
        metadata={"git_commit": "abc"},
    )

    service = FingerprintService()
    record = service.create_publish_fingerprint(
        weights_path=weights_path,
        dataset_path=dataset_path,
        calibration_package_path=package_path,
        n_clones=1,
        seed=42,
    )

    assert len(record.digest) == 16
    assert record.components is not None
    assert record.inputs["weights_path"] == str(weights_path)

    payload = service.serialize(record)
    restored = service.deserialize(payload)

    assert restored.digest == record.digest
    assert restored.components == record.components
    assert service.matches(record, restored)


def test_write_and_read_record_round_trip(monkeypatch, tmp_path):
    _, fingerprinting = _install_fake_package_hierarchy(monkeypatch)
    FingerprintService = fingerprinting.FingerprintService

    weights_path = _write_weights(
        tmp_path / "weights.npy",
        np.asarray([1.0, 2.0], dtype=np.float64),
    )
    dataset_path = _write_bytes(tmp_path / "dataset.h5", b"dataset-one")
    package_path = _write_package(
        tmp_path / "package.pkl",
        geography=_sample_geography(suffix="1"),
    )

    service = FingerprintService()
    record = service.create_publish_fingerprint(
        weights_path=weights_path,
        dataset_path=dataset_path,
        calibration_package_path=package_path,
        n_clones=1,
        seed=42,
    )

    record_path = tmp_path / "fingerprint.json"
    service.write_record(record_path, record)
    restored = service.read_record(record_path)

    assert restored == record


def test_publish_fingerprint_changes_when_geography_changes(monkeypatch, tmp_path):
    _, fingerprinting = _install_fake_package_hierarchy(monkeypatch)
    FingerprintService = fingerprinting.FingerprintService

    weights_path = _write_weights(
        tmp_path / "weights.npy",
        np.asarray([1.0, 2.0], dtype=np.float64),
    )
    dataset_path = _write_bytes(tmp_path / "dataset.h5", b"dataset-one")
    package_a = _write_package(
        tmp_path / "package-a.pkl",
        geography=_sample_geography(suffix="1"),
    )
    package_b = _write_package(
        tmp_path / "package-b.pkl",
        geography=_sample_geography(suffix="2"),
    )

    service = FingerprintService()
    record_a = service.create_publish_fingerprint(
        weights_path=weights_path,
        dataset_path=dataset_path,
        calibration_package_path=package_a,
        n_clones=1,
        seed=42,
    )
    record_b = service.create_publish_fingerprint(
        weights_path=weights_path,
        dataset_path=dataset_path,
        calibration_package_path=package_b,
        n_clones=1,
        seed=42,
    )

    assert record_a.components is not None
    assert record_b.components is not None
    assert record_a.components.geography_sha256 != record_b.components.geography_sha256
    assert record_a.digest != record_b.digest


def test_publish_fingerprint_ignores_non_geography_package_metadata(
    monkeypatch, tmp_path
):
    _, fingerprinting = _install_fake_package_hierarchy(monkeypatch)
    FingerprintService = fingerprinting.FingerprintService

    weights_path = _write_weights(
        tmp_path / "weights.npy",
        np.asarray([1.0, 2.0], dtype=np.float64),
    )
    dataset_path = _write_bytes(tmp_path / "dataset.h5", b"dataset-one")
    geography = _sample_geography(suffix="1")
    package_a = _write_package(
        tmp_path / "package-a.pkl",
        geography=geography,
        metadata={"git_commit": "abc", "created_at": "one"},
    )
    package_b = _write_package(
        tmp_path / "package-b.pkl",
        geography=geography,
        metadata={"git_commit": "def", "created_at": "two"},
    )

    service = FingerprintService()
    record_a = service.create_publish_fingerprint(
        weights_path=weights_path,
        dataset_path=dataset_path,
        calibration_package_path=package_a,
        n_clones=1,
        seed=42,
    )
    record_b = service.create_publish_fingerprint(
        weights_path=weights_path,
        dataset_path=dataset_path,
        calibration_package_path=package_b,
        n_clones=1,
        seed=42,
    )

    assert record_a.components == record_b.components
    assert record_a.digest == record_b.digest


def test_publish_fingerprint_rejects_incompatible_package_shape(monkeypatch, tmp_path):
    _, fingerprinting = _install_fake_package_hierarchy(monkeypatch)
    FingerprintService = fingerprinting.FingerprintService

    weights_path = _write_weights(
        tmp_path / "weights.npy",
        np.asarray([1.0, 2.0], dtype=np.float64),
    )
    dataset_path = _write_bytes(tmp_path / "dataset.h5", b"dataset-one")
    package_path = _write_package(
        tmp_path / "package.pkl",
        geography={
            "block_geoid": np.asarray(["060010001001001"] * 4, dtype=str),
            "cd_geoid": np.asarray(["601"] * 4, dtype=str),
            "county_fips": np.asarray(["06001"] * 4, dtype=str),
            "state_fips": np.asarray([6] * 4, dtype=np.int64),
            "n_records": 2,
            "n_clones": 2,
        },
    )

    service = FingerprintService()
    try:
        service.create_publish_fingerprint(
            weights_path=weights_path,
            dataset_path=dataset_path,
            calibration_package_path=package_path,
            n_clones=1,
            seed=42,
        )
    except ValueError as error:
        assert "incompatible with the requested publish shape" in str(error)
    else:
        raise AssertionError("Expected incompatible package shape to fail")


def test_deserialize_legacy_fingerprint_payload(monkeypatch):
    _, fingerprinting = _install_fake_package_hierarchy(monkeypatch)
    FingerprintService = fingerprinting.FingerprintService

    service = FingerprintService()
    record = service.deserialize({"fingerprint": "deadbeefdeadbeef"})

    assert record.schema_version == "legacy"
    assert record.digest == "deadbeefdeadbeef"
    assert record.components is None
