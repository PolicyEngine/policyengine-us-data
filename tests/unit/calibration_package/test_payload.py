import json

import pytest

from tests.unit.fixtures.calibration_package_stage_contract import (
    calibration_package_contract,
    calibration_package_payload,
    calibration_package_payload_without_geography,
)

from policyengine_us_data.calibration_package.payload import (
    LEGACY_MISSING_GEOGRAPHY_WARNING,
    CalibrationPackagePayload,
    CalibrationPackageReader,
    CalibrationPackageWriter,
)
from policyengine_us_data.stage_contracts.calibration_package import (
    summarize_calibration_package,
)
from policyengine_us_data.stage_contracts.calibration_package_schema import (
    CalibrationPackageSummary,
    GeographyAssignmentSummary,
)


def test_calibration_package_payload_read_write_round_trip(tmp_path):
    package_path = tmp_path / "calibration_package.pkl"
    payload = CalibrationPackagePayload.from_mapping(calibration_package_payload())

    written = CalibrationPackageWriter(package_path=package_path).write(payload)
    loaded = CalibrationPackageReader(package_path=package_path).read()

    assert written == package_path
    assert loaded.summary() == payload.summary()
    assert loaded.geography_summary() == payload.geography_summary()
    assert (
        CalibrationPackageReader(package_path=package_path)
        .checksum()
        .startswith("sha256:")
    )


@pytest.mark.parametrize(
    "missing_key",
    ["X_sparse", "targets_df", "target_names", "metadata"],
)
def test_calibration_package_payload_rejects_missing_required_keys(missing_key):
    package = calibration_package_payload()
    package.pop(missing_key)

    with pytest.raises(ValueError, match=missing_key):
        CalibrationPackagePayload.from_mapping(package)


def test_legacy_package_without_geography_records_compatibility_warning():
    payload = CalibrationPackagePayload.from_mapping(
        calibration_package_payload_without_geography()
    )

    assert payload.compatibility_warnings == (LEGACY_MISSING_GEOGRAPHY_WARNING,)
    geography = payload.geography_summary()
    assert geography.source_kind == "unavailable"


def test_payload_summary_matches_existing_contract_summary():
    payload = CalibrationPackagePayload.from_mapping(calibration_package_payload())

    assert payload.summary() == summarize_calibration_package(payload.to_mapping())


def test_metadata_sidecar_uses_payload_and_contract(tmp_path):
    package_path = tmp_path / "calibration_package.pkl"
    payload = CalibrationPackagePayload.from_mapping(calibration_package_payload())
    contract = calibration_package_contract(tmp_path)
    writer = CalibrationPackageWriter(package_path=package_path)
    writer.write(payload)

    sidecar_path = writer.write_metadata_sidecar(payload, contract=contract)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert sidecar_path == tmp_path / "calibration_package_meta.json"
    assert CalibrationPackageSummary.from_dict(sidecar["package_summary"]) == (
        payload.summary()
    )
    assert GeographyAssignmentSummary.from_dict(sidecar["geography_assignment"]) == (
        payload.geography_summary()
    )
    assert sidecar["contract"]["stage_id"] == "2_build_calibration_package"
    assert sidecar["compatibility_warnings"] == []
