from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from policyengine_us_data.fit_weights import (
    FitScope,
    FittedWeightsOutputBundle,
)
from policyengine_us_data.stage_contracts import StageContract
from policyengine_us_data.stage_contracts.calibration_package import (
    write_calibration_package_contract,
)
from tests.unit.fixtures.calibration_package_stage_contract import (
    CALIBRATION_COMPLETED_AT,
    CALIBRATION_DURATION_S,
    CALIBRATION_RUN_ID,
    CALIBRATION_STARTED_AT,
    calibration_package_parameters,
    contract_input_paths,
    write_calibration_package_payload,
)


@dataclass(frozen=True)
class Stage2ContractFixture:
    dataset_path: Path
    db_path: Path
    package_path: Path
    contract_path: Path
    contract: StageContract


class FakeBatch:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def put_file(self, file_obj, destination: str) -> None:
        self.files[destination] = file_obj.read()


@pytest.fixture
def artifacts_rel() -> str:
    return "artifacts/run-1"


@pytest.fixture
def calibration_package_path() -> Path:
    return Path("/pipeline/artifacts/run/calibration_package.pkl")


@pytest.fixture
def stage2_contract_fixture(tmp_path: Path) -> Stage2ContractFixture:
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = write_calibration_package_payload(package_path)
    contract_path = tmp_path / "calibration_package_contract.json"
    contract = write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=calibration_package_parameters(),
        run_id=CALIBRATION_RUN_ID,
        started_at=CALIBRATION_STARTED_AT,
        completed_at=CALIBRATION_COMPLETED_AT,
        duration_s=CALIBRATION_DURATION_S,
        contract_path=contract_path,
    )
    return Stage2ContractFixture(
        dataset_path=dataset_path,
        db_path=db_path,
        package_path=package_path,
        contract_path=contract_path,
        contract=contract,
    )


@pytest.fixture
def fake_batch() -> FakeBatch:
    return FakeBatch()


@pytest.fixture
def regional_result_bytes() -> dict[str, bytes]:
    return {
        "weights": b"weights",
        "geography": b"regional-geo",
        "config": b"regional-config",
        "log": b"regional-log",
        "cal_log": b"regional-epoch",
    }


@pytest.fixture
def national_result_bytes() -> dict[str, bytes]:
    return {
        "weights": b"weights",
        "geography": b"national-geo",
        "config": b"national-config",
        "log": b"national-log",
        "cal_log": b"national-epoch",
    }


@pytest.fixture
def regional_output_bundle(
    regional_result_bytes: dict[str, bytes],
) -> FittedWeightsOutputBundle:
    return FittedWeightsOutputBundle.from_result_bytes(
        scope=FitScope.REGIONAL,
        result_bytes=regional_result_bytes,
        run_id="run-1",
    )


@pytest.fixture
def national_output_bundle(
    national_result_bytes: dict[str, bytes],
) -> FittedWeightsOutputBundle:
    return FittedWeightsOutputBundle.from_result_bytes(
        scope=FitScope.NATIONAL,
        result_bytes=national_result_bytes,
        run_id="run-1",
    )


@pytest.fixture
def stage_3_substage() -> Callable[[str], dict]:
    data = yaml.safe_load(Path("docs/pipeline_map.yaml").read_text())
    substages = {substage["id"]: substage for substage in data["stages"]}
    return substages.__getitem__
