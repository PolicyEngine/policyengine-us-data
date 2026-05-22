from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from policyengine_us_data.fit_weights import (
    FitScope,
    FittedWeightsOutputBundle,
    fit_artifacts_for_scope,
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


@dataclass(frozen=True)
class ScopedFitFiles:
    scope: FitScope
    artifacts_root: Path
    diagnostics_root: Path


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
def fitted_weights_parameters() -> dict:
    return {
        "scope": "regional",
        "gpu": "T4",
        "epochs": 2,
        "target_config": "policyengine_us_data/calibration/target_config.yaml",
        "beta": 0.65,
        "lambda_l0": 1e-7,
        "lambda_l2": 1e-8,
        "log_freq": 100,
        "fit_parameter_identity": "sha256:" + "1" * 64,
        "calibration_package_sha256": "sha256:" + "2" * 64,
        "calibration_package_contract_sha256": "sha256:" + "3" * 64,
        "fitted_weights_contract_schema_version": "1",
    }


@pytest.fixture
def scoped_fit_files(tmp_path: Path) -> Callable[[FitScope | str], ScopedFitFiles]:
    def write_files(scope: FitScope | str) -> ScopedFitFiles:
        parsed_scope = FitScope.parse(scope)
        artifacts_root = tmp_path / parsed_scope.value / "artifacts"
        diagnostics_root = tmp_path / parsed_scope.value / "diagnostics"
        artifacts_root.mkdir(parents=True)
        diagnostics_root.mkdir(parents=True)
        artifacts = fit_artifacts_for_scope(parsed_scope)

        np.save(
            artifacts.weights.path_under(artifacts_root),
            np.array([1.0, 2.5, 3.5]),
        )
        np.savez(
            artifacts.geography.path_under(artifacts_root),
            block_geoid=np.array(["010010001", "010010002"]),
            cd_geoid=np.array(["0101", "0102"]),
        )
        artifacts.run_config.path_under(artifacts_root).write_text(
            json.dumps({"scope": parsed_scope.value}) + "\n"
        )
        artifacts.diagnostics.path_under(diagnostics_root).write_text(
            "target_id,error\nincome_tax,0.1\nsnap,0.2\n"
        )
        artifacts.epoch_log.path_under(diagnostics_root).write_text(
            "epoch,loss\n0,1.0\n1,0.5\n"
        )
        return ScopedFitFiles(
            scope=parsed_scope,
            artifacts_root=artifacts_root,
            diagnostics_root=diagnostics_root,
        )

    return write_files


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
