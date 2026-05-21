from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from policyengine_us_data.fit_weights import (
    FitScope,
    FittedWeightsOutputBundle,
)


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
