from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from policyengine_us_data.build_outputs.requests import AreaBuildRequest, AreaFilter
from policyengine_us_data.build_outputs.validation import AreaValidationResult
from policyengine_us_data.build_outputs.worker_service import (
    LocalH5WorkerService,
    WorkerAreaResult,
    WorkerExecutionConfig,
    WorkerIssue,
    WorkerResult,
)
from policyengine_us_data.build_outputs.worker_session import WorkerSession


def _request(
    area_type: str = "district",
    area_id: str = "NC-01",
    output_relative_path: str = "districts/NC-01.h5",
) -> AreaBuildRequest:
    return AreaBuildRequest(
        area_type=area_type,
        area_id=area_id,
        display_name=area_id,
        output_relative_path=output_relative_path,
        filters=(
            AreaFilter(
                geography_field="cd_geoid",
                op="in",
                value=("3701",),
            ),
        )
        if area_type != "national"
        else (),
        validation_geo_level="district" if area_type != "national" else "national",
        validation_geographic_ids=(area_id,),
    )


def _session(
    *,
    validation_context=None,
    weight_clones: int = 2,
    geography_clones: int = 2,
    source_loader=None,
) -> WorkerSession:
    source = SimpleNamespace(
        variable_provider=SimpleNamespace(simulation=SimpleNamespace()),
    )
    return WorkerSession(
        inputs=SimpleNamespace(),
        scope="regional",
        source=source,
        weights=SimpleNamespace(
            values=np.ones(weight_clones),
            n_records=1,
            n_clones=weight_clones,
        ),
        geography=SimpleNamespace(n_records=1, n_clones=geography_clones),
        validation_context=validation_context,
        source_loader=source_loader,
    )


class FakeBuilder:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def build(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("build failed")
        return SimpleNamespace(payload=SimpleNamespace(name="payload"))


class FakeWriter:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def write(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("write failed")
        return SimpleNamespace(path=Path(kwargs["output_path"]))


class FakeValidationService:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def validate_request(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("validation failed")
        return AreaValidationResult(
            rows=(
                {
                    "variable": "household_count",
                    "sanity_check": "PASS",
                    "rel_abs_error": 0.0,
                },
            ),
            summary={
                "n_targets": 1,
                "n_sanity_fail": 0,
                "mean_rel_abs_error": 0.0,
            },
        )


def test_worker_result_preserves_legacy_and_structured_shapes(tmp_path):
    request = _request()
    result = WorkerResult(
        area_results=(
            WorkerAreaResult(
                key="district:NC-01",
                request=request,
                status="completed",
                output_relative_path=request.output_relative_path,
                output_path=tmp_path / "districts" / "NC-01.h5",
                validation_status="passed",
                validation_rows=({"variable": "household_count"},),
                validation_summary={"n_targets": 1},
            ),
        ),
        issues=(
            WorkerIssue(
                item="district:bad",
                phase="request",
                message="bad request",
            ),
        ),
    )

    payload = result.to_legacy_dict()

    assert payload["completed"] == ["district:NC-01"]
    assert payload["failed"] == ["district:bad"]
    assert payload["errors"] == [
        {"item": "district:bad", "phase": "request", "error": "bad request"}
    ]
    assert payload["validation_rows"] == [{"variable": "household_count"}]
    assert payload["validation_summary"] == {"district:NC-01": {"n_targets": 1}}
    assert payload["results"][0]["key"] == "district:NC-01"
    assert payload["issues"][0]["item"] == "district:bad"


def test_worker_service_builds_writes_and_validates_request(tmp_path):
    builder = FakeBuilder()
    writer = FakeWriter()
    validation_service = FakeValidationService()
    request = _request()

    result = LocalH5WorkerService(
        builder=builder,
        writer=writer,
        validation_service=validation_service,
    ).execute(
        session=_session(validation_context=SimpleNamespace()),
        requests=(request,),
        config=WorkerExecutionConfig(
            output_dir=tmp_path,
            takeup_filter=("takes_up_snap",),
            validate=True,
        ),
    )

    payload = result.to_legacy_dict()

    assert payload["completed"] == ["district:NC-01"]
    assert payload["failed"] == []
    assert payload["errors"] == []
    assert payload["validation_summary"]["district:NC-01"]["n_targets"] == 1
    assert builder.calls[0]["request"] == request
    assert builder.calls[0]["takeup_filter"] == ("takes_up_snap",)
    assert writer.calls[0]["output_path"] == tmp_path / "districts" / "NC-01.h5"
    assert validation_service.calls[0]["request"] == request


def test_worker_service_does_not_apply_takeup_filter_to_national_request(tmp_path):
    builder = FakeBuilder()
    request = _request("national", "US", "national/US.h5")

    result = LocalH5WorkerService(
        builder=builder,
        writer=FakeWriter(),
        validation_service=FakeValidationService(),
    ).execute(
        session=_session(),
        requests=(request,),
        config=WorkerExecutionConfig(
            output_dir=tmp_path,
            takeup_filter=("takes_up_snap",),
            validate=False,
        ),
    )

    assert result.to_legacy_dict()["completed"] == ["national:US"]
    assert builder.calls[0]["takeup_filter"] is None


def test_worker_service_loads_fresh_source_for_each_request(tmp_path):
    builder = FakeBuilder()
    requests = (
        _request("district", "NC-01", "districts/NC-01.h5"),
        _request("state", "NC", "states/NC.h5"),
    )
    simulations = (
        SimpleNamespace(name="first-simulation"),
        SimpleNamespace(name="second-simulation"),
    )
    sources = [
        SimpleNamespace(variable_provider=SimpleNamespace(simulation=simulation))
        for simulation in simulations
    ]
    source_loads = []

    def load_source():
        source = sources[len(source_loads)]
        source_loads.append(source)
        return source

    result = LocalH5WorkerService(
        builder=builder,
        writer=FakeWriter(),
        validation_service=FakeValidationService(),
    ).execute(
        session=_session(source_loader=load_source),
        requests=requests,
        config=WorkerExecutionConfig(output_dir=tmp_path, validate=False),
    )

    assert result.to_legacy_dict()["completed"] == ["district:NC-01", "state:NC"]
    assert source_loads == sources
    assert [call["source"] for call in builder.calls] == sources
    assert [call["simulation"] for call in builder.calls] == list(simulations)


def test_worker_service_reports_build_failures(tmp_path):
    request = _request()

    result = LocalH5WorkerService(
        builder=FakeBuilder(fail=True),
        writer=FakeWriter(),
        validation_service=FakeValidationService(),
    ).execute(
        session=_session(),
        requests=(request,),
        config=WorkerExecutionConfig(output_dir=tmp_path, validate=False),
    )

    payload = result.to_legacy_dict()

    assert payload["completed"] == []
    assert payload["failed"] == ["district:NC-01"]
    assert payload["errors"][0]["phase"] == "build"
    assert payload["errors"][0]["error"] == "build failed"
    assert payload["results"][0]["status"] == "failed"


def test_worker_service_records_validation_errors_without_failing_by_default(
    tmp_path,
):
    request = _request()

    result = LocalH5WorkerService(
        builder=FakeBuilder(),
        writer=FakeWriter(),
        validation_service=FakeValidationService(fail=True),
    ).execute(
        session=_session(validation_context=SimpleNamespace()),
        requests=(request,),
        config=WorkerExecutionConfig(output_dir=tmp_path, validate=True),
    )

    payload = result.to_legacy_dict()

    assert payload["completed"] == ["district:NC-01"]
    assert payload["failed"] == []
    assert payload["errors"][0]["phase"] == "validation"
    assert payload["errors"][0]["error"] == "validation failed"
    assert payload["issues"][0]["phase"] == "validation"
    assert payload["results"][0]["validation_status"] == "error"


def test_worker_service_can_fail_on_validation_error(tmp_path):
    request = _request()

    result = LocalH5WorkerService(
        builder=FakeBuilder(),
        writer=FakeWriter(),
        validation_service=FakeValidationService(fail=True),
    ).execute(
        session=_session(validation_context=SimpleNamespace()),
        requests=(request,),
        config=WorkerExecutionConfig(
            output_dir=tmp_path,
            validate=True,
            fail_on_validation_error=True,
        ),
    )

    payload = result.to_legacy_dict()

    assert payload["completed"] == []
    assert payload["failed"] == ["district:NC-01"]
    assert payload["errors"][0]["phase"] == "validation"


def test_worker_service_rejects_output_path_escape(tmp_path):
    request = _request()
    object.__setattr__(request, "output_relative_path", "../escape.h5")

    result = LocalH5WorkerService(
        builder=FakeBuilder(),
        writer=FakeWriter(),
        validation_service=FakeValidationService(),
    ).execute(
        session=_session(),
        requests=(request,),
        config=WorkerExecutionConfig(output_dir=tmp_path, validate=False),
    )

    payload = result.to_legacy_dict()

    assert payload["failed"] == ["district:NC-01"]
    assert payload["errors"][0]["phase"] == "request"
    assert "worker output_dir" in payload["errors"][0]["error"]


def test_worker_service_rejects_national_weight_geography_mismatch(tmp_path):
    request = _request("national", "US", "national/US.h5")

    result = LocalH5WorkerService(
        builder=FakeBuilder(),
        writer=FakeWriter(),
        validation_service=FakeValidationService(),
    ).execute(
        session=_session(weight_clones=2, geography_clones=3),
        requests=(request,),
        config=WorkerExecutionConfig(output_dir=tmp_path, validate=False),
    )

    payload = result.to_legacy_dict()

    assert payload["failed"] == ["national:US"]
    assert payload["errors"][0]["phase"] == "build"
    assert (
        "National weights have 2 clones but geography has 3"
        in payload["errors"][0]["error"]
    )
