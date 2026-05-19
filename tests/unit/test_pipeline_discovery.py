import pytest

from modal_app.pipeline_discovery_core import (
    build_deployed_pipeline_runs_payload,
    derive_run_id_from_app_name,
    is_publication_pipeline_app_name,
    pipeline_app_candidates,
)
from modal_app.pipeline_discovery_schema import (
    DeployedPipelineRunsPayload,
)


def _app_record(
    name: str,
    *,
    app_id: str = "ap-1",
    state: str = "deployed",
    tasks: int = 0,
    created_at: str = "2026-05-19T12:00:00+00:00",
) -> dict:
    return {
        "app_id": app_id,
        "name": name,
        "state": state,
        "tasks": tasks,
        "created_at": created_at,
    }


def _status_payload(
    run_id: str,
    *,
    status: str = "running",
    branch: str = "main",
    updated_at: str = "2026-05-19T12:30:00+00:00",
) -> dict:
    return {
        "schema_version": "1",
        "run_id": run_id,
        "status": status,
        "message": f"Pipeline {status}.",
        "updated_at": updated_at,
        "run_manifest": {
            "run_id": run_id,
            "branch": branch,
            "sha": "abc123",
            "candidate_version": "1.115.4-minor",
            "release_version": "",
            "started_at": "2026-05-19T12:00:00+00:00",
            "completed_at": None,
            "known_step_ids": ["1_build_datasets", "2_build_calibration_package"],
            "hf_staging_prefix": f"staging/1.115.4-minor-{run_id}",
            "run_context": {
                "github_run_url": (
                    "https://github.com/PolicyEngine/policyengine-us-data/"
                    "actions/runs/123"
                )
            },
        },
        "stage_manifests": [
            {
                "step_id": "1_build_datasets",
                "stage_id": "1_build_datasets",
                "substage_id": None,
                "title": "Build datasets",
                "status": "completed",
                "manifest": {
                    "started_at": "2026-05-19T12:00:00+00:00",
                    "completed_at": "2026-05-19T12:15:00+00:00",
                    "duration_s": 900,
                    "reuse_decision": "not_applicable",
                },
            }
        ],
        "missing_expected_manifest_ids": ["2_build_calibration_package"],
        "error": None,
        "modal_app_name": f"us-data-1-115-4-minor-{run_id}",
        "modal_environment": "main",
    }


def test_derives_run_id_from_current_and_legacy_publication_app_names():
    assert (
        derive_run_id_from_app_name("us-data-1-115-4-minor-usdata-gha26114604836-a1")
        == "usdata-gha26114604836-a1"
    )
    assert (
        derive_run_id_from_app_name("policyengine-us-data-pub-usdata-gha123-a2")
        == "usdata-gha123-a2"
    )


@pytest.mark.parametrize(
    ("app_name", "expected"),
    [
        ("us-data-1-115-4-minor-usdata-gha26114604836-a1", True),
        ("policyengine-us-data-pub-usdata-gha123-a1", True),
        ("policyengine-us-data-1-115-4-minor-usdata-gha123-a1", True),
        ("us-data-pipeline-pr-1035-26117326123-1", False),
        ("us-data-local-area-pr-1035-26117326123-1", False),
        ("us-data-h5-pr-1035-26117326123-1", False),
        ("policyengine-us-data-pipeline", False),
        ("state-research-tracker", False),
    ],
)
def test_identifies_publication_pipeline_app_names(app_name, expected):
    assert is_publication_pipeline_app_name(app_name) is expected


def test_pipeline_app_candidates_filters_to_deployed_publication_apps():
    records = [
        _app_record(
            "us-data-1-115-4-minor-usdata-gha26114604836-a1",
            app_id="ap-new",
            tasks=2,
            created_at="2026-05-19T13:00:00+00:00",
        ),
        _app_record("us-data-pipeline-pr-1035-26117326123-1"),
        _app_record(
            "us-data-1-115-4-patch-usdata-gha26114905403-a1",
            state="stopped",
        ),
    ]

    candidates = pipeline_app_candidates(records)

    assert len(candidates) == 1
    assert candidates[0].app_id == "ap-new"
    assert candidates[0].app_name == ("us-data-1-115-4-minor-usdata-gha26114604836-a1")
    assert candidates[0].run_id == "usdata-gha26114604836-a1"
    assert candidates[0].task_count == 2


def test_build_deployed_pipeline_runs_payload_queries_status_by_derived_run_id():
    app_name = "us-data-1-115-4-minor-usdata-gha26114604836-a1"
    seen = []

    def lookup(candidate):
        seen.append((candidate.app_name, candidate.run_id))
        return _status_payload(candidate.run_id)

    payload = build_deployed_pipeline_runs_payload(
        [_app_record(app_name, app_id="ap-run", tasks=4)],
        lookup,
        limit=10,
        max_workers=1,
    )

    assert seen == [(app_name, "usdata-gha26114604836-a1")]
    assert isinstance(payload, DeployedPipelineRunsPayload)
    assert payload.schema_version == "1"
    assert payload.source == "modal_app_names"
    assert payload.discovered_count == 1
    assert payload.queried_count == 1
    assert payload.count == 1
    run = payload.runs[0]
    assert run.run_id == "usdata-gha26114604836-a1"
    assert run.status_lookup == "ok"
    assert run.status == "running"
    assert run.branch == "main"
    assert run.modal_app_id == "ap-run"
    assert run.modal_task_count == 4
    assert run.latest_manifest is not None
    assert run.latest_manifest.step_id == "1_build_datasets"
    assert run.progress is not None
    assert run.progress.to_dict() == {
        "expected_manifests": 2,
        "present_manifests": 1,
        "missing_manifests": 1,
    }
    assert payload.to_dict()["runs"][0]["run_id"] == "usdata-gha26114604836-a1"


def test_deployed_pipeline_runs_payload_keeps_unreachable_apps_structured():
    def lookup(_candidate):
        raise RuntimeError("lookup failed with TOKEN=secret")

    payload = build_deployed_pipeline_runs_payload(
        [_app_record("us-data-1-115-4-minor-usdata-gha26114604836-a1")],
        lookup,
        max_workers=1,
    )

    assert payload.count == 1
    run = payload.runs[0]
    assert run.run_id == "usdata-gha26114604836-a1"
    assert run.status_lookup == "unreachable"
    assert run.status == "unreachable"
    assert run.error is not None
    assert run.error.error_type == "RuntimeError"


def test_deployed_pipeline_runs_payload_applies_limit_after_filters():
    records = [
        _app_record(
            "us-data-1-115-4-minor-usdata-gha1-a1",
            created_at="2026-05-19T13:00:00+00:00",
        ),
        _app_record(
            "us-data-1-115-4-minor-usdata-gha2-a1",
            created_at="2026-05-19T12:00:00+00:00",
        ),
    ]

    def lookup(candidate):
        branch = "feature" if candidate.run_id.endswith("2-a1") else "main"
        return _status_payload(candidate.run_id, branch=branch)

    payload = build_deployed_pipeline_runs_payload(
        records,
        lookup,
        limit=1,
        branch="main",
        max_workers=1,
    )

    assert payload.limit == 1
    assert payload.filters.to_dict() == {
        "status": "",
        "branch": "main",
        "include_unreachable": True,
    }
    assert payload.queried_count == 2
    assert [run.run_id for run in payload.runs] == ["usdata-gha1-a1"]


def test_deployed_pipeline_runs_payload_can_exclude_unreachable_apps():
    def lookup(_candidate):
        raise RuntimeError("unavailable")

    payload = build_deployed_pipeline_runs_payload(
        [_app_record("us-data-1-115-4-minor-usdata-gha26114604836-a1")],
        lookup,
        include_unreachable=False,
        max_workers=1,
    )

    assert payload.discovered_count == 1
    assert payload.queried_count == 1
    assert payload.count == 0
    assert payload.runs == ()
