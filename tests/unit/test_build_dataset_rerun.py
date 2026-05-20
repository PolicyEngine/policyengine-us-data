from policyengine_us_data.build_datasets import (
    Stage1Coordinator,
    Stage1IdentityMaterial,
    Stage1RerunPlanner,
)


def _material(**overrides) -> Stage1IdentityMaterial:
    values = {
        "substep_id": "1b_base_dataset_construction",
        "code_sha": "abc123",
        "schema_version": "stage-1-rerun-v1",
        "inputs": {"dataset": "cps"},
        "parameters": {"period": 2024},
        "artifact_specs": ({"filename": "cps_2024.h5"},),
        "upstream_contract_fingerprints": ("sha256:upstream",),
        "randomness": {"seed": 1},
    }
    values.update(overrides)
    return Stage1IdentityMaterial(**values)


def test_rerun_planner_reuses_matching_identity():
    material = _material()
    planner = Stage1RerunPlanner(
        previous_identities={material.substep_id: material.fingerprint()}
    )

    decision = planner.decide(material, run_id="run-a", rerun_id="attempt-2")

    assert decision.action == "reuse"
    assert decision.reason == "identity_match"
    assert decision.rerun_id == "attempt-2"


def test_rerun_planner_recomputes_mismatched_parameters():
    previous = _material()
    current = _material(parameters={"period": 2025})
    planner = Stage1RerunPlanner(
        previous_identities={previous.substep_id: previous.fingerprint()}
    )

    decision = planner.decide(current, run_id="run-a")

    assert decision.action == "recompute"
    assert decision.reason == "identity_mismatch"


def test_rerun_planner_recomputes_mismatched_schema_or_upstream_fingerprint():
    previous = _material()
    planner = Stage1RerunPlanner(
        previous_identities={previous.substep_id: previous.fingerprint()}
    )

    schema_decision = planner.decide(
        _material(schema_version="stage-1-rerun-v2"),
        run_id="run-a",
    )
    upstream_decision = planner.decide(
        _material(upstream_contract_fingerprints=("sha256:changed",)),
        run_id="run-a",
    )

    assert schema_decision.action == "recompute"
    assert upstream_decision.action == "recompute"


def test_rerun_id_does_not_change_artifact_namespace():
    material = _material()

    decision = Stage1RerunPlanner().decide(
        material,
        run_id="canonical-run",
        rerun_id="attempt-2",
    )

    assert decision.artifact_namespace == "canonical-run"
    assert decision.rerun_id == "attempt-2"


def test_blocked_decision_serializes_into_substep_status():
    material = _material(blocked_reason="missing upstream contract")
    decision = Stage1RerunPlanner().decide(material, run_id="run-a")
    coordinator = Stage1Coordinator()

    coordinator.run_substep(
        material.substep_id,
        "Base dataset construction",
        lambda: None,
        reuse_decision=decision.to_dict(),
        checkpoint_decisions=(),
        skip=True,
        skip_reason="blocked",
    )

    [result] = coordinator.results
    assert decision.action == "blocked"
    assert result.reuse_decision == decision.to_dict()
    assert coordinator.status_events[-1].metadata["reuse_decision"] == (
        decision.to_dict()
    )
