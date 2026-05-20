from policyengine_us_data.build_datasets import (
    Stage1Coordinator,
    Stage1IdentityMaterial,
    Stage1RerunPlanner,
    Stage1ReuseManifest,
    Stage1ReuseManifestRecord,
)


def test_stage_contracts_and_lazy_reuse_exports_import_together():
    import policyengine_us_data.stage_contracts as stage_contracts
    from policyengine_us_data.build_datasets import CheckpointStore

    assert stage_contracts.fingerprint_material
    assert CheckpointStore


def _material(**overrides) -> Stage1IdentityMaterial:
    values = {
        "substep_id": "1b_base_dataset_construction",
        "identity_key": (
            "1b_base_dataset_construction:"
            "script:policyengine_us_data/datasets/cps/cps.py:cps_2024.h5"
        ),
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
    manifest = Stage1ReuseManifest.empty(
        branch="stage-1",
        commit_sha="abc123",
    ).with_record(
        Stage1ReuseManifestRecord(
            substep_id=material.substep_id,
            identity_key=material.identity_key,
            identity_fingerprint=material.fingerprint(),
            identity_material=material.to_dict(),
        )
    )
    planner = Stage1RerunPlanner(previous_identities=manifest.previous_identities())

    decision = planner.decide(material, run_id="run-a", rerun_id="attempt-2")

    assert decision.action == "reuse"
    assert decision.reason == "identity_match"
    assert decision.rerun_id == "attempt-2"


def test_rerun_planner_recomputes_without_manifest_identity():
    material = _material()
    planner = Stage1RerunPlanner(
        previous_identities=Stage1ReuseManifest.empty(
            branch="stage-1",
            commit_sha="abc123",
        ).previous_identities()
    )

    decision = planner.decide(material, run_id="run-a")

    assert decision.action == "recompute"
    assert decision.reason == "no_previous_identity"


def test_rerun_planner_recomputes_mismatched_parameters():
    previous = _material()
    current = _material(parameters={"period": 2025})
    planner = Stage1RerunPlanner(
        previous_identities={previous.identity_key: previous.fingerprint()}
    )

    decision = planner.decide(current, run_id="run-a")

    assert decision.action == "recompute"
    assert decision.reason == "identity_mismatch"


def test_rerun_planner_recomputes_mismatched_schema_or_upstream_fingerprint():
    previous = _material()
    planner = Stage1RerunPlanner(
        previous_identities={previous.identity_key: previous.fingerprint()}
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


def test_reuse_manifest_keeps_same_substep_identity_keys_distinct():
    cps = _material(
        identity_key=(
            "1b_base_dataset_construction:"
            "script:policyengine_us_data/datasets/cps/cps.py:cps_2024.h5"
        ),
        inputs={"dataset": "cps"},
        artifact_specs=({"filename": "cps_2024.h5"},),
    )
    puf = _material(
        identity_key=(
            "1b_base_dataset_construction:"
            "script:policyengine_us_data/datasets/puf/puf.py:puf_2024.h5"
        ),
        inputs={"dataset": "puf"},
        artifact_specs=({"filename": "puf_2024.h5"},),
    )
    manifest = (
        Stage1ReuseManifest.empty(branch="stage-1", commit_sha="abc123")
        .with_record(
            Stage1ReuseManifestRecord(
                substep_id=cps.substep_id,
                identity_key=cps.identity_key,
                identity_fingerprint=cps.fingerprint(),
                identity_material=cps.to_dict(),
            )
        )
        .with_record(
            Stage1ReuseManifestRecord(
                substep_id=puf.substep_id,
                identity_key=puf.identity_key,
                identity_fingerprint=puf.fingerprint(),
                identity_material=puf.to_dict(),
            )
        )
    )

    assert manifest.previous_identities() == {
        cps.identity_key: cps.fingerprint(),
        puf.identity_key: puf.fingerprint(),
    }
    planner = Stage1RerunPlanner(previous_identities=manifest.previous_identities())

    assert planner.decide(cps, run_id="run-a").action == "reuse"
    assert planner.decide(puf, run_id="run-a").action == "reuse"


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


def test_reused_decision_becomes_first_class_substep_status():
    material = _material()
    decision = Stage1RerunPlanner(
        previous_identities={material.identity_key: material.fingerprint()}
    ).decide(material, run_id="run-a")
    coordinator = Stage1Coordinator()

    coordinator.run_substep(
        material.substep_id,
        "Base dataset construction",
        lambda: None,
        reuse_decision=decision.to_dict(),
        checkpoint_decisions=(
            {
                "output_file": "cps_2024.h5",
                "checkpoint_path": "/checkpoints/branch/sha/cps_2024.h5",
                "action": "reuse",
                "reason": "valid",
                "size_bytes": 3,
            },
        ),
    )

    [result] = coordinator.results
    assert result.status == "reused"
    assert coordinator.status_events[-1].status == "reused"
