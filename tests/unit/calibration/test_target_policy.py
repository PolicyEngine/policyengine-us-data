import json

import numpy as np
import pandas as pd

from policyengine_us_data.calibration.target_policy import (
    TARGET_POLICY_ARTIFACT,
    TARGET_POLICY_SUMMARY_ARTIFACT,
    annotate_diagnostics_with_policy,
    build_target_policy,
    enforce_target_tolerances,
    load_target_policy_config,
    target_policy_arrays,
    write_target_policy_artifacts,
)


def _targets_df():
    return pd.DataFrame(
        {
            "variable": [
                "household_count",
                "person_count",
                "aca_ptc",
                "tax_unit_count",
                "snap",
            ],
            "geo_level": ["district", "state", "district", "state", "national"],
            "geographic_id": ["0101", "01", "0101", "01", "00"],
            "domain_variable": ["", "age", "", "aca_ptc", ""],
            "value": [1_000.0, 10_000.0, 50_000.0, 100_000.0, 10_000_000.0],
        }
    )


def test_build_target_policy_assigns_initial_tolerances():
    policy = build_target_policy(
        _targets_df(),
        target_names=["hh", "age", "aca_dollars", "aca_count", "snap"],
        config=load_target_policy_config(),
        row_sums=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
    )

    assert policy.loc[0, "enforcement"] == "fail"
    assert policy.loc[0, "priority"] == "P0"
    assert policy.loc[0, "tolerance_pct"] == 1.0
    assert policy.loc[0, "loss_weight"] == 40.0

    assert policy.loc[1, "enforcement"] == "warn"
    assert policy.loc[1, "priority"] == "P1"
    assert policy.loc[1, "tolerance_pct"] == 2.0

    assert policy.loc[2, "enforcement"] == "warn"
    assert policy.loc[2, "policy_rule_id"] == "aca_variable_warning_targets"
    assert policy.loc[2, "tolerance_pct"] == 7.5

    assert policy.loc[3, "enforcement"] == "warn"
    assert policy.loc[3, "policy_rule_id"] == "aca_warning_targets"
    assert policy.loc[3, "tolerance_pct"] == 7.5

    assert policy.loc[4, "enforcement"] == "diagnostic_only"
    assert policy.loc[4, "loss_weight"] == 0.0
    assert not bool(policy.loc[4, "loss_enabled"])


def test_target_policy_arrays_apply_scale_floors():
    targets = np.array([100.0, 20_000.0, 500.0, 1_000.0, 10_000_000.0])
    policy = build_target_policy(
        _targets_df(),
        config=load_target_policy_config(),
        row_sums=np.ones(5),
    )

    weights, tolerances, scales = target_policy_arrays(policy, targets)

    assert weights[0] == 40.0
    assert tolerances[0] == 0.01
    assert scales[0] == 1_000.0
    assert scales[1] == 20_000.0
    assert scales[2] == 1_000_000.0


def test_diagnostics_policy_enforces_only_hard_failures():
    policy = build_target_policy(
        _targets_df().iloc[:4],
        config=load_target_policy_config(),
        row_sums=np.ones(4),
    )
    diagnostics = pd.DataFrame(
        {
            "target": ["hh", "age", "aca_dollars", "aca_count"],
            "true_value": [100.0, 100.0, 100.0, 100.0],
            "estimate": [102.0, 103.0, 120.0, 110.0],
            "rel_error": [0.02, 0.03, 0.20, 0.10],
            "abs_rel_error": [0.02, 0.03, 0.20, 0.10],
            "achievable": [True, True, True, True],
        }
    )

    annotated = annotate_diagnostics_with_policy(diagnostics, policy)

    assert annotated.loc[0, "validation_status"] == "fail"
    assert annotated.loc[1, "validation_status"] == "warn"
    assert annotated.loc[2, "validation_status"] == "warn"
    assert annotated.loc[3, "validation_status"] == "warn"

    try:
        enforce_target_tolerances(annotated)
    except ValueError as exc:
        assert "hh" in str(exc)
    else:
        raise AssertionError("hard-fail target miss should raise")


def test_write_target_policy_artifacts(tmp_path):
    policy = build_target_policy(
        _targets_df().iloc[:2],
        config=load_target_policy_config(),
        row_sums=np.ones(2),
    )

    jsonl_path, summary_path = write_target_policy_artifacts(policy, tmp_path)

    assert jsonl_path.name == TARGET_POLICY_ARTIFACT
    assert summary_path.name == TARGET_POLICY_SUMMARY_ARTIFACT
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    assert records[0]["enforcement"] == "fail"
    assert summary["enforcement_counts"] == {"fail": 1, "warn": 1}
