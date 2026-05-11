from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from policyengine_us_data.build_outputs.fingerprinting import PublishingInputBundle
from policyengine_us_data.build_outputs.validation import (
    AreaValidationService,
    ValidationContext,
    ValidationPolicy,
)


def _inputs(tmp_path: Path, *, with_db: bool = True) -> PublishingInputBundle:
    return PublishingInputBundle(
        weights_path=tmp_path / "weights.npy",
        source_dataset_path=tmp_path / "source.h5",
        target_db_path=tmp_path / "policy_data.db" if with_db else None,
        exact_geography_path=tmp_path / "geography.npz",
        calibration_package_path=None,
        run_config_path=None,
        run_id="run-123",
        version="0.0.0",
        n_clones=2,
        seed=42,
    )


def test_validation_service_returns_none_when_disabled(tmp_path):
    service = AreaValidationService()

    context = service.prepare_context(
        inputs=_inputs(tmp_path),
        policy=ValidationPolicy(enabled=False),
        period=2024,
    )

    assert context is None


def test_validation_service_returns_empty_context_without_db_path(tmp_path):
    service = AreaValidationService()

    context = service.prepare_context(
        inputs=_inputs(tmp_path, with_db=False),
        policy=ValidationPolicy(),
        period=2024,
        target_config_path=tmp_path / "target_config.yaml",
    )

    assert isinstance(context, ValidationContext)
    assert context.target_db_path is None
    assert context.validation_targets is None
    assert context.training_mask is None
    assert context.target_config_path == tmp_path / "target_config.yaml"


def test_validation_service_prepares_targets_training_mask_and_constraints(tmp_path):
    engine_urls = []
    constraint_calls = []
    disposed = []
    target_config = tmp_path / "target_config.yaml"
    validation_config = tmp_path / "validation_config.yaml"

    class FakeEngine:
        def dispose(self):
            disposed.append(True)

    def engine_factory(url: str):
        engine_urls.append(url)
        return FakeEngine()

    def query_targets(engine, period: int):
        assert period == 2024
        return pd.DataFrame(
            {
                "variable": ["household_count", "income", "rent"],
                "stratum_id": [1, 2, 3],
                "geo_level": ["state", "state", "state"],
                "geographic_id": ["37", "37", "37"],
            }
        )

    def batch_constraints(engine, stratum_ids: list[int]):
        constraint_calls.append(tuple(stratum_ids))
        return {stratum_id: [f"constraint-{stratum_id}"] for stratum_id in stratum_ids}

    def load_config(path: Path | str):
        if Path(path) == validation_config:
            return {"exclude": [{"variable": "rent"}]}
        if Path(path) == target_config:
            return {"include": [{"variable": "income"}]}
        return {}

    def match_rules(targets, rules):
        variables = {rule["variable"] for rule in rules}
        return targets["variable"].isin(variables).to_numpy()

    service = AreaValidationService(
        engine_factory=engine_factory,
        query_targets=query_targets,
        batch_constraints=batch_constraints,
        load_target_config=load_config,
        match_rules=match_rules,
    )

    context = service.prepare_context(
        inputs=_inputs(tmp_path),
        policy=ValidationPolicy(),
        period=2024,
        target_config_path=target_config,
        validation_config_path=validation_config,
    )

    assert engine_urls == [f"sqlite:///{tmp_path / 'policy_data.db'}"]
    assert disposed == [True]
    assert context.validation_targets["variable"].tolist() == [
        "household_count",
        "income",
    ]
    assert np.array_equal(context.training_mask, np.array([False, True]))
    assert context.constraints_map == {
        1: ["constraint-1"],
        2: ["constraint-2"],
    }
    assert constraint_calls == [(1, 2)]
