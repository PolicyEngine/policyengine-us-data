"""Worker-scoped validation context for local H5 publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from policyengine_us_data.pipeline_metadata import pipeline_node

from .fingerprinting import PublishingInputBundle

__all__ = [
    "AreaValidationService",
    "ValidationContext",
    "ValidationPolicy",
]


@pipeline_node(
    id="local_h5_validation_policy",
    label="ValidationPolicy",
    node_type="library",
    description="Worker-scoped local H5 validation policy contract.",
    source_file="policyengine_us_data/build_outputs/validation.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_validation.py"],
)
@dataclass(frozen=True)
class ValidationPolicy:
    """Validation switch for a local H5 worker session."""

    enabled: bool = True


@pipeline_node(
    id="local_h5_validation_context",
    label="ValidationContext",
    node_type="library",
    description="Prepared per-worker local H5 validation target context.",
    source_file="policyengine_us_data/build_outputs/validation.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_validation.py"],
)
@dataclass(frozen=True)
class ValidationContext:
    """Prepared validation data reused across all requests in one worker."""

    policy: ValidationPolicy
    target_db_path: Path | None
    period: int
    validation_targets: Any = None
    training_mask: np.ndarray | None = None
    constraints_map: Mapping[int, Any] | None = None
    target_config_path: Path | None = None
    validation_config_path: Path | None = None

    def __post_init__(self) -> None:
        target_db_path = (
            Path(self.target_db_path) if self.target_db_path is not None else None
        )
        target_config_path = (
            Path(self.target_config_path)
            if self.target_config_path is not None
            else None
        )
        validation_config_path = (
            Path(self.validation_config_path)
            if self.validation_config_path is not None
            else None
        )
        object.__setattr__(self, "target_db_path", target_db_path)
        object.__setattr__(self, "period", int(self.period))
        object.__setattr__(self, "target_config_path", target_config_path)
        object.__setattr__(self, "validation_config_path", validation_config_path)
        if self.training_mask is not None:
            object.__setattr__(
                self,
                "training_mask",
                np.asarray(self.training_mask, dtype=bool),
            )
        if self.constraints_map is not None:
            object.__setattr__(
                self,
                "constraints_map",
                {int(key): value for key, value in self.constraints_map.items()},
            )


@pipeline_node(
    id="local_h5_area_validation_service",
    label="AreaValidationService",
    node_type="library",
    description="Prepare local H5 validation targets once per worker session.",
    source_file="policyengine_us_data/build_outputs/validation.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    artifacts_in=["policy_data.db", "target_config.yaml", "target_config_full.yaml"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_validation.py"],
)
class AreaValidationService:
    """Build validation state for all H5 requests handled by one worker."""

    def __init__(
        self,
        *,
        engine_factory: Callable[[str], Any] | None = None,
        query_targets: Callable[[Any, int], Any] | None = None,
        batch_constraints: Callable[[Any, list[int]], Mapping[int, Any]] | None = None,
        load_target_config: Callable[[Path | str], Mapping[str, Any]] | None = None,
        match_rules: Callable[[Any, list[Mapping[str, Any]]], Any] | None = None,
    ) -> None:
        """Create a validation service with injectable seams for tests."""

        self._engine_factory = engine_factory
        self._query_targets = query_targets
        self._batch_constraints = batch_constraints
        self._load_target_config = load_target_config
        self._match_rules = match_rules

    def prepare_context(
        self,
        *,
        inputs: PublishingInputBundle,
        policy: ValidationPolicy,
        period: int,
        target_config_path: Path | None = None,
        validation_config_path: Path | None = None,
    ) -> ValidationContext | None:
        """Load validation targets and constraints once for a worker.

        Returns `None` when validation is disabled. When validation is enabled
        but no target database path exists, this returns an empty context so
        callers can still inspect the policy and configured paths.
        """

        if not policy.enabled:
            return None

        if inputs.target_db_path is None:
            return ValidationContext(
                policy=policy,
                target_db_path=None,
                period=period,
                target_config_path=target_config_path,
                validation_config_path=validation_config_path,
            )

        engine = self._create_engine(Path(inputs.target_db_path))
        try:
            validation_targets = self._query_all_targets(engine, period)
            validation_targets = self._apply_validation_rules(
                validation_targets,
                validation_config_path,
            )
            training_mask = self._training_mask(
                validation_targets,
                target_config_path,
            )
            stratum_ids = [
                int(item) for item in validation_targets["stratum_id"].unique().tolist()
            ]
            constraints_map = self._load_constraints(engine, stratum_ids)
        finally:
            dispose = getattr(engine, "dispose", None)
            if callable(dispose):
                dispose()

        return ValidationContext(
            policy=policy,
            target_db_path=Path(inputs.target_db_path),
            period=period,
            validation_targets=validation_targets,
            training_mask=training_mask,
            constraints_map=constraints_map,
            target_config_path=target_config_path,
            validation_config_path=validation_config_path,
        )

    def _create_engine(self, target_db_path: Path):
        if self._engine_factory is not None:
            return self._engine_factory(f"sqlite:///{target_db_path}")

        from sqlalchemy import create_engine

        return create_engine(f"sqlite:///{target_db_path}")

    def _query_all_targets(self, engine, period: int):
        if self._query_targets is not None:
            return self._query_targets(engine, int(period))

        from policyengine_us_data.calibration.validate_staging import (
            _query_all_active_targets,
        )

        return _query_all_active_targets(engine, int(period))

    def _load_constraints(self, engine, stratum_ids: list[int]):
        if self._batch_constraints is not None:
            return self._batch_constraints(engine, stratum_ids)

        from policyengine_us_data.calibration.validate_staging import (
            _batch_stratum_constraints,
        )

        return _batch_stratum_constraints(engine, stratum_ids)

    def _config(self, path: Path | None) -> Mapping[str, Any]:
        if path is None:
            return {}

        if self._load_target_config is not None:
            return self._load_target_config(path)

        from policyengine_us_data.calibration.unified_calibration import (
            load_target_config,
        )

        return load_target_config(path)

    def _match(self, targets, rules: list[Mapping[str, Any]]):
        if self._match_rules is not None:
            return self._match_rules(targets, rules)

        from policyengine_us_data.calibration.unified_calibration import _match_rules

        return _match_rules(targets, rules)

    def _apply_validation_rules(self, validation_targets, config_path: Path | None):
        config = self._config(config_path)
        exclude_rules = list(config.get("exclude", []))
        if exclude_rules:
            exclude_mask = self._match(validation_targets, exclude_rules)
            validation_targets = validation_targets[~exclude_mask].reset_index(
                drop=True
            )

        include_rules = list(config.get("include", []))
        if include_rules:
            include_mask = self._match(validation_targets, include_rules)
            validation_targets = validation_targets[include_mask].reset_index(drop=True)

        return validation_targets

    def _training_mask(self, validation_targets, config_path: Path | None):
        config = self._config(config_path)
        include_rules = list(config.get("include", []))
        if not include_rules:
            return np.ones(len(validation_targets), dtype=bool)
        return np.asarray(self._match(validation_targets, include_rules), dtype=bool)
