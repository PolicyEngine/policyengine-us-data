from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from policyengine_us_data.utils.policyengine import (
    PolicyEngineUSBuildInfo,
    assert_locked_policyengine_us_version,
)


ENTITY_ID_VARIABLES = {
    "person": "person_id",
    "tax_unit": "tax_unit_id",
    "family": "family_id",
    "spm_unit": "spm_unit_id",
    "household": "household_id",
}


class DatasetContractError(Exception):
    """Raised when a built dataset does not match the active country package."""


@dataclass(frozen=True)
class DatasetContractSummary:
    file_path: str
    variable_count: int
    entity_counts: dict[str, int]
    policyengine_us: PolicyEngineUSBuildInfo


def _top_level_dataset_lengths(file_path: Path) -> dict[str, int]:
    dataset_lengths: dict[str, int] = {}
    with h5py.File(file_path, "r") as h5_file:
        for name in h5_file.keys():
            obj = h5_file[name]
            if isinstance(obj, h5py.Dataset):
                dataset_lengths[name] = (
                    int(obj.shape[0]) if obj.shape else int(obj.size)
                )
    return dataset_lengths


def _resolve_validation_dependencies(
    tax_benefit_system,
    microsimulation_cls,
    dataset_loader,
):
    if (
        tax_benefit_system is not None
        and microsimulation_cls is not None
        and dataset_loader is not None
    ):
        return tax_benefit_system, microsimulation_cls, dataset_loader

    from policyengine_core.data import Dataset
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    return (
        tax_benefit_system or CountryTaxBenefitSystem(),
        microsimulation_cls or Microsimulation,
        dataset_loader or Dataset.from_file,
    )


def validate_dataset_contract(
    file_path: str | Path,
    *,
    tax_benefit_system=None,
    microsimulation_cls=None,
    dataset_loader=None,
    smoke_test_variable: str = "household_weight",
) -> DatasetContractSummary:
    file_path = Path(file_path)
    policyengine_us_info = assert_locked_policyengine_us_version()
    tax_benefit_system, microsimulation_cls, dataset_loader = (
        _resolve_validation_dependencies(
            tax_benefit_system=tax_benefit_system,
            microsimulation_cls=microsimulation_cls,
            dataset_loader=dataset_loader,
        )
    )

    dataset_lengths = _top_level_dataset_lengths(file_path)
    unknown_variables = sorted(
        variable_name
        for variable_name in dataset_lengths
        if variable_name not in tax_benefit_system.variables
    )
    if unknown_variables:
        display = ", ".join(unknown_variables[:5])
        if len(unknown_variables) > 5:
            display += ", ..."
        raise DatasetContractError(
            f"{file_path.name} contains {len(unknown_variables)} variable(s) missing "
            f"from the active country package: {display}"
        )

    missing_entity_ids = [
        id_variable
        for entity_key, id_variable in ENTITY_ID_VARIABLES.items()
        if any(
            getattr(
                getattr(
                    tax_benefit_system.variables.get(variable_name), "entity", None
                ),
                "key",
                None,
            )
            == entity_key
            for variable_name in dataset_lengths
        )
        and id_variable not in dataset_lengths
    ]
    if missing_entity_ids:
        raise DatasetContractError(
            f"{file_path.name} is missing entity id variable(s): "
            + ", ".join(missing_entity_ids)
        )

    entity_counts = {
        entity_key: dataset_lengths[id_variable]
        for entity_key, id_variable in ENTITY_ID_VARIABLES.items()
        if id_variable in dataset_lengths
    }
    mismatches = []
    for variable_name, actual_length in dataset_lengths.items():
        variable = tax_benefit_system.variables.get(variable_name)
        entity_key = getattr(getattr(variable, "entity", None), "key", None)
        expected_length = entity_counts.get(entity_key)
        if expected_length is None:
            continue
        if actual_length != expected_length:
            mismatches.append(
                f"{variable_name} ({entity_key}: expected {expected_length}, found {actual_length})"
            )
    if mismatches:
        display = ", ".join(mismatches[:5])
        if len(mismatches) > 5:
            display += ", ..."
        raise DatasetContractError(
            f"{file_path.name} has inconsistent entity lengths: {display}"
        )

    dataset = dataset_loader(file_path)
    simulation = microsimulation_cls(dataset=dataset)
    if smoke_test_variable in tax_benefit_system.variables:
        result = simulation.calculate(smoke_test_variable)
        values: Any = getattr(result, "values", result)
        np.asarray(values)

    return DatasetContractSummary(
        file_path=str(file_path),
        variable_count=len(dataset_lengths),
        entity_counts=entity_counts,
        policyengine_us=policyengine_us_info,
    )
