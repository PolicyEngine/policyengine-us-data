from __future__ import annotations

import logging
from typing import Any

import numpy as np


ENTITY_ID_VARIABLES = {
    "person": "person_id",
    "tax_unit": "tax_unit_id",
    "family": "family_id",
    "spm_unit": "spm_unit_id",
    "household": "household_id",
}


def _format_variable_list(variable_names: list[str], max_display: int = 5) -> str:
    displayed = variable_names[:max_display]
    suffix = "" if len(variable_names) <= max_display else ", ..."
    return ", ".join(displayed) + suffix


def _restore_original_dtype(
    variable_name: str,
    values: Any,
    original_dtype: np.dtype | None,
):
    if original_dtype is None or not hasattr(values, "dtype"):
        return values
    if values.dtype == original_dtype:
        return values
    try:
        return values.astype(original_dtype)
    except Exception:
        logging.warning(
            "Could not convert %s back to %s after downsampling.",
            variable_name,
            original_dtype,
        )
        return values


def _infer_entity_from_length(
    variable_name: str,
    variable_length: int,
    entity_ids: dict[str, np.ndarray],
    dataset_name: str,
) -> str:
    candidate_entities = [
        entity_key
        for entity_key, ids in entity_ids.items()
        if len(np.asarray(ids)) == variable_length
    ]
    if len(candidate_entities) == 1:
        return candidate_entities[0]
    if len(candidate_entities) == 0:
        raise ValueError(
            f"Cannot downsample {dataset_name}: could not align auxiliary variable "
            f"{variable_name} (length {variable_length}) to any entity ids."
        )
    raise ValueError(
        f"Cannot downsample {dataset_name}: auxiliary variable {variable_name} "
        f"(length {variable_length}) matches multiple entity sizes "
        f"({_format_variable_list(candidate_entities)})."
    )


def _resample_auxiliary_variable(
    variable_name: str,
    original_values,
    *,
    original_entity_ids: dict[str, np.ndarray],
    resampled_entity_ids: dict[str, np.ndarray],
    dataset_name: str,
):
    entity_key = _infer_entity_from_length(
        variable_name=variable_name,
        variable_length=len(np.asarray(original_values)),
        entity_ids=original_entity_ids,
        dataset_name=dataset_name,
    )
    source_ids = np.asarray(original_entity_ids[entity_key])
    target_ids = np.asarray(resampled_entity_ids[entity_key])
    source_index = {int(entity_id): idx for idx, entity_id in enumerate(source_ids)}
    try:
        positions = [source_index[int(entity_id)] for entity_id in target_ids]
    except KeyError as exc:
        raise ValueError(
            f"Cannot downsample {dataset_name}: auxiliary variable {variable_name} "
            f"could not align entity id {exc.args[0]!r}."
        ) from exc
    return np.asarray(original_values)[positions], entity_key


def _validate_entity_lengths(
    resampled_data: dict,
    tax_benefit_system,
    dataset_name: str,
    auxiliary_entity_keys: dict[str, str] | None = None,
):
    entity_counts = {
        entity_key: len(np.asarray(resampled_data[id_variable]))
        for entity_key, id_variable in ENTITY_ID_VARIABLES.items()
        if id_variable in resampled_data
    }

    mismatches = []
    for variable_name, values in resampled_data.items():
        variable = tax_benefit_system.variables.get(variable_name)
        entity_key = getattr(getattr(variable, "entity", None), "key", None)
        if entity_key is None and auxiliary_entity_keys is not None:
            entity_key = auxiliary_entity_keys.get(variable_name)
        expected_length = entity_counts.get(entity_key)
        if expected_length is None:
            continue
        actual_length = len(np.asarray(values))
        if actual_length != expected_length:
            mismatches.append(
                f"{variable_name} ({entity_key}: expected {expected_length}, found {actual_length})"
            )

    if mismatches:
        raise ValueError(
            f"Cannot save downsampled {dataset_name}: entity lengths are inconsistent "
            f"({_format_variable_list(mismatches)})."
        )


def downsample_dataset_arrays(original_data: dict, sim, dataset_name: str) -> dict:
    original_dtypes = {
        key: values.dtype
        for key, values in original_data.items()
        if hasattr(values, "dtype")
    }
    original_entity_ids = {
        entity_key: np.asarray(original_data[id_variable])
        for entity_key, id_variable in ENTITY_ID_VARIABLES.items()
        if id_variable in original_data
    }
    resampled_data = {}
    resampled_entity_ids = {}
    for entity_key, id_variable in ENTITY_ID_VARIABLES.items():
        if (
            id_variable in original_data
            and id_variable in sim.tax_benefit_system.variables
        ):
            resampled_entity_ids[entity_key] = np.asarray(
                sim.calculate(id_variable).values
            )
    auxiliary_entity_keys = {}
    for variable_name in original_data:
        if variable_name in sim.tax_benefit_system.variables:
            values = sim.calculate(variable_name).values
        else:
            values, entity_key = _resample_auxiliary_variable(
                variable_name=variable_name,
                original_values=original_data[variable_name],
                original_entity_ids=original_entity_ids,
                resampled_entity_ids=resampled_entity_ids,
                dataset_name=dataset_name,
            )
            auxiliary_entity_keys[variable_name] = entity_key
        resampled_data[variable_name] = _restore_original_dtype(
            variable_name=variable_name,
            values=values,
            original_dtype=original_dtypes.get(variable_name),
        )

    _validate_entity_lengths(
        resampled_data=resampled_data,
        tax_benefit_system=sim.tax_benefit_system,
        dataset_name=dataset_name,
        auxiliary_entity_keys=auxiliary_entity_keys,
    )
    return resampled_data
