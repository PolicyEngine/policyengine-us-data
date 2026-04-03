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


def _validate_known_variables(
    original_data: dict, tax_benefit_system, dataset_name: str
):
    unknown_variables = sorted(
        key for key in original_data if key not in tax_benefit_system.variables
    )
    if unknown_variables:
        raise ValueError(
            f"Cannot downsample {dataset_name}: found {len(unknown_variables)} "
            "dataset variables missing from the current country package "
            f"({_format_variable_list(unknown_variables)}). This usually means "
            "policyengine-us-data and policyengine-us are out of sync."
        )


def _validate_entity_lengths(
    resampled_data: dict,
    tax_benefit_system,
    dataset_name: str,
):
    entity_counts = {
        entity_key: len(np.asarray(resampled_data[id_variable]))
        for entity_key, id_variable in ENTITY_ID_VARIABLES.items()
        if id_variable in resampled_data
    }

    mismatches = []
    for variable_name, values in resampled_data.items():
        variable = tax_benefit_system.variables.get(variable_name)
        if variable is None:
            continue
        entity_key = getattr(getattr(variable, "entity", None), "key", None)
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
    _validate_known_variables(
        original_data=original_data,
        tax_benefit_system=sim.tax_benefit_system,
        dataset_name=dataset_name,
    )

    original_dtypes = {
        key: values.dtype
        for key, values in original_data.items()
        if hasattr(values, "dtype")
    }
    resampled_data = {}
    for variable_name in original_data:
        values = sim.calculate(variable_name).values
        resampled_data[variable_name] = _restore_original_dtype(
            variable_name=variable_name,
            values=values,
            original_dtype=original_dtypes.get(variable_name),
        )

    _validate_entity_lengths(
        resampled_data=resampled_data,
        tax_benefit_system=sim.tax_benefit_system,
        dataset_name=dataset_name,
    )
    return resampled_data
