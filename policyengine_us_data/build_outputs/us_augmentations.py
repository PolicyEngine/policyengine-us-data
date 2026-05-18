"""US-specific payload postprocessors for local H5 outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from policyengine_us.variables.gov.hud.is_eligible_for_housing_assistance import (
    housing_assistance_eligibility_from_income_limits,
)

from policyengine_us_data.calibration.block_assignment import (
    derive_geography_from_blocks,
)
from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.utils.takeup import (
    SIMPLE_TAKEUP_VARS,
    _sum_person_values_to_tax_units,
    apply_block_takeup_to_arrays,
    reported_subsidized_marketplace_by_tax_unit,
)

from .builder import PayloadPostProcessorSpec
from .payload import H5Payload, PayloadBuildContext
from .selection import CloneSelection
from .simulation_access import calculate_variable_values
from .variables import GEOGRAPHY_VARIABLES

__all__ = [
    "TAKEUP_VARIABLE_ENTITIES",
    "US_ENTITY_POSTPROCESSOR_KEY",
    "US_GEOGRAPHY_POSTPROCESSOR_KEY",
    "US_TAKEUP_POSTPROCESSOR_KEY",
    "USEntityPostProcessor",
    "USEntityPostProcessorResult",
    "USGeographyPostProcessor",
    "USGeographyPostProcessorResult",
    "USTakeupPostProcessor",
    "USTakeupPostProcessorResult",
    "default_us_postprocessors",
]

PeriodData = dict[Any, np.ndarray]
PayloadData = dict[str, PeriodData]
GeographyDeriver = Callable[[np.ndarray], Mapping[str, np.ndarray]]
TakeupApplier = Callable[..., Mapping[str, np.ndarray]]
TAKEUP_VARIABLE_ENTITIES = {
    str(spec["variable"]): str(spec["entity"]) for spec in SIMPLE_TAKEUP_VARS
}
REQUIRED_TAKEUP_SUBENTITIES = ("tax_unit", "spm_unit")
US_ENTITY_POSTPROCESSOR_KEY = "us_entity"
US_GEOGRAPHY_POSTPROCESSOR_KEY = "us_geography"
US_TAKEUP_POSTPROCESSOR_KEY = "us_takeup"


@pipeline_node(
    id="local_h5_us_entity_postprocessor_result",
    label="USEntityPostProcessorResult",
    node_type="library",
    description="US entity ID and household-weight local H5 payload data.",
    source_file="policyengine_us_data/build_outputs/us_augmentations.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_us_augmentations.py"
    ],
)
@dataclass(frozen=True)
class USEntityPostProcessorResult:
    """Payload after US entity ID and household-weight fields are applied."""

    payload: H5Payload

    @property
    def data(self) -> PayloadData:
        """Augmented payload data retained for transitional callers."""

        return self.payload.data


@pipeline_node(
    id="local_h5_us_geography_postprocessor_result",
    label="USGeographyPostProcessorResult",
    node_type="library",
    description="US geography local H5 payload data and block-derived geography.",
    source_file="policyengine_us_data/build_outputs/us_augmentations.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_us_augmentations.py"
    ],
)
@dataclass(frozen=True)
class USGeographyPostProcessorResult:
    """Payload after US geography fields are applied."""

    payload: H5Payload
    clone_geography: Mapping[str, np.ndarray]

    @property
    def data(self) -> PayloadData:
        """Augmented payload data retained for transitional callers."""

        return self.payload.data


@pipeline_node(
    id="local_h5_us_takeup_postprocessor_result",
    label="USTakeupPostProcessorResult",
    node_type="library",
    description="US take-up local H5 payload data and generated take-up variables.",
    source_file="policyengine_us_data/build_outputs/us_augmentations.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_us_augmentations.py"
    ],
)
@dataclass(frozen=True)
class USTakeupPostProcessorResult:
    """Payload after US take-up fields are applied."""

    payload: H5Payload
    takeup_variables: tuple[str, ...] = ()

    @property
    def data(self) -> PayloadData:
        """Augmented payload data retained for transitional callers."""

        return self.payload.data


@pipeline_node(
    id="local_h5_us_entity_postprocessor",
    label="USEntityPostProcessor",
    node_type="library",
    description="Apply US entity ID and household-weight fields to local H5 payloads.",
    source_file="policyengine_us_data/build_outputs/us_augmentations.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_us_augmentations.py"
    ],
)
@dataclass(frozen=True)
class USEntityPostProcessor:
    """Apply US entity IDs and calibrated household weights."""

    spec = PayloadPostProcessorSpec(key=US_ENTITY_POSTPROCESSOR_KEY)

    def apply(
        self,
        *,
        payload: H5Payload,
        context: PayloadBuildContext,
    ) -> USEntityPostProcessorResult:
        """Return a payload with structural ID and weight overrides applied."""

        output = _copy_payload(payload.data)
        time_period = context.time_period
        reindexed = context.reindexed
        output["household_id"] = {time_period: reindexed.household_ids}
        output["person_id"] = {time_period: reindexed.person_ids}
        output["person_household_id"] = {
            time_period: reindexed.person_household_ids,
        }
        for entity_key, entity_ids in reindexed.subentity_ids.items():
            output[f"{entity_key}_id"] = {time_period: entity_ids}
            output[f"person_{entity_key}_id"] = {
                time_period: reindexed.person_subentity_ids[entity_key],
            }
        output["household_weight"] = {
            time_period: context.selection.weights.astype(np.float32),
        }
        return USEntityPostProcessorResult(
            payload=H5Payload(
                data=output,
                time_period=payload.time_period,
                entity_lengths=payload.entity_lengths,
                variable_entities=payload.variable_entities,
            ),
        )


@pipeline_node(
    id="local_h5_us_geography_postprocessor",
    label="USGeographyPostProcessor",
    node_type="library",
    description="Apply US geography fields to local H5 payloads.",
    source_file="policyengine_us_data/build_outputs/us_augmentations.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_us_augmentations.py"
    ],
)
@dataclass(frozen=True)
class USGeographyPostProcessor:
    """Apply block-derived US geography overrides."""

    spec = PayloadPostProcessorSpec(key=US_GEOGRAPHY_POSTPROCESSOR_KEY)

    geography_deriver: GeographyDeriver = derive_geography_from_blocks
    _string_geography_variables: tuple[str, ...] = field(
        default=GEOGRAPHY_VARIABLES,
        init=False,
        repr=False,
    )

    def apply(
        self,
        *,
        payload: H5Payload,
        context: PayloadBuildContext,
    ) -> USGeographyPostProcessorResult:
        """Return a payload with block-derived geography fields applied."""

        output = _copy_payload(payload.data)
        clone_geography = self._derive_clone_geography(context.selection)
        self._apply_geography_overrides(output, clone_geography, context)
        return USGeographyPostProcessorResult(
            payload=H5Payload(
                data=output,
                time_period=payload.time_period,
                entity_lengths=payload.entity_lengths,
                variable_entities=payload.variable_entities,
            ),
            clone_geography=clone_geography,
        )

    def _derive_clone_geography(
        self,
        selection: CloneSelection,
    ) -> Mapping[str, np.ndarray]:
        unique_blocks, block_inverse = np.unique(
            selection.block_geoids,
            return_inverse=True,
        )
        unique_geography = self.geography_deriver(unique_blocks)
        return {
            key: np.asarray(values)[block_inverse]
            for key, values in unique_geography.items()
        }

    def _apply_geography_overrides(
        self,
        data: PayloadData,
        clone_geography: Mapping[str, np.ndarray],
        context: PayloadBuildContext,
    ) -> None:
        time_period = context.time_period
        data["state_fips"] = {
            time_period: np.asarray(clone_geography["state_fips"]).astype(np.int32),
        }
        data["county"] = {
            time_period: np.asarray(clone_geography["county_index"]).astype(np.int32),
        }
        data["county_fips"] = {
            time_period: np.asarray(clone_geography["county_fips"]).astype(np.int32),
        }
        for variable in self._string_geography_variables:
            if variable in clone_geography:
                data[variable] = {
                    time_period: np.asarray(clone_geography[variable]).astype("S"),
                }
        self._apply_los_angeles_zip_patch(data, clone_geography, time_period)
        data["congressional_district_geoid"] = {
            time_period: np.asarray(
                [int(cd) for cd in context.selection.congressional_district_geoids],
                dtype=np.int32,
            ),
        }

    def _apply_los_angeles_zip_patch(
        self,
        data: PayloadData,
        clone_geography: Mapping[str, np.ndarray],
        time_period: int,
    ) -> None:
        county_fips = np.asarray(clone_geography["county_fips"]).astype(str)
        los_angeles_mask = county_fips == "06037"
        if not los_angeles_mask.any():
            return
        zip_codes = np.full(len(los_angeles_mask), "00000")
        zip_codes[los_angeles_mask] = "90001"
        data["zip_code"] = {time_period: zip_codes.astype("S")}


@pipeline_node(
    id="local_h5_us_takeup_postprocessor",
    label="USTakeupPostProcessor",
    node_type="library",
    description="Apply US take-up fields to local H5 payloads.",
    source_file="policyengine_us_data/build_outputs/us_augmentations.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_us_augmentations.py"
    ],
)
@dataclass(frozen=True)
class USTakeupPostProcessor:
    """Apply US take-up draws after entity and geography postprocessing."""

    spec = PayloadPostProcessorSpec(
        key=US_TAKEUP_POSTPROCESSOR_KEY,
        requires=(US_ENTITY_POSTPROCESSOR_KEY, US_GEOGRAPHY_POSTPROCESSOR_KEY),
    )
    takeup_applier: TakeupApplier = apply_block_takeup_to_arrays
    sum_person_values_to_tax_units: Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
    ] = _sum_person_values_to_tax_units

    def apply(
        self,
        *,
        payload: H5Payload,
        context: PayloadBuildContext,
    ) -> USTakeupPostProcessorResult:
        """Return a payload with take-up variables applied."""

        self._validate_required_subentities(context)
        output = _copy_payload(payload.data)
        time_period = context.time_period
        self._validate_required_payload_fields(output, time_period)
        results = self._build_takeup_results(output, context)
        takeup_variables = tuple(str(variable) for variable in results)
        self._validate_takeup_variables(takeup_variables)
        for variable, values in results.items():
            output[str(variable)] = {time_period: np.asarray(values)}
        variable_entities = {
            **payload.variable_entities,
            **{
                variable: TAKEUP_VARIABLE_ENTITIES[variable]
                for variable in takeup_variables
            },
        }
        return USTakeupPostProcessorResult(
            payload=H5Payload(
                data=output,
                time_period=payload.time_period,
                entity_lengths=payload.entity_lengths,
                variable_entities=variable_entities,
            ),
            takeup_variables=takeup_variables,
        )

    def _validate_required_subentities(self, context: PayloadBuildContext) -> None:
        reindexed = context.reindexed
        missing = [
            entity_key
            for entity_key in REQUIRED_TAKEUP_SUBENTITIES
            if entity_key not in reindexed.subentity_ids
            or entity_key not in reindexed.person_subentity_ids
            or entity_key not in reindexed.subentity_source_indices
            or entity_key not in reindexed.subentity_household_clone_indices
        ]
        if missing:
            raise ValueError(
                f"US take-up requires reindexed subentities: {', '.join(missing)}"
            )

    def _validate_required_payload_fields(
        self,
        data: PayloadData,
        time_period: int,
    ) -> None:
        _required_period_array(
            data,
            "state_fips",
            time_period,
            "US take-up requires state_fips from USGeographyPostProcessor",
        )
        if _has_period_array(
            data,
            "reported_has_subsidized_marketplace_health_coverage_at_interview",
            time_period,
        ):
            for variable in ("person_tax_unit_id", "tax_unit_id"):
                _required_period_array(
                    data,
                    variable,
                    time_period,
                    "US take-up reported ACA anchors require "
                    "person_tax_unit_id and tax_unit_id from "
                    "USEntityPostProcessor",
                )

    def _build_takeup_results(
        self,
        data: PayloadData,
        context: PayloadBuildContext,
    ) -> Mapping[str, np.ndarray]:
        reindexed = context.reindexed
        selection = context.selection
        time_period = context.time_period
        subentity_source_indices = reindexed.subentity_source_indices
        subentity_ids = reindexed.subentity_ids
        person_subentity_ids = reindexed.person_subentity_ids
        state_fips = _required_period_array(
            data,
            "state_fips",
            time_period,
            "US take-up requires state_fips from USGeographyPostProcessor",
        )

        entity_hh_indices = {
            "person": reindexed.person_household_clone_indices.astype(np.int64),
            "tax_unit": reindexed.subentity_household_clone_indices["tax_unit"].astype(
                np.int64
            ),
            "spm_unit": reindexed.subentity_household_clone_indices["spm_unit"].astype(
                np.int64
            ),
        }
        entity_counts = {
            "person": len(reindexed.person_ids),
            "tax_unit": len(subentity_source_indices["tax_unit"]),
            "spm_unit": len(subentity_source_indices["spm_unit"]),
        }
        reported_anchors = _build_reported_takeup_anchors(data, time_period)
        eligibility_masks = self._build_eligibility_masks(
            data=data,
            context=context,
            subentity_source_indices=subentity_source_indices,
        )
        voluntary_filing_inputs = self._build_voluntary_filing_inputs(
            context=context,
            tax_unit_source_indices=subentity_source_indices["tax_unit"],
            new_tax_unit_ids=subentity_ids["tax_unit"],
            new_person_tax_unit_ids=person_subentity_ids["tax_unit"],
        )
        return self.takeup_applier(
            hh_blocks=selection.block_geoids,
            hh_state_fips=np.asarray(state_fips).astype(np.int32),
            hh_ids=context.source.household_ids[
                selection.source_household_indices
            ].astype(np.int64),
            hh_clone_indices=selection.clone_indices.astype(np.int64),
            entity_hh_indices=entity_hh_indices,
            entity_counts=entity_counts,
            time_period=time_period,
            takeup_filter=(
                list(context.takeup_filter)
                if context.takeup_filter is not None
                else None
            ),
            reported_anchors=reported_anchors,
            eligibility_masks=eligibility_masks,
            voluntary_filing_inputs=voluntary_filing_inputs,
        )

    def _validate_takeup_variables(self, takeup_variables: tuple[str, ...]) -> None:
        unknown_variables = [
            variable
            for variable in takeup_variables
            if variable not in TAKEUP_VARIABLE_ENTITIES
        ]
        if unknown_variables:
            raise ValueError(
                "Unknown take-up variable(s) returned by takeup applier: "
                f"{', '.join(unknown_variables)}"
            )

    def _build_voluntary_filing_inputs(
        self,
        *,
        context: PayloadBuildContext,
        tax_unit_source_indices: np.ndarray,
        new_tax_unit_ids: np.ndarray,
        new_person_tax_unit_ids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        time_period = context.time_period
        return {
            "tax_unit_child_dependents": calculate_variable_values(
                context.simulation,
                "tax_unit_child_dependents",
                period=time_period,
                map_to="tax_unit",
            )[tax_unit_source_indices],
            "tax_unit_wage_income": self.sum_person_values_to_tax_units(
                calculate_variable_values(
                    context.simulation,
                    "employment_income",
                    period=time_period,
                    map_to="person",
                )[context.reindexed.person_source_indices],
                new_person_tax_unit_ids,
                new_tax_unit_ids,
            ),
            "age_head": calculate_variable_values(
                context.simulation,
                "age_head",
                period=time_period,
                map_to="tax_unit",
            )[tax_unit_source_indices],
        }

    def _build_eligibility_masks(
        self,
        *,
        data: PayloadData,
        context: PayloadBuildContext,
        subentity_source_indices: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        time_period = context.time_period
        spm_unit_source_indices = subentity_source_indices["spm_unit"]
        spm_unit_household_indices = (
            context.reindexed.subentity_household_clone_indices["spm_unit"].astype(
                np.int64
            )
        )
        household_county_fips = _required_period_array(
            data,
            "county_fips",
            time_period,
            "US take-up housing assistance eligibility requires county_fips "
            "from USGeographyPostProcessor",
        )
        if (
            "receives_housing_assistance" in data
            and time_period in data["receives_housing_assistance"]
        ):
            receives_housing_assistance = data["receives_housing_assistance"][
                time_period
            ].astype(bool)
        else:
            receives_housing_assistance = calculate_variable_values(
                context.simulation,
                "receives_housing_assistance",
                period=time_period,
                map_to="spm_unit",
            )[spm_unit_source_indices].astype(bool)

        return {
            "takes_up_housing_assistance_if_eligible": (
                housing_assistance_eligibility_from_income_limits(
                    county_fips=np.asarray(household_county_fips)[
                        spm_unit_household_indices
                    ],
                    annual_income=calculate_variable_values(
                        context.simulation,
                        "hud_annual_income",
                        period=time_period,
                        map_to="spm_unit",
                    )[spm_unit_source_indices],
                    spm_unit_size=calculate_variable_values(
                        context.simulation,
                        "spm_unit_size",
                        period=time_period,
                        map_to="spm_unit",
                    )[spm_unit_source_indices],
                    spm_unit_tenure_type=calculate_variable_values(
                        context.simulation,
                        "spm_unit_tenure_type",
                        period=time_period,
                        map_to="spm_unit",
                    )[spm_unit_source_indices],
                    receives_housing_assistance=receives_housing_assistance,
                    year=time_period,
                ).astype(bool)
            )
        }


def default_us_postprocessors() -> tuple[
    USEntityPostProcessor | USGeographyPostProcessor | USTakeupPostProcessor, ...
]:
    """Return production US postprocessors in their required order."""

    return (
        USEntityPostProcessor(),
        USGeographyPostProcessor(),
        USTakeupPostProcessor(),
    )


def _copy_payload(data: Mapping[str, Mapping[Any, np.ndarray]]) -> PayloadData:
    return {variable: dict(periods) for variable, periods in data.items()}


def _required_period_array(
    data: Mapping[str, Mapping[Any, np.ndarray]],
    variable: str,
    time_period: int,
    message: str,
) -> np.ndarray:
    if variable not in data or time_period not in data[variable]:
        raise ValueError(message)
    return np.asarray(data[variable][time_period])


def _has_period_array(
    data: Mapping[str, Mapping[Any, np.ndarray]],
    variable: str,
    time_period: int,
) -> bool:
    return variable in data and time_period in data[variable]


def _build_reported_takeup_anchors(
    data: Mapping[str, Mapping[Any, np.ndarray]],
    time_period: int,
) -> dict[str, np.ndarray]:
    reported_anchors = {}
    if (
        "reported_has_subsidized_marketplace_health_coverage_at_interview" in data
        and time_period
        in data["reported_has_subsidized_marketplace_health_coverage_at_interview"]
    ):
        reported_anchors["takes_up_aca_if_eligible"] = (
            reported_subsidized_marketplace_by_tax_unit(
                data["person_tax_unit_id"][time_period],
                data["tax_unit_id"][time_period],
                data[
                    "reported_has_subsidized_marketplace_health_coverage_at_interview"
                ][time_period],
            )
        )
    if (
        "has_medicaid_health_coverage_at_interview" in data
        and time_period in data["has_medicaid_health_coverage_at_interview"]
    ):
        reported_anchors["takes_up_medicaid_if_eligible"] = data[
            "has_medicaid_health_coverage_at_interview"
        ][time_period].astype(bool)
    if (
        "receives_housing_assistance" in data
        and time_period in data["receives_housing_assistance"]
    ):
        reported_anchors["takes_up_housing_assistance_if_eligible"] = data[
            "receives_housing_assistance"
        ][time_period].astype(bool)
    return reported_anchors
