import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from policyengine_data.calibration.target_rescaling import download_database

logger = logging.getLogger(__name__)


# NOTE (juaristi22): This could fail if trying to filter by more than one
# stratum constraint if there are mismatches between the filtering variable,
# value and operation.
def fetch_targets_from_database(
    engine,
    time_period: int,
    reform_id: Optional[int] = 0,
    stratum_filter_variable: Optional[str] = None,
    stratum_filter_value: Optional[str] = None,
    stratum_filter_operation: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch all targets for a specific time period and reform from the database.

    Args:
        engine: SQLAlchemy engine
        time_period: The year to fetch targets for
        reform_id: The reform scenario ID (0 for baseline)
        stratum_filter_variable: Optional variable name to filter strata by
        stratum_filter_value: Optional value to filter strata by
        stratum_filter_operation: Optional operation for filtering ('equals', 'in', etc.)

    Returns:
        DataFrame with target data including target_id, variable, value, etc.
    """
    # Base query
    query = """
    SELECT 
        t.target_id,
        t.stratum_id,
        t.variable,
        t.period,
        t.reform_id,
        t.value,
        t.active,
        t.tolerance,
        t.notes,
        s.stratum_group_id,
        s.parent_stratum_id
    FROM targets t
    JOIN strata s ON t.stratum_id = s.stratum_id
    WHERE t.period = :period
      AND t.reform_id = :reform_id
    """

    params = {"period": time_period, "reform_id": reform_id}

    # Add stratum filtering if specified
    if all(
        [
            stratum_filter_variable,
            stratum_filter_value,
            stratum_filter_operation,
        ]
    ):
        # Special case: if filtering by ucgid_str for a state, also include national targets
        if (stratum_filter_variable == "ucgid_str" and 
            stratum_filter_value and 
            stratum_filter_value.startswith("0400000US")):
            # Include both state-specific and national targets
            national_ucgid = "0100000US"
            query += """
      AND t.stratum_id IN (
          SELECT sc.stratum_id 
          FROM stratum_constraints sc 
          WHERE sc.constraint_variable = :filter_variable
            AND sc.operation = :filter_operation
            AND (sc.value = :filter_value OR sc.value = :national_value)
      )
            """
            params.update(
                {
                    "filter_variable": stratum_filter_variable,
                    "filter_operation": stratum_filter_operation,
                    "filter_value": stratum_filter_value,
                    "national_value": national_ucgid,
                }
            )
        else:
            # Standard filtering for non-geographic or non-state filters
            query += """
      AND t.stratum_id IN (
          SELECT sc.stratum_id 
          FROM stratum_constraints sc 
          WHERE sc.constraint_variable = :filter_variable
            AND sc.operation = :filter_operation
            AND sc.value = :filter_value
      )
            """
            params.update(
                {
                    "filter_variable": stratum_filter_variable,
                    "filter_operation": stratum_filter_operation,
                    "filter_value": stratum_filter_value,
                }
            )

    query += " ORDER BY t.target_id"

    return pd.read_sql(query, engine, params=params)


def fetch_stratum_constraints(engine, stratum_id: int) -> pd.DataFrame:
    """
    Fetch all constraints for a specific stratum from the database.

    Args:
        engine: SQLAlchemy engine
        stratum_id: The stratum ID

    Returns:
        DataFrame with constraint data
    """
    query = """
    SELECT 
        stratum_id,
        constraint_variable,
        value,
        operation,
        notes
    FROM stratum_constraints
    WHERE stratum_id = :stratum_id
    ORDER BY constraint_variable
    """

    return pd.read_sql(query, engine, params={"stratum_id": stratum_id})


def parse_constraint_value(value: str, operation: str):
    """
    Parse constraint value based on its type and operation.

    Args:
        value: String value from constraint
        operation: Operation type

    Returns:
        Parsed value (could be list, float, int, or string)
    """
    # Handle special operations that might use lists
    if operation == "in" and "," in value:
        # Parse as list
        return [v.strip() for v in value.split(",")]

    # Try to convert to boolean
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # Try to convert to numeric
    try:
        num_value = float(value)
        if num_value.is_integer():
            return int(num_value)
        return num_value
    except ValueError:
        return value


def apply_single_constraint(
    values: np.ndarray, operation: str, constraint_value
) -> np.ndarray:
    """
    Apply a single constraint operation to create a boolean mask.

    Args:
        values: Array of values to apply constraint to
        operation: Operation type
        constraint_value: Parsed constraint value

    Returns:
        Boolean array indicating which values meet the constraint
    """
    # TODO (bogorek): These should be in the database, with integrity enforced
    operations = {
        "equals": lambda v, cv: v == cv,
        "is_greater_than": lambda v, cv: v > cv,
        "greater_than": lambda v, cv: v > cv,
        "greater_than_or_equal": lambda v, cv: v >= cv,
        "less_than": lambda v, cv: v < cv,
        "less_than_or_equal": lambda v, cv: v <= cv,
        "not_equals": lambda v, cv: v != cv,
    }

    # TODO (bogorek): we want to fix "in". As a temporary workaround (hack), I could use this
    # section to pass in any special logic that has to do with ucgid_str values,
    # because that's what's going to show up here!
    if operation == "in":
        # Hack: since "in" is only used with ucgid_str, return everything!
        return np.ones(len(values), dtype=bool)
        #if isinstance(constraint_value, list):
        #    mask = np.zeros(len(values), dtype=bool)
        #    for cv in constraint_value:
        #        mask |= np.array(
        #            [str(cv) in str(v) for v in values], dtype=bool
        #        )
        #    return mask
        #else:
        #    return np.array(
        #        [str(constraint_value) in str(v) for v in values], dtype=bool
        #    )

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")

    result = operations[operation](values, constraint_value)
    return np.array(result, dtype=bool)


def apply_constraints_at_entity_level(
    sim, constraints_df: pd.DataFrame, target_entity: str
) -> np.ndarray:
    """
    Create a boolean mask at the target entity level by applying all constraints.

    Args:
        sim: Microsimulation instance
        constraints_df: DataFrame with constraint data
        target_entity: Entity level of the target variable ('person', 'tax_unit', 'household', etc.)

    Returns:
        Boolean array at the target entity level
    """
    # Get the number of entities at the target level
    entity_count = len(sim.calculate(f"{target_entity}_id").values)

    if constraints_df.empty:
        return np.ones(entity_count, dtype=bool)

    # Start with an open mask (all ones), then poke holes like swiss cheese
    combined_mask = np.ones(entity_count, dtype=bool)

    # Apply each constraint
    for _, constraint in constraints_df.iterrows():
        constraint_var = constraint["constraint_variable"]
        if constraint_var != 'ucgid_str':
            # NOTE: ucgid_str
            constraint_values = sim.calculate(constraint_var).values
            constraint_entity = sim.tax_benefit_system.variables[
                constraint_var
            ].entity.key

            parsed_value = parse_constraint_value(
                constraint["value"], constraint["operation"]
            )

            # Apply the constraint at its native level
            constraint_mask = apply_single_constraint(
                constraint_values, constraint["operation"], parsed_value
            )

            # Map the constraint mask to the target entity level if needed
            if constraint_entity != target_entity:
                constraint_mask = sim.map_result(
                    constraint_mask, constraint_entity, target_entity
                )

            # Ensure it's boolean
            constraint_mask = np.array(constraint_mask, dtype=bool)

            # Combine
            combined_mask = combined_mask & constraint_mask

            assert (
                len(combined_mask) == entity_count
            ), f"Combined mask length {len(combined_mask)} does not match entity count {entity_count}."

    return combined_mask


def process_single_target(
    sim,
    target: pd.Series,
    constraints_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Process a single target by applying constraints at the appropriate entity level.

    Args:
        sim: Microsimulation instance
        target: pandas Series with target data
        constraints_df: DataFrame with constraint data

    Returns:
        Tuple of (metric_values at household level, target_info_dict)
    """
    target_var = target["variable"]
    target_entity = sim.tax_benefit_system.variables[target_var].entity.key

    # Create constraint mask at the target entity level
    entity_mask = apply_constraints_at_entity_level(
        sim, constraints_df, target_entity
    )

    # Calculate the target variable at its native level
    target_values = sim.calculate(target_var).values

    # Apply the mask at the entity level
    masked_values = target_values * entity_mask
    masked_values_sum_true = masked_values.sum()

    # Map the masked result to household level
    if target_entity != "household":
        household_values = sim.map_result(
            masked_values, target_entity, "household"
        )
    else:
        household_values = masked_values

    household_values_sum = household_values.sum()

    if target_var == "person_count":
        assert (
            household_values_sum == masked_values_sum_true
        ), f"Household values sum {household_values_sum} does not match masked values sum {masked_values_sum_true} for person_count with age constraints."

    # Build target info dictionary
    target_info = {
        "name": build_target_name(target["variable"], constraints_df),
        "active": bool(target["active"]),
        "tolerance": (
            target["tolerance"] if pd.notna(target["tolerance"]) else None
        ),
    }

    return household_values, target_info


def parse_constraint_for_name(constraint: pd.Series) -> str:
    """
    Parse a single constraint into a human-readable format for naming.

    Args:
        constraint: pandas Series with constraint data

    Returns:
        Human-readable constraint description
    """
    var = constraint["constraint_variable"]
    op = constraint["operation"]
    val = constraint["value"]

    # Map operations to symbols for readability
    op_symbols = {
        "equals": "=",
        "is_greater_than": ">",
        "greater_than": ">",
        "greater_than_or_equal": ">=",
        "less_than": "<",
        "less_than_or_equal": "<=",
        "not_equals": "!=",
        "in": "in",
    }

    # Get the symbol or use the operation name if not found
    symbol = op_symbols.get(op, op)

    # Format the constraint
    if op == "in":
        # Replace commas with underscores for "in" operations
        return f"{var}_in_{val.replace(',', '_')}"
    else:
        # Use the symbol format for all other operations
        return f"{var}{symbol}{val}"


def build_target_name(variable: str, constraints_df: pd.DataFrame) -> str:
    """
    Build a descriptive name for a target with variable and constraints.

    Args:
        variable: Target variable name
        constraints_df: DataFrame with constraint data

    Returns:
        Descriptive string name
    """
    parts = [variable]

    if not constraints_df.empty:
        # Sort constraints to ensure consistent naming
        # First by whether it's ucgid, then alphabetically
        constraints_sorted = constraints_df.copy()
        constraints_sorted["is_ucgid"] = constraints_sorted[
            "constraint_variable"
        ].str.contains("ucgid")
        constraints_sorted = constraints_sorted.sort_values(
            ["is_ucgid", "constraint_variable"], ascending=[False, True]
        )

        # Add each constraint
        for _, constraint in constraints_sorted.iterrows():
            parts.append(parse_constraint_for_name(constraint))

    return "_".join(parts)


def create_metrics_matrix(
    db_uri: str,
    time_period: int,
    microsimulation_class,
    sim=None,
    dataset: Optional[type] = None,
    reform_id: Optional[int] = 0,
    stratum_filter_variable: Optional[str] = None,
    stratum_filter_value: Optional[str] = None,
    stratum_filter_operation: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[int, Dict[str, any]]]:
    """
    Create the metrics matrix from the targets database.

    This function processes all targets in the database to create a matrix where:
    - Rows represent households
    - Columns represent targets
    - Values represent the metric calculation for each household-target combination

    Args:
        db_uri: Database connection string
        time_period: Time period for the simulation
        microsimulation_class: The Microsimulation class to use for creating simulations
        sim: Optional existing Microsimulation instance
        dataset: Optional dataset type for creating new simulation
        reform_id: Reform scenario ID (0 for baseline)
        stratum_filter_variable: Optional variable name to filter strata by
        stratum_filter_value: Optional value to filter strata by
        stratum_filter_operation: Optional operation for filtering ('equals', 'in', etc.)

    Returns:
        Tuple of:
        - metrics_matrix: DataFrame with target_id as columns, households as rows
        - target_values: Array of target values in same order as columns
        - target_info: Dictionary mapping target_id to info dict with keys:
            - name: Descriptive name
            - active: Boolean active status
            - tolerance: Tolerance percentage (or None)
    """
    # Setup database connection
    engine = create_engine(db_uri)

    # Initialize simulation
    if sim is None:
        if dataset is None:
            raise ValueError("Either 'sim' or 'dataset' must be provided")
        sim = microsimulation_class(dataset=dataset)
        sim.default_calculation_period = time_period
        sim.build_from_dataset()

    # Get household IDs for matrix index
    household_ids = sim.calculate("household_id").values
    n_households = len(household_ids)

    # Fetch all targets from database
    targets_df = fetch_targets_from_database(
        engine,
        time_period,
        reform_id,
        stratum_filter_variable,
        stratum_filter_value,
        stratum_filter_operation,
    )
    logger.info(
        f"Processing {len(targets_df)} targets for period {time_period}"
    )

    # Initialize outputs
    target_values = []
    target_info = {}
    metrics_list = []
    target_ids = []

    # Process each target
    for _, target in targets_df.iterrows():
        target_id = target["target_id"]

        try:
            # Fetch constraints for this target's stratum
            constraints_df = fetch_stratum_constraints(
                engine, int(target["stratum_id"])
            )

            # Process the target
            household_values, info_dict = process_single_target(
                sim, target, constraints_df
            )

            # Store results
            metrics_list.append(household_values)
            target_ids.append(target_id)
            target_values.append(target["value"])
            target_info[target_id] = info_dict

            logger.debug(
                f"Processed target {target_id}: {info_dict['name']} "
                f"(active={info_dict['active']}, tolerance={info_dict['tolerance']})"
            )

        except Exception as e:
            logger.error(f"Error processing target {target_id}: {str(e)}")
            # Add zero column for failed targets
            metrics_list.append(np.zeros(n_households))
            target_ids.append(target_id)
            target_values.append(target["value"])
            target_info[target_id] = {
                "name": f"ERROR_{target['variable']}",
                "active": False,
                "tolerance": None,
            }

    # Create the metrics matrix DataFrame
    metrics_matrix = pd.DataFrame(
        data=np.column_stack(metrics_list),
        index=household_ids,
        columns=target_ids,
    )

    # Convert target values to numpy array
    target_values = np.array(target_values)

    logger.info(f"Created metrics matrix with shape {metrics_matrix.shape}")
    logger.info(
        f"Active targets: {sum(info['active'] for info in target_info.values())}"
    )

    return metrics_matrix, target_values, target_info


def validate_metrics_matrix(
    metrics_matrix: pd.DataFrame,
    target_values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    target_info: Optional[Dict[int, Dict[str, any]]] = None,
    raise_error: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Validate the metrics matrix by checking estimates vs targets.

    Args:
        metrics_matrix: The metrics matrix
        target_values: Array of target values
        weights: Optional weights array (defaults to uniform weights)
        target_info: Optional target info dictionary
        raise_error: Whether to raise an error for invalid estimates

    Returns:
        DataFrame with validation results
    """
    if weights is None:
        weights = np.ones(len(metrics_matrix)) / len(metrics_matrix)

    estimates = weights @ metrics_matrix.values

    if raise_error:
        for _, record in metrics_matrix.iterrows():
            if record.sum() == 0:
                raise ValueError(
                    f"Record {record.name} has all zero estimates. None of the target constraints were met by this household and its individuals."
                )
        if not np.all(estimates != 0):
            zero_indices = np.where(estimates == 0)[0]
            zero_targets = [metrics_matrix.columns[i] for i in zero_indices]
            raise ValueError(
                f"{(estimates == 0).sum()} estimate(s) contain zero values for targets: {zero_targets}"
            )

    validation_data = {
        "target_id": metrics_matrix.columns,
        "target_value": target_values,
        "estimate": estimates,
        "absolute_error": np.abs(estimates - target_values),
        "relative_error": np.abs(
            (estimates - target_values) / (target_values + 1e-10)
        ),
    }

    # Add target info if provided
    if target_info is not None:
        validation_data["name"] = [
            target_info.get(tid, {}).get("name", "Unknown")
            for tid in metrics_matrix.columns
        ]
        validation_data["active"] = [
            target_info.get(tid, {}).get("active", False)
            for tid in metrics_matrix.columns
        ]
        validation_data["tolerance"] = [
            target_info.get(tid, {}).get("tolerance", None)
            for tid in metrics_matrix.columns
        ]

    validation_df = pd.DataFrame(validation_data)

    return validation_df


if __name__ == "__main__":

    # TODO: an abstraction "leak"
    from policyengine_us import Microsimulation

    # Download the database from Hugging Face Hub
    db_uri = download_database()

    # Create metrics matrix
    metrics_matrix, target_values, target_info = create_metrics_matrix(
        db_uri=db_uri,
        time_period=2023,
        microsimulation_class=Microsimulation,
        dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
        reform_id=0,
    )

    # Validate the matrix
    validation_results = validate_metrics_matrix(
        metrics_matrix, target_values, target_info=target_info
    )

    print("\nValidation Results Summary:")
    print(f"Total targets: {len(validation_results)}")
    print(f"Active targets: {validation_results['active'].sum()}")
    print(validation_results)
