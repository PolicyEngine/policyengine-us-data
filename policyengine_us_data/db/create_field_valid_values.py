"""Populate the field_valid_values table with valid values for semantic fields.

This module provides functionality to populate the field_valid_values table
with static values (operations, active flags, periods) and dynamic values
(policyengine-us variables).
"""

import logging
from typing import Optional

from sqlmodel import Field, Session, SQLModel

from policyengine_us.system import system


class FieldValidValues(SQLModel, table=True):
    """Lookup table for valid field values, enforced by SQL triggers."""

    __tablename__ = "field_valid_values"

    field_name: str = Field(primary_key=True)
    valid_value: str = Field(primary_key=True)
    description: Optional[str] = Field(default=None)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def populate_field_valid_values(session: Session) -> None:
    """Populate the field_valid_values table with valid values for fields.

    This function populates the table with:
    - Static values: operation, active, period
    - Dynamic values: variable, constraint_variable (from policyengine-us)

    Args:
        session: SQLModel Session instance for database operations.
    """
    # Static values for operation field
    operation_values = [
        ("operation", "==", "Equals"),
        ("operation", "!=", "Not equals"),
        ("operation", ">", "Greater than"),
        ("operation", ">=", "Greater than or equal"),
        ("operation", "<", "Less than"),
        ("operation", "<=", "Less than or equal"),
    ]

    # Static values for active field
    active_values = [
        ("active", "0", "Inactive"),
        ("active", "1", "Active"),
    ]

    # Static values for period field (years)
    period_values = [
        ("period", "2022", None),
        ("period", "2023", None),
        ("period", "2024", None),
        ("period", "2025", None),
    ]

    # Add all static values
    static_count = 0
    for field_name, valid_value, description in (
        operation_values + active_values + period_values
    ):
        session.add(
            FieldValidValues(
                field_name=field_name,
                valid_value=valid_value,
                description=description,
            )
        )
        static_count += 1

    # Dynamic values from policyengine-us
    variable_count = 0
    for var_name in system.variables.keys():
        # Add for 'variable' field (targets table)
        session.add(
            FieldValidValues(
                field_name="variable",
                valid_value=var_name,
            )
        )
        # Add for 'constraint_variable' field (stratum_constraints table)
        session.add(
            FieldValidValues(
                field_name="constraint_variable",
                valid_value=var_name,
            )
        )
        variable_count += 1

    # Domain-specific constraint variables (not in policyengine-us)
    extra_constraint_vars = [
        ("constraint_variable", "ucgid_str", "Census UCGID string"),
    ]
    for field_name, valid_value, description in extra_constraint_vars:
        session.add(
            FieldValidValues(
                field_name=field_name,
                valid_value=valid_value,
                description=description,
            )
        )

    session.commit()

    logger.info(
        f"Populated field_valid_values with {static_count} static values "
        f"and {variable_count * 2} variable values "
        f"({variable_count} variables x 2 fields)"
    )


if __name__ == "__main__":
    from sqlmodel import create_engine

    from policyengine_us_data.storage import STORAGE_FOLDER

    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    engine = create_engine(f"sqlite:///{db_path}")

    with Session(engine) as session:
        populate_field_valid_values(session)
