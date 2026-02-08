import logging
import hashlib
from typing import List, Optional
from enum import Enum

from sqlalchemy import event, UniqueConstraint
from sqlalchemy.orm.attributes import get_history
from sqlmodel import (
    Field,
    Relationship,
    SQLModel,
    create_engine,
)
from pydantic import validator
from policyengine_us.system import system

from policyengine_us_data.storage import STORAGE_FOLDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# An Enum type to ensure the variable exists in policyengine-us
USVariable = Enum(
    "USVariable", {name: name for name in system.variables.keys()}, type=str
)


class ConstraintOperation(str, Enum):
    """Allowed operations for stratum constraints."""

    EQ = "=="  # Equals
    NE = "!="  # Not equals
    GT = ">"  # Greater than
    GE = ">="  # Greater than or equal
    LT = "<"  # Less than
    LE = "<="  # Less than or equal


class Stratum(SQLModel, table=True):
    """Represents a unique population subgroup (stratum)."""

    __tablename__ = "strata"
    __table_args__ = (
        UniqueConstraint("definition_hash", name="uq_strata_definition_hash"),
    )

    stratum_id: Optional[int] = Field(
        default=None,
        primary_key=True,
        description="Unique identifier for the stratum.",
    )
    definition_hash: str = Field(
        sa_column_kwargs={
            "comment": "SHA-256 hash of the stratum's constraints."
        },
        max_length=64,
    )
    parent_stratum_id: Optional[int] = Field(
        default=None,
        foreign_key="strata.stratum_id",
        index=True,
        description="Identifier for a parent stratum, creating a hierarchy.",
    )
    stratum_group_id: Optional[int] = Field(
        default=None, description="Identifier for a group of related strata."
    )
    notes: Optional[str] = Field(
        default=None, description="Descriptive notes about the stratum."
    )

    children_rel: List["Stratum"] = Relationship(
        back_populates="parent_rel",
        sa_relationship_kwargs={"remote_side": "Stratum.parent_stratum_id"},
    )
    parent_rel: Optional["Stratum"] = Relationship(
        back_populates="children_rel",
        sa_relationship_kwargs={"remote_side": "Stratum.stratum_id"},
    )
    constraints_rel: List["StratumConstraint"] = Relationship(
        back_populates="strata_rel",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "lazy": "joined",
        },
    )
    targets_rel: List["Target"] = Relationship(
        back_populates="strata_rel",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class StratumConstraint(SQLModel, table=True):
    """Defines the rules that make up a stratum."""

    __tablename__ = "stratum_constraints"

    stratum_id: int = Field(foreign_key="strata.stratum_id", primary_key=True)
    constraint_variable: str = Field(
        primary_key=True,
        description="The variable the constraint applies to (e.g., 'age').",
    )
    operation: str = Field(
        primary_key=True,
        description="The comparison operator (==, !=, >, >=, <, <=).",
    )
    value: str = Field(
        description="The value for the constraint rule (e.g., '25')."
    )
    notes: Optional[str] = Field(
        default=None, description="Optional notes about the constraint."
    )

    strata_rel: Stratum = Relationship(back_populates="constraints_rel")

    @validator("operation")
    def validate_operation(cls, v):
        """Validate that the operation is one of the allowed values."""
        allowed_ops = [op.value for op in ConstraintOperation]
        if v not in allowed_ops:
            raise ValueError(
                f"Invalid operation '{v}'. Must be one of: {', '.join(allowed_ops)}"
            )
        return v


class Target(SQLModel, table=True):
    """Stores the data values for a specific stratum."""

    __tablename__ = "targets"
    __table_args__ = (
        UniqueConstraint(
            "variable",
            "period",
            "stratum_id",
            "reform_id",
            name="_target_unique",
        ),
    )

    target_id: Optional[int] = Field(default=None, primary_key=True)
    variable: USVariable = Field(
        description="A variable defined in policyengine-us (e.g., 'income_tax')."
    )
    period: int = Field(
        description="The time period for the data, typically a year."
    )
    stratum_id: int = Field(foreign_key="strata.stratum_id", index=True)
    reform_id: int = Field(
        default=0,
        description="Identifier for a policy reform scenario (0 for baseline).",
    )
    value: Optional[float] = Field(
        default=None,
        description="The numerical value of the target variable.",
    )
    raw_value: Optional[float] = Field(
        default=None,
        description="Original value from the data source, before reconciliation.",
    )
    source_id: Optional[int] = Field(
        default=None,
        foreign_key="sources.source_id",
        description="Identifier for the data source.",
    )
    active: bool = Field(
        default=True,
        description="Flag to indicate if the record is currently active.",
    )
    tolerance: Optional[float] = Field(
        default=None,
        description="Allowed relative error as a percent (e.g., 25 for 25%).",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional descriptive notes about the target row.",
    )

    strata_rel: Stratum = Relationship(back_populates="targets_rel")
    source_rel: Optional["Source"] = Relationship()


class SourceType(str, Enum):
    """Types of data sources."""

    ADMINISTRATIVE = "administrative"
    SURVEY = "survey"
    SYNTHETIC = "synthetic"
    DERIVED = "derived"
    HARDCODED = (
        "hardcoded"  # Values from various sources, hardcoded into the system
    )


class Source(SQLModel, table=True):
    """Metadata about data sources."""

    __tablename__ = "sources"
    __table_args__ = (
        UniqueConstraint("name", "vintage", name="uq_source_name_vintage"),
    )

    source_id: Optional[int] = Field(
        default=None,
        primary_key=True,
        description="Unique identifier for the data source.",
    )
    name: str = Field(
        description="Name of the data source (e.g., 'IRS SOI', 'Census ACS').",
        index=True,
    )
    type: SourceType = Field(
        description="Type of data source (administrative, survey, etc.)."
    )
    description: Optional[str] = Field(
        default=None, description="Detailed description of the data source."
    )
    url: Optional[str] = Field(
        default=None,
        description="URL or reference to the original data source.",
    )
    vintage: Optional[str] = Field(
        default=None, description="Version or release date of the data source."
    )
    notes: Optional[str] = Field(
        default=None, description="Additional notes about the source."
    )


class VariableGroup(SQLModel, table=True):
    """Groups of related variables that form logical units."""

    __tablename__ = "variable_groups"

    group_id: Optional[int] = Field(
        default=None,
        primary_key=True,
        description="Unique identifier for the variable group.",
    )
    name: str = Field(
        description="Name of the variable group (e.g., 'age_distribution', 'snap_recipients').",
        index=True,
        unique=True,
    )
    category: str = Field(
        description="High-level category (e.g., 'demographic', 'benefit', 'tax', 'income').",
        index=True,
    )
    is_histogram: bool = Field(
        default=False,
        description="Whether this group represents a histogram/distribution.",
    )
    is_exclusive: bool = Field(
        default=False,
        description="Whether variables in this group are mutually exclusive.",
    )
    aggregation_method: Optional[str] = Field(
        default=None,
        description="How to aggregate variables in this group (sum, weighted_avg, etc.).",
    )
    display_order: Optional[int] = Field(
        default=None,
        description="Order for displaying this group in matrices/reports.",
    )
    description: Optional[str] = Field(
        default=None, description="Description of what this group represents."
    )


class VariableMetadata(SQLModel, table=True):
    """Maps PolicyEngine variables to their groups and provides metadata."""

    __tablename__ = "variable_metadata"
    __table_args__ = (
        UniqueConstraint("variable", name="uq_variable_metadata_variable"),
    )

    metadata_id: Optional[int] = Field(default=None, primary_key=True)
    variable: str = Field(
        description="PolicyEngine variable name.", index=True
    )
    group_id: Optional[int] = Field(
        default=None,
        foreign_key="variable_groups.group_id",
        description="ID of the variable group this belongs to.",
    )
    display_name: Optional[str] = Field(
        default=None,
        description="Human-readable name for display in matrices.",
    )
    display_order: Optional[int] = Field(
        default=None,
        description="Order within its group for display purposes.",
    )
    units: Optional[str] = Field(
        default=None,
        description="Units of measurement (dollars, count, percent, etc.).",
    )
    is_primary: bool = Field(
        default=True,
        description="Whether this is a primary variable vs derived/auxiliary.",
    )
    notes: Optional[str] = Field(
        default=None, description="Additional notes about the variable."
    )

    group_rel: Optional[VariableGroup] = Relationship()


# This SQLAlchemy event listener works directly with the SQLModel class
@event.listens_for(Stratum, "before_insert")
@event.listens_for(Stratum, "before_update")
def calculate_definition_hash(mapper, connection, target: Stratum):
    """
    Calculate and set the definition_hash before saving a Stratum instance.
    """
    constraints_history = get_history(target, "constraints_rel")
    if not (
        constraints_history.has_changes() or target.definition_hash is None
    ):
        return

    if not target.constraints_rel:  # Handle cases with no constraints
        # Include parent_stratum_id to make hash unique per parent
        parent_str = (
            str(target.parent_stratum_id) if target.parent_stratum_id else ""
        )
        target.definition_hash = hashlib.sha256(
            parent_str.encode("utf-8")
        ).hexdigest()
        return

    constraint_strings = [
        f"{c.constraint_variable}|{c.operation}|{c.value}"
        for c in target.constraints_rel
    ]

    constraint_strings.sort()
    # Include parent_stratum_id in the hash to ensure uniqueness per parent
    parent_str = (
        str(target.parent_stratum_id) if target.parent_stratum_id else ""
    )
    fingerprint_text = parent_str + "\n" + "\n".join(constraint_strings)
    h = hashlib.sha256(fingerprint_text.encode("utf-8"))
    target.definition_hash = h.hexdigest()


def create_database(
    db_uri: str = f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}",
):
    """
    Creates a SQLite database and all the defined tables.

    Args:
        db_uri (str): The connection string for the database.

    Returns:
        An SQLAlchemy Engine instance connected to the database.
    """
    engine = create_engine(db_uri)
    SQLModel.metadata.create_all(engine)
    logger.info(f"Database and tables created successfully at {db_uri}")
    return engine


if __name__ == "__main__":
    engine = create_database()
