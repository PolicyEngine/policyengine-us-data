import logging
import hashlib
from typing import List, Optional

from sqlalchemy import event, UniqueConstraint
from sqlalchemy.orm.attributes import get_history

from sqlmodel import (
    Field,
    Relationship,
    SQLModel,
    create_engine,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class Strata(SQLModel, table=True):
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

    children_rel: List["Strata"] = Relationship(
        back_populates="parent_rel",
        sa_relationship_kwargs={"remote_side": "Strata.parent_stratum_id"},
    )
    parent_rel: Optional["Strata"] = Relationship(
        back_populates="children_rel",
        sa_relationship_kwargs={"remote_side": "Strata.stratum_id"},
    )
    constraints_rel: List["StratumConstraints"] = Relationship(
        back_populates="strata_rel",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "lazy": "joined",
        },
    )
    targets_rel: List["Targets"] = Relationship(
        back_populates="strata_rel",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class StratumConstraints(SQLModel, table=True):
    """Defines the rules that make up a stratum."""

    __tablename__ = "stratum_constraints"

    stratum_id: int = Field(foreign_key="strata.stratum_id", primary_key=True)
    constraint_variable: str = Field(
        primary_key=True,
        description="The variable the constraint applies to (e.g., 'age').",
    )
    operation: str = Field(
        primary_key=True,
        description="The comparison operator (e.g., 'greater_than_or_equal').",
    )
    value: str = Field(
        description="The value for the constraint rule (e.g., '25')."
    )
    notes: Optional[str] = Field(
        default=None, description="Optional notes about the constraint."
    )

    strata_rel: Strata = Relationship(back_populates="constraints_rel")


class Targets(SQLModel, table=True):
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
    variable: str = Field(
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
        default=None, description="The numerical value of the target variable."
    )
    source_id: Optional[int] = Field(
        default=None, description="Identifier for the data source."
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

    strata_rel: Strata = Relationship(back_populates="targets_rel")


# This SQLAlchemy event listener works directly with the SQLModel class
@event.listens_for(Strata, "before_insert")
@event.listens_for(Strata, "before_update")
def calculate_definition_hash(mapper, connection, target: Strata):
    """
    Calculate and set the definition_hash before saving a Strata instance.
    """
    constraints_history = get_history(target, "constraints_rel")
    if not (
        constraints_history.has_changes() or target.definition_hash is None
    ):
        return

    if not target.constraints_rel:  # Handle cases with no constraints
        target.definition_hash = hashlib.sha256(b"").hexdigest()
        return

    constraint_strings = [
        f"{c.constraint_variable}|{c.operation}|{c.value}"
        for c in target.constraints_rel
    ]

    constraint_strings.sort()
    fingerprint_text = "\n".join(constraint_strings)
    h = hashlib.sha256(fingerprint_text.encode("utf-8"))
    target.definition_hash = h.hexdigest()
    logger.info(
        f"Set definition_hash for Strata to '{target.definition_hash}'"
    )


def create_database(db_uri="sqlite:///policy_data.db"):
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
