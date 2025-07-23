import io

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    ForeignKey,
    PrimaryKeyConstraint,
)
from sqlalchemy.orm import relationship


Base = declarative_base()

## Data Models


class Stratum(Base):
    """
    Represents a stratum, which is a specific subgroup of a population.

    This table acts as the parent for stratum-related data, defining each unique
    stratum with an ID.
    """

    __tablename__ = "stratum"

    stratum_id = Column(
        Integer, primary_key=True, comment="Unique identifier for the stratum."
    )
    stratum_group_id = Column(
        Integer, comment="Identifier for a group of related strata."
    )
    parent_stratum_id = Column(
        Integer,
        comment="Identifier for a parent stratum, creating a hierarchy.",
    )
    notes = Column(String, comment="Descriptive notes about the stratum.")

    # Define one-to-many relationships
    # A stratum can have multiple constraints and multiple target values.
    constraints = relationship(
        "StratumConstraint",
        back_populates="stratum",
        cascade="all, delete-orphan",
    )
    targets = relationship(
        "Target", back_populates="stratum", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Stratum(stratum_id={self.stratum_id}, notes='{self.notes}')>"


class StratumConstraint(Base):
    """
    Defines the specific rules or conditions that make up a stratum.

    For example, a stratum for 'high-income individuals in New York' would have
    constraints for both income level and state.
    """

    __tablename__ = "stratum_constraint"

    # A composite primary key is used because a single constraint is uniquely
    # defined by the combination of these four columns.
    stratum_id = Column(
        Integer, ForeignKey("stratum.stratum_id"), primary_key=True
    )
    constraint_variable = Column(
        String,
        primary_key=True,
        comment="The variable the constraint applies to (e.g., 'age').",
    )
    value = Column(
        String,
        primary_key=True,
        comment="The value for the constraint rule (e.g., '25' or '[NY,NJ]').",
    )
    operation = Column(
        String,
        primary_key=True,
        comment="The comparison operator (e.g., 'greater_than_or_equal', 'in').",
    )
    notes = Column(
        String, nullable=True, comment="Optional notes about the constraint."
    )

    # Define the many-to-one relationship back to the Stratum table
    stratum = relationship("Stratum", back_populates="constraints")

    def __repr__(self):
        return f"<StratumConstraint(stratum_id={self.stratum_id}, variable='{self.constraint_variable}', operation='{self.operation}')>"


class Target(Base):
    """
    Stores the actual data values for different variables within a specific
    stratum, period, and reform scenario.
    """

    __tablename__ = "target"

    # The primary key will auto-increment, so you don't need to specify it on insert.
    target_id = Column(Integer, primary_key=True)
    variable = Column(
        String,
        nullable=False,
        comment="The name of the data variable (e.g., 'income_tax').",
    )
    period = Column(
        Integer,
        nullable=False,
        comment="The time period for the data, typically a year.",
    )

    # A standard foreign key ensures that the stratum_id must exist in the 'stratum' table.
    # NOTE: Your requirement that the stratum_id must *also* exist in the 'stratum_constraint'
    # table is a business rule that should be enforced in your application logic
    # before you attempt to insert data into this table.
    stratum_id = Column(
        Integer, ForeignKey("stratum.stratum_id"), nullable=False, index=True
    )

    reform_id = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Identifier for a policy reform scenario (0 for baseline).",
    )
    value = Column(
        Float, comment="The numerical value of the target variable."
    )
    source_id = Column(Integer, comment="Identifier for the data source.")
    active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Flag to indicate if the record is currently active.",
    )

    # Define the many-to-one relationship back to the Stratum table
    stratum = relationship("Stratum", back_populates="targets")

    def __repr__(self):
        return f"<Target(target_id={self.target_id}, variable='{self.variable}', value={self.value})>"


## Database Creation Example


def create_database(db_uri="sqlite:///policy_data.db"):
    """
    Creates a SQLite database and all the defined tables.

    Args:
        db_uri (str): The connection string for the database.

    Returns:
        An SQLAlchemy Engine instance connected to the database.
    """
    engine = create_engine(db_uri)
    Base.metadata.create_all(engine)
    print(f"✅ Database and tables created successfully at {db_uri}")
    return engine


if __name__ == "__main__":
    engine = create_database()
