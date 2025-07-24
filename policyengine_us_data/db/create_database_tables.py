import io
import logging

import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    ForeignKey,
    UniqueConstraint,
    PrimaryKeyConstraint,
)

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase, relationship


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# Create a single, shared Base that all models can inherit from
class Base(DeclarativeBase):
    pass


class Strata(Base):
    """
    Represents a stratum (the plural of which is "strata"), which is a
    specific subgroup of a population.

    This table acts as the parent for stratum-related data, defining each unique
    stratum with an ID.
    """

    __tablename__ = "strata"

    # Columns --- 
    stratum_id = Column(
        Integer, primary_key=True, comment="Unique identifier for the stratum."
    )

    parent_stratum_id = Column(
        Integer,
        ForeignKey("strata.stratum_id"),
        index=True,
        comment="Identifier for a parent stratum, creating a hierarchy.",
    )

    stratum_group_id = Column(
        Integer, comment="Identifier for a group of related strata."
    )

    notes = Column(String, comment="Descriptive notes about the stratum.")

    # Relationships --- 
    children_rel = relationship("Strata", 
                               back_populates="parent_rel",
                               remote_side=[parent_stratum_id])

    parent_rel = relationship("Strata", 
                             back_populates="children_rel",
                             remote_side=[stratum_id])

    constraints_rel = relationship(
        "StratumConstraints",
        back_populates="strata_rel",
        cascade="all, delete-orphan",
    )

    targets_rel = relationship(
        "Targets", back_populates="strata_rel", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Strata(stratum_id={self.stratum_id}, notes='{self.notes}')>"


class StratumConstraints(Base):
    """
    Defines the specific rules or conditions that make up a stratum.

    For example, a stratum for 'high-income individuals in New York' would have
    constraints for both income level and state.
    """

    __tablename__ = "stratum_constraints"

    # Columns ---- 
    stratum_id = Column(
        Integer, ForeignKey("strata.stratum_id"), primary_key=True
    )

    constraint_variable = Column(
        String,
        primary_key=True,
        comment="The variable the constraint applies to (e.g., 'age').",
    )

    operation = Column(
        String,
        primary_key=True,
        comment="The comparison operator (e.g., 'greater_than_or_equal').",
    )

    value = Column(
        String,
        nullable=False,
        comment="The value for the constraint rule (e.g., '25').",
    )

    notes = Column(
        String, nullable=True, comment="Optional notes about the constraint."
    )

    # Relationships ----- 
    strata_rel = relationship("Strata", back_populates="constraints_rel")

    def __repr__(self):
        return f"<StratumConstraints(stratum_id={self.stratum_id}, variable='{self.constraint_variable}', operation='{self.operation}')>"


class Targets(Base):
    """
    Stores the actual data values for different variables within a specific
    stratum, period, and reform scenario.
    """

    __tablename__ = "targets"
    __table_args__ = (
        UniqueConstraint('variable', 'period', 'stratum_id', 'reform_id',
                         name='_target_unique'),
    )

    # Columns ----------- 
    target_id = Column(Integer, primary_key=True)  # Auto-incrementing

    variable = Column(
        String,
        nullable=False,
        comment="A variable defined in policyengine-us (e.g., 'income_tax').",
    )

    period = Column(
        Integer,
        nullable=False,
        comment="The time period for the data, typically a year.",
    )

    # Foreign key to ensure that stratum_id exists in the 'strata' table.
    stratum_id = Column(
        Integer, ForeignKey("strata.stratum_id"), nullable=False, index=True
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

    tolerance = Column(
        Float,
        nullable=True,
        comment="Allowed relative error as a percent (e.g., 25 for 25%)."
    )

    notes = Column(String, nullable=True,
                   comment="Optional descriptive notes about the target row.")

    # Relationships ------------ 
    strata_rel = relationship("Strata", back_populates="targets_rel")

    def __repr__(self):
        return f"<Targets(target_id={self.target_id}, variable='{self.variable}', value={self.value})>"


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
    logger.info(f"Database and tables created successfully at {db_uri}")
    return engine


if __name__ == "__main__":
    engine = create_database()
