import logging
import hashlib
from typing import List, Optional

from sqlalchemy import event, text, UniqueConstraint
from sqlalchemy.orm.attributes import get_history
from sqlmodel import (
    Field,
    Relationship,
    SQLModel,
    create_engine,
)

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_field_valid_values import (
    populate_field_valid_values,
    FieldValidValues,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


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
    active: bool = Field(
        default=True,
        description="Flag to indicate if the record is currently active.",
    )
    tolerance: Optional[float] = Field(
        default=None,
        description="Allowed relative error as a percent (e.g., 25 for 25%).",
    )
    source: Optional[str] = Field(
        default=None,
        description="Data source identifier.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional descriptive notes about the target row.",
    )

    strata_rel: Stratum = Relationship(back_populates="targets_rel")


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


@event.listens_for(Stratum, "before_insert")
@event.listens_for(Stratum, "before_update")
def validate_stratum_constraints(mapper, connection, target: Stratum):
    """Validate constraint consistency before saving a Stratum."""
    if not target.constraints_rel:
        return
    from policyengine_us_data.utils.constraint_validation import (
        Constraint,
        ensure_consistent_constraint_set,
    )

    constraints = [
        Constraint(
            variable=c.constraint_variable,
            operation=c.operation,
            value=c.value,
        )
        for c in target.constraints_rel
    ]
    ensure_consistent_constraint_set(constraints)


GEOGRAPHIC_CONSTRAINT_VARIABLES = {
    "state_fips",
    "congressional_district_geoid",
}


def _validate_geographic_consistency(parent_rows, child_constraints):
    """Validate that child geography is contained within parent geography.

    Args:
        parent_rows: List of (constraint_variable, operation, value) tuples.
        child_constraints: List of StratumConstraint objects.

    Raises:
        ValueError: If the child's geography is inconsistent with
            the parent's.
    """
    parent_dict = {var: val for var, _, val in parent_rows}
    child_dict = {c.constraint_variable: c.value for c in child_constraints}

    # Shared geographic variables must have identical values.
    # Compare as integers to handle zero-padding differences
    # (e.g., state_fips "1" vs "01").
    for var in GEOGRAPHIC_CONSTRAINT_VARIABLES:
        if var in parent_dict and var in child_dict:
            if int(parent_dict[var]) != int(child_dict[var]):
                raise ValueError(
                    f"Geographic inconsistency: child has "
                    f"{var} = {child_dict[var]} but parent "
                    f"has {var} = {parent_dict[var]}"
                )

    # CD must belong to the parent state.
    if (
        "state_fips" in parent_dict
        and "congressional_district_geoid" in child_dict
    ):
        parent_state = int(parent_dict["state_fips"])
        child_cd = int(child_dict["congressional_district_geoid"])
        cd_state = child_cd // 100
        if cd_state != parent_state:
            raise ValueError(
                f"Geographic inconsistency: CD {child_cd} belongs "
                f"to state {cd_state}, not parent state "
                f"{parent_state}"
            )


@event.listens_for(Stratum, "before_insert")
@event.listens_for(Stratum, "before_update")
def validate_parent_child_constraints(mapper, connection, target: Stratum):
    """Ensure child strata include all parent constraints."""
    if target.parent_stratum_id is None:
        return

    parent_rows = connection.execute(
        text(
            "SELECT constraint_variable, operation, value "
            "FROM stratum_constraints "
            "WHERE stratum_id = :pid"
        ),
        {"pid": target.parent_stratum_id},
    ).fetchall()

    if not parent_rows:
        return

    parent_vars = {var for var, _, _ in parent_rows}
    child_vars = {c.constraint_variable for c in target.constraints_rel}

    # Geographic hierarchy: validate containment instead of
    # requiring literal constraint inheritance.
    if (
        parent_vars <= GEOGRAPHIC_CONSTRAINT_VARIABLES
        and child_vars <= GEOGRAPHIC_CONSTRAINT_VARIABLES
    ):
        _validate_geographic_consistency(parent_rows, target.constraints_rel)
        return

    child_set = {
        (c.constraint_variable, c.operation, c.value)
        for c in target.constraints_rel
    }

    for var, op, val in parent_rows:
        if (var, op, val) in child_set:
            continue
        # Geographic values may differ only by zero-padding
        # (e.g., "1" vs "01"); compare as integers.
        if var in GEOGRAPHIC_CONSTRAINT_VARIABLES:
            child_vals = {
                c.value
                for c in target.constraints_rel
                if c.constraint_variable == var and c.operation == op
            }
            if any(int(cv) == int(val) for cv in child_vals):
                continue
        raise ValueError(
            f"Child stratum must include parent constraint "
            f"({var} {op} {val})"
        )


STRATUM_DOMAIN_VIEW = """\
CREATE VIEW IF NOT EXISTS stratum_domain AS
SELECT DISTINCT
    sc.stratum_id,
    sc.constraint_variable AS domain_variable
FROM stratum_constraints sc
WHERE sc.constraint_variable NOT IN (
    'state_fips', 'congressional_district_geoid',
    'tax_unit_is_filer', 'ucgid_str'
);
"""

TARGET_OVERVIEW_VIEW = """\
CREATE VIEW IF NOT EXISTS target_overview AS
SELECT
    t.target_id,
    t.stratum_id,
    t.variable,
    t.value,
    t.period,
    t.active,
    CASE
        WHEN MAX(CASE
            WHEN sc.constraint_variable = 'congressional_district_geoid'
                THEN 1
            WHEN sc.constraint_variable = 'ucgid_str'
                AND length(sc.value) = 13 THEN 1
            ELSE 0 END) = 1 THEN 'district'
        WHEN MAX(CASE
            WHEN sc.constraint_variable = 'state_fips' THEN 1
            WHEN sc.constraint_variable = 'ucgid_str'
                AND length(sc.value) = 11 THEN 1
            ELSE 0 END) = 1 THEN 'state'
        ELSE 'national'
    END AS geo_level,
    COALESCE(
        MAX(CASE
            WHEN sc.constraint_variable
                = 'congressional_district_geoid'
            THEN sc.value END),
        MAX(CASE
            WHEN sc.constraint_variable = 'state_fips'
            THEN sc.value END),
        MAX(CASE
            WHEN sc.constraint_variable = 'ucgid_str'
            THEN sc.value END),
        'US'
    ) AS geographic_id,
    GROUP_CONCAT(DISTINCT CASE
        WHEN sc.constraint_variable NOT IN (
            'state_fips', 'congressional_district_geoid',
            'tax_unit_is_filer', 'ucgid_str'
        ) THEN sc.constraint_variable
    END) AS domain_variable
FROM targets t
LEFT JOIN stratum_constraints sc ON t.stratum_id = sc.stratum_id
GROUP BY t.target_id, t.stratum_id, t.variable,
         t.value, t.period, t.active;
"""


def create_validation_triggers(engine) -> None:
    """Create SQL triggers that validate fields against field_valid_values.

    Args:
        engine: SQLAlchemy Engine instance.
    """
    triggers = [
        # --- stratum_constraints triggers ---
        """\
CREATE TRIGGER IF NOT EXISTS validate_stratum_constraints_insert
BEFORE INSERT ON stratum_constraints
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM field_valid_values
              WHERE field_name = 'operation'
              AND valid_value = NEW.operation) = 0
        THEN RAISE(ABORT,
             'Invalid operation value for stratum_constraints')
    END;
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM field_valid_values
              WHERE field_name = 'constraint_variable'
              AND valid_value = NEW.constraint_variable) = 0
        THEN RAISE(ABORT,
             'Invalid constraint_variable for stratum_constraints')
    END;
END;""",
        """\
CREATE TRIGGER IF NOT EXISTS validate_stratum_constraints_update
BEFORE UPDATE ON stratum_constraints
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM field_valid_values
              WHERE field_name = 'operation'
              AND valid_value = NEW.operation) = 0
        THEN RAISE(ABORT,
             'Invalid operation value for stratum_constraints')
    END;
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM field_valid_values
              WHERE field_name = 'constraint_variable'
              AND valid_value = NEW.constraint_variable) = 0
        THEN RAISE(ABORT,
             'Invalid constraint_variable for stratum_constraints')
    END;
END;""",
        # --- targets triggers ---
        """\
CREATE TRIGGER IF NOT EXISTS validate_targets_insert
BEFORE INSERT ON targets
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM field_valid_values
              WHERE field_name = 'active'
              AND valid_value = CAST(NEW.active AS TEXT)) = 0
        THEN RAISE(ABORT,
             'Invalid active value for targets')
    END;
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM field_valid_values
              WHERE field_name = 'period'
              AND valid_value = CAST(NEW.period AS TEXT)) = 0
        THEN RAISE(ABORT,
             'Invalid period value for targets')
    END;
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM field_valid_values
              WHERE field_name = 'variable'
              AND valid_value = NEW.variable) = 0
        THEN RAISE(ABORT,
             'Invalid variable value for targets')
    END;
    SELECT CASE
        WHEN NEW.source IS NOT NULL
        AND (SELECT COUNT(*) FROM field_valid_values
              WHERE field_name = 'source'
              AND valid_value = NEW.source) = 0
        THEN RAISE(ABORT,
             'Invalid source value for targets')
    END;
END;""",
        """\
CREATE TRIGGER IF NOT EXISTS validate_targets_update
BEFORE UPDATE ON targets
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM field_valid_values
              WHERE field_name = 'active'
              AND valid_value = CAST(NEW.active AS TEXT)) = 0
        THEN RAISE(ABORT,
             'Invalid active value for targets')
    END;
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM field_valid_values
              WHERE field_name = 'period'
              AND valid_value = CAST(NEW.period AS TEXT)) = 0
        THEN RAISE(ABORT,
             'Invalid period value for targets')
    END;
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM field_valid_values
              WHERE field_name = 'variable'
              AND valid_value = NEW.variable) = 0
        THEN RAISE(ABORT,
             'Invalid variable value for targets')
    END;
    SELECT CASE
        WHEN NEW.source IS NOT NULL
        AND (SELECT COUNT(*) FROM field_valid_values
              WHERE field_name = 'source'
              AND valid_value = NEW.source) = 0
        THEN RAISE(ABORT,
             'Invalid source value for targets')
    END;
END;""",
    ]

    with engine.connect() as conn:
        for trigger_sql in triggers:
            conn.execute(text(trigger_sql))
        conn.commit()

    logger.info("Validation triggers created successfully")


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

    # Populate field_valid_values (must come before triggers)
    from sqlmodel import Session

    with Session(engine) as session:
        populate_field_valid_values(session)

    # Create validation triggers
    create_validation_triggers(engine)

    # Create SQL views
    with engine.connect() as conn:
        conn.execute(text(STRATUM_DOMAIN_VIEW))
        conn.execute(text(TARGET_OVERVIEW_VIEW))
        conn.commit()

    logger.info(f"Database and tables created successfully at {db_uri}")
    return engine


if __name__ == "__main__":
    engine = create_database()
