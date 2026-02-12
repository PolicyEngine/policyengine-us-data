import hashlib

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    create_database,
)


@pytest.fixture
def engine(tmp_path):
    db_uri = f"sqlite:///{tmp_path/'test.db'}"
    return create_database(db_uri)


# TODO: Re-enable this test once database issues are resolved in PR #437
@pytest.mark.skip(
    reason="Temporarily disabled - database functionality being fixed in PR #437"
)
def test_stratum_hash_and_relationships(engine):
    with Session(engine) as session:
        stratum = Stratum(notes="test")
        stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="ucgid_str",
                operation="==",
                value="0400000US30",
            ),
            StratumConstraint(
                constraint_variable="age",
                operation=">",
                value="20",
            ),
            StratumConstraint(
                constraint_variable="age",
                operation="<",
                value="65",
            ),
        ]
        stratum.targets_rel = [
            Target(variable="person_count", period=2023, value=100.0)
        ]
        session.add(stratum)
        session.commit()
        expected_hash = hashlib.sha256(
            "\n".join(
                sorted(
                    [
                        "ucgid_str|==|0400000US30",
                        "age|>|20",
                        "age|<|65",
                    ]
                )
            ).encode("utf-8")
        ).hexdigest()
        assert stratum.definition_hash == expected_hash
        retrieved = session.get(Stratum, stratum.stratum_id)
        assert len(retrieved.constraints_rel) == 3
        assert retrieved.targets_rel[0].value == 100.0


def test_unique_definition_hash(engine):
    with Session(engine) as session:
        s1 = Stratum()
        s1.constraints_rel = [
            StratumConstraint(
                constraint_variable="ucgid_str",
                operation="==",
                value="0400000US30",
            )
        ]
        session.add(s1)
        session.commit()
        s2 = Stratum()
        s2.constraints_rel = [
            StratumConstraint(
                constraint_variable="ucgid_str",
                operation="==",
                value="0400000US30",
            )
        ]
        session.add(s2)
        with pytest.raises(IntegrityError):
            session.commit()


def test_valid_parent_child_constraints(engine):
    with Session(engine) as session:
        parent = Stratum()
        parent.constraints_rel = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value="06",
            )
        ]
        session.add(parent)
        session.commit()
        session.refresh(parent)

        child = Stratum(parent_stratum_id=parent.stratum_id)
        child.constraints_rel = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value="06",
            ),
            StratumConstraint(
                constraint_variable="age",
                operation=">",
                value="24",
            ),
        ]
        session.add(child)
        session.commit()

        retrieved = session.get(Stratum, child.stratum_id)
        assert retrieved.parent_stratum_id == parent.stratum_id
        assert len(retrieved.constraints_rel) == 2


def test_invalid_parent_child_constraints(engine):
    with Session(engine) as session:
        parent = Stratum()
        parent.constraints_rel = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value="06",
            )
        ]
        session.add(parent)
        session.commit()
        session.refresh(parent)

        child = Stratum(parent_stratum_id=parent.stratum_id)
        child.constraints_rel = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value="30",
            ),
        ]
        session.add(child)
        with pytest.raises(ValueError, match="parent constraint"):
            session.commit()


def test_parent_with_no_constraints(engine):
    with Session(engine) as session:
        parent = Stratum()
        session.add(parent)
        session.commit()
        session.refresh(parent)

        child = Stratum(parent_stratum_id=parent.stratum_id)
        child.constraints_rel = [
            StratumConstraint(
                constraint_variable="age",
                operation=">",
                value="24",
            ),
        ]
        session.add(child)
        session.commit()

        retrieved = session.get(Stratum, child.stratum_id)
        assert retrieved.parent_stratum_id == parent.stratum_id
