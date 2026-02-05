import hashlib
from enum import Enum

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    create_database,
)
from policyengine_us_data.db.create_field_valid_values import (
    populate_field_valid_values,
)


@pytest.fixture
def engine(tmp_path):
    db_uri = f"sqlite:///{tmp_path/'test.db'}"
    eng = create_database(db_uri)
    # Populate field_valid_values for trigger validation
    with Session(eng) as session:
        populate_field_valid_values(session)
    return eng


# TODO: Re-enable this test once database issues are resolved in PR #437
@pytest.mark.skip(
    reason="Temporarily disabled - database functionality being fixed in PR #437"
)
def test_stratum_hash_and_relationships(engine):
    with Session(engine) as session:
        stratum = Stratum(notes="test", stratum_group_id=0)
        stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value="30",
            ),
            StratumConstraint(
                constraint_variable="age", operation=">", value="20"
            ),
            StratumConstraint(
                constraint_variable="age", operation="<", value="65"
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
                        "state_fips|==|30",
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
        s1 = Stratum(stratum_group_id=0)
        s1.constraints_rel = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value="30",
            )
        ]
        session.add(s1)
        session.commit()
        s2 = Stratum(stratum_group_id=0)
        s2.constraints_rel = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value="30",
            )
        ]
        session.add(s2)
        with pytest.raises(IntegrityError):
            session.commit()
