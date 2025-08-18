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
from policyengine_us_data.db import create_initial_strata


@pytest.fixture
def engine(tmp_path):
    db_uri = f"sqlite:///{tmp_path/'test.db'}"
    return create_database(db_uri)


def test_stratum_hash_and_relationships(engine):
    with Session(engine) as session:
        stratum = Stratum(notes="test", stratum_group_id=0)
        stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="ucgid_str", operation="in", value="0001"
            ),
            StratumConstraint(
                constraint_variable="age", operation="greater_than", value="20"
            ),
            StratumConstraint(
                constraint_variable="age", operation="less_than", value="65"
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
                        "ucgid_str|in|0001",
                        "age|greater_than|20",
                        "age|less_than|65",
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
                constraint_variable="ucgid_str", operation="in", value="0001"
            )
        ]
        session.add(s1)
        session.commit()
        s2 = Stratum(stratum_group_id=0)
        s2.constraints_rel = [
            StratumConstraint(
                constraint_variable="ucgid_str", operation="in", value="0001"
            )
        ]
        session.add(s2)
        with pytest.raises(IntegrityError):
            session.commit()


def test_create_initial_strata(monkeypatch, engine, tmp_path):
    # ``monkeypatch`` is a pytest fixture that lets us temporarily modify or replace
    # objects during a test. Here we use it to point ``STORAGE_FOLDER`` to a
    # temporary directory so the test doesn't touch real data on disk.
    monkeypatch.setattr(create_initial_strata, "STORAGE_FOLDER", tmp_path)

    class FakeEnum(Enum):
        NAT = "NAT"
        STATE = "STATE"
        DIST = "DIST"

        def get_hierarchical_codes(self):
            mapping = {
                FakeEnum.NAT: ["NAT"],
                FakeEnum.STATE: ["STATE", "NAT"],
                FakeEnum.DIST: ["DIST", "STATE", "NAT"],
            }
            return mapping[self]

    # Replace the real ``UCGID`` enumeration with our simplified version so the
    # test can run without downloading geographic data.
    monkeypatch.setattr(create_initial_strata, "UCGID", FakeEnum)
    create_initial_strata.main()
    with Session(engine) as session:
        strata = session.exec(select(Stratum).order_by(Stratum.stratum_id)).all()
        assert len(strata) == 3
        nat, state, dist = strata
        assert state.parent_stratum_id == nat.stratum_id
        assert dist.parent_stratum_id == state.stratum_id
        codes = [s.constraints_rel[0].value for s in strata]
        assert codes == ["NAT", "STATE", "DIST"]
