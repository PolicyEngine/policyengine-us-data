import sqlite3

import pytest

from policyengine_us_data.build_outputs.target_universe import (
    RegionalTargetUniverse,
    TargetUniverseReader,
)


def _write_target_cd_db(db_path, cd_geoids: tuple[str, ...]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE stratum_constraints "
            "(constraint_variable TEXT NOT NULL, value TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO stratum_constraints VALUES (?, ?)",
            [("congressional_district_geoid", cd_geoid) for cd_geoid in cd_geoids],
        )
        conn.execute(
            "INSERT INTO stratum_constraints VALUES (?, ?)",
            ("other_constraint", "9999"),
        )


def test_target_universe_reader_loads_sorted_regional_cd_geoids(tmp_path):
    db_path = tmp_path / "policy_data.db"
    _write_target_cd_db(db_path, ("102", "101"))

    universe = TargetUniverseReader.from_sqlite(db_path).regional()

    assert universe == RegionalTargetUniverse(cd_geoids=("101", "102"))


def test_regional_target_universe_rejects_empty_cd_geoids():
    with pytest.raises(ValueError, match="must contain CD GEOIDs"):
        RegionalTargetUniverse(cd_geoids=())
