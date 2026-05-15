"""Target-universe contracts for local H5 publication."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from policyengine_us_data.pipeline_metadata import pipeline_node

__all__ = [
    "RegionalTargetUniverse",
    "TargetUniverseReader",
]


@pipeline_node(
    id="local_h5_regional_target_universe",
    label="RegionalTargetUniverse",
    node_type="library",
    description="Target congressional district universe used to enumerate regional local H5 outputs.",
    source_file="policyengine_us_data/build_outputs/target_universe.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_target_universe.py"
    ],
)
@dataclass(frozen=True)
class RegionalTargetUniverse:
    """Congressional district target universe for regional H5 outputs."""

    cd_geoids: tuple[str, ...]

    def __post_init__(self) -> None:
        cd_geoids = tuple(str(item) for item in self.cd_geoids)
        if not cd_geoids:
            raise ValueError("Regional target universe must contain CD GEOIDs")
        object.__setattr__(self, "cd_geoids", cd_geoids)


@pipeline_node(
    id="local_h5_target_universe_reader",
    label="TargetUniverseReader",
    node_type="library",
    description="Read local H5 target-universe contracts from the staged target database.",
    source_file="policyengine_us_data/build_outputs/target_universe.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_target_universe.py"
    ],
)
@dataclass(frozen=True)
class TargetUniverseReader:
    """Adapter from the Stage 1 target database artifact to H5 target contracts."""

    db_path: Path

    @classmethod
    def from_sqlite(cls, db_path: Path | str) -> "TargetUniverseReader":
        """Create a reader for a SQLite `policy_data.db` artifact."""

        return cls(db_path=Path(db_path))

    def regional(self) -> RegionalTargetUniverse:
        """Read the regional congressional district target universe."""

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT value AS cd_geoid
                FROM stratum_constraints
                WHERE constraint_variable = 'congressional_district_geoid'
                ORDER BY value
                """
            ).fetchall()
        return RegionalTargetUniverse(cd_geoids=tuple(str(row[0]) for row in rows))
