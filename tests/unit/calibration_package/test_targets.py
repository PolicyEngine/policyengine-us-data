import json

import pytest
from sqlalchemy import create_engine, text

from policyengine_us_data.calibration_package.specs import TargetConfigIdentity
from policyengine_us_data.calibration_package.targets import (
    TARGET_OVERVIEW_VIEW,
    TargetCatalogReader,
    TargetSelectionPolicy,
    target_facets_from_rows,
)


@pytest.fixture
def target_db(tmp_path):
    db_path = tmp_path / "targets.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE strata ("
                "stratum_id INTEGER PRIMARY KEY, "
                "definition_hash VARCHAR(64), "
                "parent_stratum_id INTEGER, "
                "notes VARCHAR)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE stratum_constraints ("
                "constraint_id INTEGER PRIMARY KEY, "
                "stratum_id INTEGER, "
                "constraint_variable TEXT, "
                "operation TEXT, "
                "value TEXT)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE targets ("
                "target_id INTEGER PRIMARY KEY, "
                "stratum_id INTEGER, "
                "variable TEXT, "
                "reform_id INTEGER DEFAULT 0, "
                "value REAL, "
                "period INTEGER, "
                "active INTEGER DEFAULT 1, "
                "source TEXT, "
                "notes TEXT)"
            )
        )
        conn.execute(text(TARGET_OVERVIEW_VIEW))
        conn.execute(text("INSERT INTO strata VALUES (1, NULL, NULL, 'national')"))
        conn.execute(text("INSERT INTO strata VALUES (2, NULL, 1, 'state snap')"))
        conn.execute(text("INSERT INTO strata VALUES (3, NULL, 1, 'national rent')"))
        conn.execute(
            text(
                "INSERT INTO stratum_constraints VALUES "
                "(1, 2, 'state_fips', '=', '6'), "
                "(2, 2, 'snap', '>', '0'), "
                "(3, 3, 'rent', '>', '0')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO targets "
                "(target_id, stratum_id, variable, reform_id, value, period, active, source, notes) "
                "VALUES "
                "(1, 1, 'snap', 0, 100.0, 2022, 1, 'SOI', 'base snap'), "
                "(2, 1, 'snap', 0, 200.0, 2024, 0, 'SOI', 'disabled newer snap'), "
                "(3, 2, 'eitc+ctc', 0, 300.0, 2024, 1, 'IRS', 'additive'), "
                "(4, 3, 'rent', 0, 400.0, 2024, 1, 'ACS', 'rent domain')"
            )
        )
        conn.commit()
    try:
        yield f"sqlite:///{db_path}"
    finally:
        engine.dispose()


def _identity() -> TargetConfigIdentity:
    return TargetConfigIdentity(
        path="policyengine_us_data/calibration/target_config.yaml",
        sha256="sha256:target-config",
        mode="default",
        resolved_path="/repo/policyengine_us_data/calibration/target_config.yaml",
    )


def test_target_catalog_reader_loads_active_and_disabled_targets(target_db):
    catalog = TargetCatalogReader(db_uri=target_db, time_period=2024).load()

    assert catalog.targets["target_id"].tolist() == [1, 3, 4]
    assert catalog.disabled_targets["target_id"].tolist() == [2]
    assert catalog.constraints_for(2)[0]["variable"] == "state_fips"


def test_additive_target_expressions_require_valid_components(target_db):
    catalog = TargetCatalogReader(db_uri=target_db, time_period=2024).load(
        {"variables": ["eitc+ctc"]}
    )
    policy = TargetSelectionPolicy.from_config({})

    selected = policy.select(catalog, valid_variables={"eitc", "ctc"})

    assert selected.targets_df["variable"].tolist() == ["eitc+ctc"]
    with pytest.raises(ValueError, match="ctc"):
        policy.select(catalog, valid_variables={"eitc"})


def test_target_selection_policy_filters_config_and_reports_disabled(target_db):
    catalog = TargetCatalogReader(db_uri=target_db, time_period=2024).load()
    policy = TargetSelectionPolicy.from_config(
        {
            "include": [
                {"variable": "snap", "geo_level": "national"},
                {"variable": "rent", "geo_level": "national"},
            ],
            "exclude": [{"variable": "rent", "geo_level": "national"}],
        }
    )

    selected = policy.select(catalog, target_config_identity=_identity())

    assert selected.target_ids == [1]
    assert selected.disabled_rows()[0]["target_id"] == 2
    assert selected.summary()["target_config_sha256"] == "sha256:target-config"


def test_target_selection_order_and_checksum_change_with_config(target_db):
    catalog = TargetCatalogReader(db_uri=target_db, time_period=2024).load()
    all_targets = TargetSelectionPolicy.from_config({}).select(catalog)
    snap_only = TargetSelectionPolicy.from_config(
        {"include": [{"variable": "snap", "geo_level": "national"}]}
    ).select(catalog)

    assert all_targets.target_ids == [1, 3, 4]
    assert snap_only.target_ids == [1]
    assert all_targets.checksum != snap_only.checksum


def test_target_metadata_artifacts_match_package_order_and_facets(target_db, tmp_path):
    catalog = TargetCatalogReader(db_uri=target_db, time_period=2024).load()
    selected = TargetSelectionPolicy.from_config({}).select(
        catalog,
        target_config_identity=_identity(),
    )
    matrix_order = selected.targets_df.iloc[[1, 0, 2]].reset_index(drop=True)
    selected = selected.with_matrix_order(
        matrix_order,
        ["state_6/eitc+ctc[snap>0]", "national/snap", "national/rent[rent>0]"],
    )

    targets_path, facets_path = selected.write_artifacts(
        tmp_path / "calibration_targets.jsonl",
        tmp_path / "calibration_target_facets.json",
    )
    rows = [
        json.loads(line)
        for line in targets_path.read_text(encoding="utf-8").splitlines()
    ]
    facets = json.loads(facets_path.read_text(encoding="utf-8"))

    assert [row["target_id"] for row in rows] == [3, 1, 4]
    assert [row["target_index"] for row in rows] == [0, 1, 2]
    assert rows[0]["target_expression"] == "eitc+ctc"
    assert rows[0]["target_components"] == ["eitc", "ctc"]
    assert facets == target_facets_from_rows(rows)

    fake_fit_rows = [{"target_id": 3, "target_index": 0, "fitted": 301.0}]
    rows_by_id = {row["target_id"]: row for row in rows}
    rows_by_index = {row["target_index"]: row for row in rows}
    assert (
        rows_by_id[fake_fit_rows[0]["target_id"]]["target_name"]
        == (rows_by_index[fake_fit_rows[0]["target_index"]]["target_name"])
    )
