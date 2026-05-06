from sqlmodel import Session, select

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    create_database,
)
from policyengine_us_data.db.etl_aca_agi_state_targets import _load_agi_state_targets


def test_state_agi_targets_use_filer_strata(tmp_path, monkeypatch):
    calibration_targets = tmp_path / "calibration_targets"
    calibration_targets.mkdir()
    (calibration_targets / "agi_state.csv").write_text(
        "\n".join(
            [
                "GEO_ID,GEO_NAME,AGI_LOWER_BOUND,AGI_UPPER_BOUND,VALUE,IS_COUNT,VARIABLE",
                "0400000US06,CA,-inf,1.0,123,1,adjusted_gross_income/count",
                "0400000US06,CA,-inf,1.0,-456,0,adjusted_gross_income/amount",
            ]
        )
    )
    monkeypatch.setattr(
        "policyengine_us_data.db.etl_aca_agi_state_targets.STORAGE_FOLDER",
        tmp_path,
    )

    engine = create_database(f"sqlite:///{tmp_path / 'policy_data.db'}")
    with Session(engine) as session:
        state_stratum = Stratum(notes="California")
        state_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value="6",
            )
        ]
        session.add(state_stratum)
        session.commit()
        session.refresh(state_stratum)

        _load_agi_state_targets(
            session,
            2024,
            {"state": {6: state_stratum.stratum_id}},
        )
        session.commit()

        targets = session.exec(select(Target)).all()
        assert {target.variable for target in targets} == {
            "tax_unit_count",
            "adjusted_gross_income",
        }

        strata = {
            target.variable: session.get(Stratum, target.stratum_id)
            for target in targets
        }
        for stratum in strata.values():
            constraints = {
                (
                    constraint.constraint_variable,
                    constraint.operation,
                    constraint.value,
                )
                for constraint in stratum.constraints_rel
            }
            assert ("tax_unit_is_filer", "==", "1") in constraints
            assert ("state_fips", "==", "6") in constraints
            assert ("adjusted_gross_income", ">=", "-inf") in constraints
            assert ("adjusted_gross_income", "<", "1.0") in constraints
            assert ("adjusted_gross_income", ">", "0") not in constraints
