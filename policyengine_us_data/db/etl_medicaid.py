import requests

import pandas as pd
from sqlmodel import Session, create_engine

from policyengine_us_data.storage import STORAGE_FOLDER

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.utils.census import STATE_ABBREV_TO_FIPS


def extract_medicaid_data(year):
    base_url = (
        f"https://api.census.gov/data/{year}/acs/acs1/subject?get=group(S2704)"
    )
    url = f"{base_url}&for=congressional+district:*"
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    headers = data[0]
    data_rows = data[1:]
    cd_survey_df = pd.DataFrame(data_rows, columns=headers)

    item = "6165f45b-ca93-5bb5-9d06-db29c692a360"
    response = requests.get(
        f"https://data.medicaid.gov/api/1/metastore/schemas/dataset/items/{item}?show-reference-ids=false"
    )
    metadata = response.json()

    data_url = metadata["distribution"][0]["data"]["downloadURL"]
    state_admin_df = pd.read_csv(data_url)

    return cd_survey_df, state_admin_df


def transform_medicaid_data(state_admin_df, cd_survey_df, year):

    reporting_period = year * 100 + 12
    print(f"Reporting period is {reporting_period}")
    state_df = state_admin_df.loc[
        (state_admin_df["Reporting Period"] == reporting_period)
        & (state_admin_df["Final Report"] == "Y"),
        [
            "State Abbreviation",
            "Reporting Period",
            "Total Medicaid Enrollment",
        ],
    ]

    state_df["FIPS"] = state_df["State Abbreviation"].map(STATE_ABBREV_TO_FIPS)

    cd_df = cd_survey_df[
        ["GEO_ID", "state", "congressional district", "S2704_C02_006E"]
    ]

    nc_cd_sum = cd_df.loc[cd_df.state == "37"].S2704_C02_006E.astype(int).sum()
    nc_state_sum = state_df.loc[state_df.FIPS == "37"][
        "Total Medicaid Enrollment"
    ].values[0]
    assert nc_cd_sum > 0.5 * nc_state_sum
    assert nc_cd_sum <= nc_state_sum

    state_df = state_df.rename(
        columns={"Total Medicaid Enrollment": "medicaid_enrollment"}
    )
    state_df["ucgid_str"] = "0400000US" + state_df["FIPS"].astype(str)

    cd_df = cd_df.rename(
        columns={
            "S2704_C02_006E": "medicaid_enrollment",
            "GEO_ID": "ucgid_str",
        }
    )
    cd_df = cd_df.loc[cd_df.state != "72"]

    out_cols = ["ucgid_str", "medicaid_enrollment"]
    return state_df[out_cols], cd_df[out_cols]


def load_medicaid_data(long_state, long_cd, year):

    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)

    stratum_lookup = {}

    with Session(engine) as session:
        # National ----------------
        nat_stratum = Stratum(
            parent_stratum_id=None,
            stratum_group_id=0,
            notes="Geo: 0100000US Medicaid Enrolled",
        )
        nat_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="ucgid_str",
                operation="in",
                value="0100000US",
            ),
            StratumConstraint(
                constraint_variable="medicaid_enrolled",
                operation="equals",
                value="True",
            ),
        ]
        # No target at the national level is provided at this time.

        session.add(nat_stratum)
        session.flush()
        stratum_lookup["National"] = nat_stratum.stratum_id

        # State -------------------
        stratum_lookup["State"] = {}
        for _, row in long_state.iterrows():

            note = f"Geo: {row['ucgid_str']} Medicaid Enrolled"
            parent_stratum_id = nat_stratum.stratum_id

            new_stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                stratum_group_id=0,
                notes=note,
            )
            new_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="ucgid_str",
                    operation="in",
                    value=row["ucgid_str"],
                ),
                StratumConstraint(
                    constraint_variable="medicaid_enrolled",
                    operation="equals",
                    value="True",
                ),
            ]
            new_stratum.targets_rel.append(
                Target(
                    variable="person_count",
                    period=year,
                    value=row["medicaid_enrollment"],
                    source_id=2,
                    active=True,
                )
            )
            session.add(new_stratum)
            session.flush()
            stratum_lookup["State"][row["ucgid_str"]] = new_stratum.stratum_id

        # District -------------------
        for _, row in long_cd.iterrows():

            note = f"Geo: {row['ucgid_str']} Medicaid Enrolled"
            parent_stratum_id = stratum_lookup["State"][
                f'0400000US{row["ucgid_str"][-4:-2]}'
            ]

            new_stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                stratum_group_id=0,
                notes=note,
            )
            new_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="ucgid_str",
                    operation="in",
                    value=row["ucgid_str"],
                ),
                StratumConstraint(
                    constraint_variable="medicaid_enrolled",
                    operation="equals",
                    value="True",
                ),
            ]
            new_stratum.targets_rel.append(
                Target(
                    variable="person_count",
                    period=year,
                    value=row["medicaid_enrollment"],
                    source_id=2,
                    active=True,
                )
            )
            session.add(new_stratum)
            session.flush()

        session.commit()


if __name__ == "__main__":

    year = 2023

    # Extract ------------------------------
    cd_survey_df, state_admin_df = extract_medicaid_data(year)

    # Transform -------------------
    long_state, long_cd = transform_medicaid_data(
        state_admin_df, cd_survey_df, year
    )

    # Load -----------------------
    load_medicaid_data(long_state, long_cd, year)
