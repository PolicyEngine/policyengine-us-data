import pandas as pd
import numpy as np
from sqlmodel import Session, create_engine, select

from policyengine_us_data.storage import STORAGE_FOLDER

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.utils.census import get_census_docs, pull_acs_table
from policyengine_us_data.utils.db import parse_ucgid, get_geographic_strata


LABEL_TO_SHORT = {
    "Estimate!!Total!!Total population!!AGE!!Under 5 years": "0-4",
    "Estimate!!Total!!Total population!!AGE!!5 to 9 years": "5-9",
    "Estimate!!Total!!Total population!!AGE!!10 to 14 years": "10-14",
    "Estimate!!Total!!Total population!!AGE!!15 to 19 years": "15-19",
    "Estimate!!Total!!Total population!!AGE!!20 to 24 years": "20-24",
    "Estimate!!Total!!Total population!!AGE!!25 to 29 years": "25-29",
    "Estimate!!Total!!Total population!!AGE!!30 to 34 years": "30-34",
    "Estimate!!Total!!Total population!!AGE!!35 to 39 years": "35-39",
    "Estimate!!Total!!Total population!!AGE!!40 to 44 years": "40-44",
    "Estimate!!Total!!Total population!!AGE!!45 to 49 years": "45-49",
    "Estimate!!Total!!Total population!!AGE!!50 to 54 years": "50-54",
    "Estimate!!Total!!Total population!!AGE!!55 to 59 years": "55-59",
    "Estimate!!Total!!Total population!!AGE!!60 to 64 years": "60-64",
    "Estimate!!Total!!Total population!!AGE!!65 to 69 years": "65-69",
    "Estimate!!Total!!Total population!!AGE!!70 to 74 years": "70-74",
    "Estimate!!Total!!Total population!!AGE!!75 to 79 years": "75-79",
    "Estimate!!Total!!Total population!!AGE!!80 to 84 years": "80-84",
    "Estimate!!Total!!Total population!!AGE!!85 years and over": "85-999",
}
AGE_COLS = list(LABEL_TO_SHORT.values())


def transform_age_data(age_data, docs):
    df = age_data.copy()

    label_to_variable_mapping = dict(
        [
            (value["label"], key)
            for key, value in docs["variables"].items()
            if value["group"] == "S0101"
            and value["concept"] == "Age and Sex"
            and value["label"] in LABEL_TO_SHORT.keys()
        ]
    )

    # By transitivity, map the data set variable names to short names
    rename_mapping = dict(
        [
            (label_to_variable_mapping[v], LABEL_TO_SHORT[v])
            for v in LABEL_TO_SHORT.keys()
        ]
    )

    df = df.drop(columns="NAME")
    df = df.rename({"GEO_ID": "ucgid_str"}, axis=1)
    df_data = df.rename(columns=rename_mapping)[["ucgid_str"] + list(AGE_COLS)]

    # Filter out Puerto Rico's district and state records, if needed
    df_geos = df_data[
        ~df_data["ucgid_str"].isin(["5001800US7298", "0400000US72"])
    ].copy()

    df = df_geos[["ucgid_str"] + AGE_COLS]

    df_long = df.melt(
        id_vars="ucgid_str",
        value_vars=AGE_COLS,
        var_name="age_range",
        value_name="value",
    )
    age_bounds = df_long["age_range"].str.split("-", expand=True).astype(int)
    age_bounds.columns = ["ge", "le"]
    age_bounds[["gt"]] = age_bounds[["ge"]] - 1
    age_bounds[["lt"]] = age_bounds[["le"]] + 1

    df_long["age_greater_than"] = age_bounds[["gt"]]
    df_long["age_less_than"] = age_bounds[["lt"]]
    df_long["variable"] = "person_count"
    df_long["reform_id"] = 0
    df_long["source_id"] = 1
    df_long["active"] = True

    return df_long


def load_age_data(df_long, geo, year):

    # Quick data quality check before loading ----
    if geo == "National":
        assert len(set(df_long.ucgid_str)) == 1
    elif geo == "State":
        assert len(set(df_long.ucgid_str)) == 51
    elif geo == "District":
        assert len(set(df_long.ucgid_str)) == 436
    else:
        raise ValueError('geo must be one of "National", "State", "District"')

    # Prepare to load data -----------
    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)

    with Session(engine) as session:
        # Fetch existing geographic strata
        geo_strata = get_geographic_strata(session)
        
        for _, row in df_long.iterrows():
            # Parse the UCGID to determine geographic info
            geo_info = parse_ucgid(row["ucgid_str"])
            
            # Determine parent stratum based on geographic level
            if geo_info["type"] == "national":
                parent_stratum_id = geo_strata["national"]
            elif geo_info["type"] == "state":
                parent_stratum_id = geo_strata["state"][geo_info["state_fips"]]
            elif geo_info["type"] == "district":
                parent_stratum_id = geo_strata["district"][
                    geo_info["congressional_district_geoid"]
                ]
            else:
                raise ValueError(f"Unknown geography type: {geo_info['type']}")
            
            # Create the age stratum as a child of the geographic stratum
            # Build a proper geographic identifier for the notes
            if geo_info["type"] == "national":
                geo_desc = "US"
            elif geo_info["type"] == "state":
                geo_desc = f"State FIPS {geo_info['state_fips']}"
            elif geo_info["type"] == "district":
                geo_desc = f"CD {geo_info['congressional_district_geoid']}"
            else:
                geo_desc = "Unknown"
            
            note = f"Age: {row['age_range']}, {geo_desc}"
            
            # Check if this age stratum already exists
            existing_stratum = session.exec(
                select(Stratum).where(
                    Stratum.parent_stratum_id == parent_stratum_id,
                    Stratum.stratum_group_id == 0,
                    Stratum.notes == note
                )
            ).first()
            
            if existing_stratum:
                # Update the existing stratum's target instead of creating a duplicate
                existing_target = session.exec(
                    select(Target).where(
                        Target.stratum_id == existing_stratum.stratum_id,
                        Target.variable == row["variable"],
                        Target.period == year
                    )
                ).first()
                
                if existing_target:
                    # Update existing target
                    existing_target.value = row["value"]
                else:
                    # Add new target to existing stratum
                    new_target = Target(
                        stratum_id=existing_stratum.stratum_id,
                        variable=row["variable"],
                        period=year,
                        value=row["value"],
                        source_id=row["source_id"],
                        active=row["active"],
                    )
                    session.add(new_target)
                continue  # Skip creating a new stratum
            
            new_stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                stratum_group_id=0,  # Age strata group
                notes=note,
            )

            # Create constraints including both age and geographic for uniqueness
            new_stratum.constraints_rel = []
            
            # Add geographic constraints based on level
            if geo_info["type"] == "state":
                new_stratum.constraints_rel.append(
                    StratumConstraint(
                        constraint_variable="state_fips",
                        operation="==",
                        value=str(geo_info["state_fips"]),
                    )
                )
            elif geo_info["type"] == "district":
                new_stratum.constraints_rel.append(
                    StratumConstraint(
                        constraint_variable="congressional_district_geoid",
                        operation="==",
                        value=str(geo_info["congressional_district_geoid"]),
                    )
                )
            # For national level, no geographic constraint needed
            
            # Add age constraints
            new_stratum.constraints_rel.append(
                StratumConstraint(
                    constraint_variable="age",
                    operation=">",
                    value=str(row["age_greater_than"]),
                )
            )

            age_lt_value = row["age_less_than"]
            if not np.isinf(age_lt_value):
                new_stratum.constraints_rel.append(
                    StratumConstraint(
                        constraint_variable="age",
                        operation="<",
                        value=str(row["age_less_than"]),
                    )
                )

            # Create the Target and link it to the parent.
            new_stratum.targets_rel.append(
                Target(
                    variable=row["variable"],
                    period=year,
                    value=row["value"],
                    source_id=row["source_id"],
                    active=row["active"],
                )
            )

            # Add ONLY the parent object to the session.
            # The 'cascade' setting will handle the children automatically.
            session.add(new_stratum)

        # Commit all the new objects at once.
        session.commit()


if __name__ == "__main__":

    # --- ETL: Extract, Transform, Load ----
    year = 2023

    # ---- Extract ----------
    docs = get_census_docs(year)
    national_df = pull_acs_table("S0101", "National", year)
    state_df = pull_acs_table("S0101", "State", year)
    district_df = pull_acs_table("S0101", "District", year)

    # --- Transform ----------
    long_national_df = transform_age_data(national_df, docs)
    long_state_df = transform_age_data(state_df, docs)
    long_district_df = transform_age_data(district_df, docs)

    # --- Load --------
    # Note: The geographic strata must already exist in the database
    # (created by create_initial_strata.py)
    load_age_data(long_national_df, "National", year)
    load_age_data(long_state_df, "State", year)
    load_age_data(long_district_df, "District", year)