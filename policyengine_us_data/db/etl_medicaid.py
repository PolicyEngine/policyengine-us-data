import requests
import pandas as pd


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)


# State abbreviation to FIPS code mapping
state_fips_map = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
    'CO': '08', 'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13',
    'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19',
    'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24',
    'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29',
    'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
    'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
    'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45',
    'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50',
    'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55', 'WY': '56',
    'DC': '11'
}


#"S2704_C02_006E": {
#  "label": "Estimate!!Public Coverage!!COVERAGE ALONE OR IN COMBINATION!!Medicaid/means-tested public coverage alone or in combination",
#  "concept": "Public Health Insurance Coverage by Type and Selected Characteristics",
#  "predicateType": "int",
#  "group": "S2704",
#  "limit": 0,
#  "attributes": "S2704_C02_006EA,S2704_C02_006M,S2704_C02_006MA"
#},


def extract_medicaid_data():
    year = 2023
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
    
    data_url = metadata['distribution'][0]['data']['downloadURL']
    state_admin_df = pd.read_csv(data_url)
    
    return cd_survey_df, state_admin_df


def transform_medicaid_data(state_admin_df, cd_survey_df):
    state_df = state_admin_df.loc[
        (state_admin_df["Reporting Period"] == 202312) &
        (state_admin_df["Final Report"] == "Y"),
        ["State Abbreviation", "Reporting Period", "Total Medicaid Enrollment"]
    ]

    state_df["FIPS"] = state_df["State Abbreviation"].map(state_fips_map)

    cd_df = cd_survey_df[["GEO_ID", "state", "congressional district", "S2704_C02_006E"]]

    nc_cd_sum = cd_df.loc[cd_df.state == "37"].S2704_C02_006E.astype(int).sum()
    nc_state_sum = state_df.loc[state_df.FIPS == '37']['Total Medicaid Enrollment'].values[0]
    assert nc_cd_sum > .5 * nc_state_sum
    assert nc_cd_sum <= nc_state_sum

    state_df = state_df.rename(columns={'Total Medicaid Enrollment': 'medicaid_enrollment'})
    state_df['ucgid_str'] = '0400000US' + state_df['FIPS'].astype(str)

    cd_df = cd_df.rename(columns={'S2704_C02_006E': 'medicaid_enrollment', 'GEO_ID': 'ucgid_str'})
    cd_df = cd_df.loc[cd_df.state != '72']

    out_cols = ['ucgid_str', 'medicaid_enrollment']
    return state_df[out_cols], cd_df[out_cols]


def load_medicaid_data(long_state, long_cd):

    DATABASE_URL = "sqlite:///policy_data.db"
    engine = create_engine(DATABASE_URL)

    Session = sessionmaker(bind=engine)
    session = Session()

    stratum_lookup = {}

    # Wow, the first time we're making these geos with no breakdown variable

    # National ----------------
    nat_stratum = Stratum(
        parent_stratum_id=None, stratum_group_id=0, notes="Geo: 0100000US"
    )
    nat_stratum.constraints_rel = [
        StratumConstraint(
            constraint_variable="ucgid_str",
            operation="in",
            value="0100000US",
        )
    ]

    session.add(nat_stratum)
    session.flush()
    stratum_lookup["National"] = nat_stratum.stratum_id

    # State -------------------
    stratum_lookup["State"] = {} 
    for _, row in long_state.iterrows():

        note = f"Geo: {row['ucgid_str']}"
        parent_stratum_id = nat_stratum.stratum_id

        new_stratum = Stratum(
            parent_stratum_id=parent_stratum_id, stratum_group_id=0, notes=note
        )
        new_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="ucgid_str",
                operation="in",
                value=row["ucgid_str"],
            ),
        ]
        new_stratum.targets_rel.append(
            Target(
                variable="medicaid_enrollment",
                period=2023,
                value=row["medicaid_enrollment"],
                source_id=2,
                active=True,
            )
        )
        session.add(new_stratum)
        session.flush()
        stratum_lookup["State"][row['ucgid_str']] = new_stratum.stratum_id


    # District -------------------
    stratum_lookup["District"] = {} 
    for _, row in long_cd.iterrows():

        note = f"Geo: {row['ucgid_str']}"
        parent_stratum_id = stratum_lookup["State"][f'0400000US{row["ucgid_str"][-4:-2]}']

        new_stratum = Stratum(
            parent_stratum_id=parent_stratum_id, stratum_group_id=0, notes=note
        )
        new_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="ucgid_str",
                operation="in",
                value=row["ucgid_str"],
            ),
        ]
        new_stratum.targets_rel.append(
            Target(
                variable="medicaid_enrollment",
                period=2023,
                value=row["medicaid_enrollment"],
                source_id=2,
                active=True,
            )
        )
        session.add(new_stratum)
        session.flush()


    session.commit()

    return stratum_lookup

if __name__ == "__main__":
    cd_survey_df, state_admin_df = extract_medicaid_data()

    long_state, long_cd = transform_medicaid_data(state_admin_df, cd_survey_df)

    load_medicaid_data(long_state, long_cd)
