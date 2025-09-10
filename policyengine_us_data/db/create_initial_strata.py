from typing import Dict
import requests

import pandas as pd
import pandas as pd
from sqlmodel import Session, create_engine

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
)


def fetch_congressional_districts(year):

    # Fetch from Census API
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME",
        "for": "congressional district:*",
        "in": "state:*"
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data[1:], columns=data[0])
    df['state_fips'] = df['state'].astype(int)
    df = df[df['state_fips'] <= 56].copy()
    df['district_number'] = df['congressional district'].apply(
        lambda x: 0 if x in ['ZZ', '98'] else int(x)
    )
    
    # Filter out statewide summary records for multi-district states
    df['n_districts'] = df.groupby('state_fips')['state_fips'].transform('count')
    df = df[(df['n_districts'] == 1) | (df['district_number'] > 0)].copy()
    df = df.drop(columns=['n_districts'])
    
    df.loc[df['district_number'] == 0, 'district_number'] = 1
    df['congressional_district_geoid'] = df['state_fips'] * 100 + df['district_number']

    df = df[['state_fips', 'district_number', 'congressional_district_geoid', 'NAME']]
    df = df.sort_values('congressional_district_geoid')
    
    return df


def main():
    # State FIPS to name/abbreviation mapping
    STATE_NAMES = {
        1: "Alabama (AL)", 2: "Alaska (AK)", 4: "Arizona (AZ)", 5: "Arkansas (AR)",
        6: "California (CA)", 8: "Colorado (CO)", 9: "Connecticut (CT)", 10: "Delaware (DE)",
        11: "District of Columbia (DC)", 12: "Florida (FL)", 13: "Georgia (GA)", 15: "Hawaii (HI)",
        16: "Idaho (ID)", 17: "Illinois (IL)", 18: "Indiana (IN)", 19: "Iowa (IA)",
        20: "Kansas (KS)", 21: "Kentucky (KY)", 22: "Louisiana (LA)", 23: "Maine (ME)",
        24: "Maryland (MD)", 25: "Massachusetts (MA)", 26: "Michigan (MI)", 27: "Minnesota (MN)",
        28: "Mississippi (MS)", 29: "Missouri (MO)", 30: "Montana (MT)", 31: "Nebraska (NE)",
        32: "Nevada (NV)", 33: "New Hampshire (NH)", 34: "New Jersey (NJ)", 35: "New Mexico (NM)",
        36: "New York (NY)", 37: "North Carolina (NC)", 38: "North Dakota (ND)", 39: "Ohio (OH)",
        40: "Oklahoma (OK)", 41: "Oregon (OR)", 42: "Pennsylvania (PA)", 44: "Rhode Island (RI)",
        45: "South Carolina (SC)", 46: "South Dakota (SD)", 47: "Tennessee (TN)", 48: "Texas (TX)",
        49: "Utah (UT)", 50: "Vermont (VT)", 51: "Virginia (VA)", 53: "Washington (WA)",
        54: "West Virginia (WV)", 55: "Wisconsin (WI)", 56: "Wyoming (WY)"
    }
    
    # Fetch congressional district data for year 2023
    year = 2023
    cd_df = fetch_congressional_districts(year)
    
    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)
    
    with Session(engine) as session:
        # Truncate existing tables
        session.query(StratumConstraint).delete()
        session.query(Stratum).delete()
        session.commit()
        
        # Create national level stratum
        us_stratum = Stratum(
            parent_stratum_id=None,
            notes="United States",
            stratum_group_id=1,
        )
        us_stratum.constraints_rel = []  # No constraints for national level
        session.add(us_stratum)
        session.flush()
        us_stratum_id = us_stratum.stratum_id
        
        # Track state strata for parent relationships
        state_stratum_ids = {}
        
        # Create state-level strata
        unique_states = cd_df['state_fips'].unique()
        for state_fips in sorted(unique_states):
            state_name = STATE_NAMES.get(state_fips, f"State FIPS {state_fips}")
            state_stratum = Stratum(
                parent_stratum_id=us_stratum_id,
                notes=state_name,
                stratum_group_id=1,
            )
            state_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value=str(state_fips),
                )
            ]
            session.add(state_stratum)
            session.flush()
            state_stratum_ids[state_fips] = state_stratum.stratum_id
        
        # Create congressional district strata
        for _, row in cd_df.iterrows():
            state_fips = row['state_fips']
            cd_geoid = row['congressional_district_geoid']
            name = row['NAME']
            
            cd_stratum = Stratum(
                parent_stratum_id=state_stratum_ids[state_fips],
                notes=f"{name} (CD GEOID {cd_geoid})",
                stratum_group_id=1,
            )
            cd_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="congressional_district_geoid",
                    operation="==",
                    value=str(cd_geoid),
                )
            ]
            session.add(cd_stratum)
        
        session.commit()


if __name__ == "__main__":
    main()
