import os

import requests
import pandas as pd

# The user needs to have a CENSUS_API_KEY environment variable set to run this code
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
YEAR = 2023

# Get national median 2BR rent
national_url = (
    f"https://api.census.gov/data/{YEAR}/acs/acs5"
    f"?get=B25031_004E"
    f"&for=us:*"
    f"&key={CENSUS_API_KEY}"
)
national_rent = float(requests.get(national_url).json()[1][0])
print(f"National median 2BR rent ({YEAR}): ${national_rent:,.0f}")

# Get all congressional districts (loop through states)
all_rows = []
for state_fips in range(1, 57):
    state_str = f"{state_fips:02d}"
    url = (
        f"https://api.census.gov/data/{YEAR}/acs/acs5"
        f"?get=B25031_004E,NAME"
        f"&for=congressional%20district:*"
        f"&in=state:{state_str}"
        f"&key={CENSUS_API_KEY}"
    )
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            for row in data[1:]:
                all_rows.append({
                    "state_fips": row[2],
                    "district": row[3],
                    "cd_id": row[2] + row[3],
                    "name": row[1],
                    "median_2br_rent": float(row[0]) if row[0] else None,
                })
    except Exception as e:
        pass

df = pd.DataFrame(all_rows)
df["national_median_2br_rent"] = national_rent

print(f"\nFetched {len(df)} congressional districts")
print(df.head(10))

outfile = f"./national_and_district_rents_{YEAR}.csv"
df.to_csv(outfile, index=False)
print(outfile)
