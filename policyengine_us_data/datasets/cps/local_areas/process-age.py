import requests
import pandas as pd
import io

# USER INPUT -----------
GEO = "District"
GEO = "State"
GEO = "National"
YEAR = 2022
# END USER INPUT -------

base_url = f"https://api.census.gov/data/{YEAR}/acs/acs1/subject?get=group(S0101)"
docs_url = f"https://api.census.gov/data/{YEAR}/acs/acs1/subject/variables.json"

if GEO == "State":
    url = f"{base_url}&for=state:*"
elif GEO == "District":
    url = f"{base_url}&for=congressional+district:*"
elif GEO == "National":
    url = f"{base_url}&for=us:*"
else:
    raise ValueError("GEO must be either 'National', 'State', or 'District'")


try:
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    docs_response = requests.get(docs_url)
    docs_response.raise_for_status()

    docs = docs_response.json()

    headers = data[0]
    data_rows = data[1:]
    df = pd.DataFrame(data_rows, columns=headers)

except requests.exceptions.RequestException as e:
    print(f"Error during API request: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Keys: descriptions of the variables we want. Values: short names
label_to_short_name_mapping = {
    'Estimate!!Total!!Total population!!AGE!!Under 5 years': '0-4',
    'Estimate!!Total!!Total population!!AGE!!5 to 9 years': '5-9',
    'Estimate!!Total!!Total population!!AGE!!10 to 14 years': '10-14',
    'Estimate!!Total!!Total population!!AGE!!15 to 19 years': '15-19',
    'Estimate!!Total!!Total population!!AGE!!20 to 24 years': '20-24',
    'Estimate!!Total!!Total population!!AGE!!25 to 29 years': '25-29',
    'Estimate!!Total!!Total population!!AGE!!30 to 34 years': '30-34',
    'Estimate!!Total!!Total population!!AGE!!35 to 39 years': '35-39',
    'Estimate!!Total!!Total population!!AGE!!40 to 44 years': '40-44',
    'Estimate!!Total!!Total population!!AGE!!45 to 49 years': '45-49',
    'Estimate!!Total!!Total population!!AGE!!50 to 54 years': '50-54',
    'Estimate!!Total!!Total population!!AGE!!55 to 59 years': '55-59',
    'Estimate!!Total!!Total population!!AGE!!60 to 64 years': '60-64',
    'Estimate!!Total!!Total population!!AGE!!65 to 69 years': '65-69',
    'Estimate!!Total!!Total population!!AGE!!70 to 74 years': '70-74',
    'Estimate!!Total!!Total population!!AGE!!75 to 79 years': '75-79',
    'Estimate!!Total!!Total population!!AGE!!80 to 84 years': '80-84',
    'Estimate!!Total!!Total population!!AGE!!85 years and over': '85+',
}

# map the documentation labels to the actual data set variables
label_to_variable_mapping = dict([
  (value['label'], key) for key, value in docs['variables'].items()
      if value['group'] == "S0101" and value['concept'] == "Age and Sex"
      and value['label'] in label_to_short_name_mapping.keys()
])

# By transitivity, map the data set variable names to short names
rename_mapping = dict(
    [
        (label_to_variable_mapping[v], label_to_short_name_mapping[v])
        for v in label_to_short_name_mapping.keys()
    ]
)

df_data = df.rename(columns=rename_mapping)[
    ['GEO_ID', 'NAME'] + list(label_to_short_name_mapping.values())
]

# Filter out non-voting districts, e.g., DC and Puerto Rico
df_geos = df_data[
    ~df_data['GEO_ID'].isin(
        ['5001800US7298', '5001800US1198', '0400000US72', '0400000US11']
    )
].copy()

omitted_rows = df_data[~df_data['GEO_ID'].isin(df_geos['GEO_ID'])]
print(f"Ommitted {GEO} geographies:\n\n{omitted_rows[['GEO_ID', 'NAME']]}")

if GEO == 'District':
    assert(df_geos.shape[0] == 435)
    df_geos.to_csv("./districts/targets/age-districts.csv", index=False)
elif GEO == 'State':
    assert(df_geos.shape[0] == 50)
    df_geos.to_csv("./states/targets/age-state.csv", index=False)
elif GEO == 'National':
    assert(df_geos.shape[0] == 1)
    df_geos.to_csv("./age-national.csv", index=False)
