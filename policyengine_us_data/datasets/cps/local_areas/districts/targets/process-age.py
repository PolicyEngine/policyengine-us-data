#  ## Description
#  This file contains estimated population counts by age group for United States Congressional Districts (including non-voting delegate/resident commissioner districts) based on the 2023 American Community Survey 1-Year Estimates. Each row represents one district or equivalent area.
#  
#  ## Data Source
#  * **Survey/Product:** U.S. Census Bureau American Community Survey (ACS) 1-Year Estimates
#  * **Year:** 2023
#  * **Table ID:** S0101 (Selected Population Profile)
#  * **Source File:** `ACSST1Y2023.S0101-Data.csv` (Downloaded from data.census.gov or similar source)
#  
#  ## Geographic Level
#  * **Level:** Congressional Districts (118th Congress)
#  * **Coverage:** Includes the 435 standard voting districts from the 50 states, plus the non-voting districts for the District of Columbia and Puerto Rico. *(Adjust this statement if you excluded DC/PR)*.
#  
import pandas as pd
import re

# --- Configuration ---
data_file = 'ACSST1Y2023.S0101-Data.csv'

# Define the columns we need, using 'Geography' instead of 'GEO_ID'
acs_columns_needed = {
    'Geography': 'code', # *** CORRECTED *** Geography ID
    'Geographic Area Name': 'name', # District Name
    'Estimate!!Total!!Total population': 'all', # Total Population
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

# --- Load Data ---
try:
    df_data = pd.read_csv(data_file, header=1)
    print(f"Successfully loaded data from {data_file}")
    print(f"Initial data shape: {df_data.shape}")
    # Optional: Print columns again to confirm
    # print("Columns in df_data:", df_data.columns)

except FileNotFoundError:
    print(f"Error: Could not find the data file '{data_file}'. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data file: {e}")
    exit()

# --- Filter for Congressional Districts ---
# Filter using the correct column name 'Geography'
df_districts = df_data[
    ~df_data['Geography'].isin(['5001800US7298', '5001800US1198'])  # Non-voting members, DC & Puerto Rico
].copy()

# There are 435 congressional districts in the United States with voting rights
assert(df_districts.shape[0] == 435)

omitted_rows = df_data[~df_data['Geography'].isin(df_districts['Geography'])]
print(omitted_rows[['Geography', 'Geographic Area Name']])

# --- Select and Rename Columns ---
available_acs_columns = [col for col in acs_columns_needed.keys() if col in df_districts.columns]
missing_acs_columns = [col for col in acs_columns_needed.keys() if col not in df_districts.columns]

if missing_acs_columns:
    print("\nWarning: The following expected columns were not found in the data file:")
    for col in missing_acs_columns:
        print(f"- {col}")

rename_mapping = {k: v for k, v in acs_columns_needed.items() if k in available_acs_columns}

df_processed = df_districts[available_acs_columns].copy()
df_processed.rename(columns=rename_mapping, inplace=True)

print("\nSelected and renamed columns.")

# --- Clean Data ---
def clean_district_name(name):
    match = re.match(r"Congressional District (\d+|At Large)(?: \(.+\))?, (.+)", name)
    if match:
        district_num = match.group(1)
        state = match.group(2)
        # Handle cases like Washington D.C. which might not have a typical state name
        if state.lower() == 'district of columbia':
             state_abbr = 'DC' # Or however you want to represent it
        # Add other state name to abbreviation mappings if needed, or keep full name
        # else:
        #     # Basic state abbreviation example (needs a full map for accuracy)
        #     state_abbr_map = {'Alabama': 'AL', 'Alaska': 'AK', ...}
        #     state_abbr = state_abbr_map.get(state, state)
        return f"{state} - District {district_num}"
    return name.replace('(118th Congress)', '').strip()

df_processed['name'] = df_processed['name'].apply(clean_district_name)
print("Cleaned district names.")

population_cols = list(rename_mapping.values())
# Ensure 'code' and 'name' exist before trying to remove
if 'code' in population_cols:
    population_cols.remove('code')
if 'name' in population_cols:
    population_cols.remove('name')

for col in population_cols:
    if col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

df_processed[population_cols] = df_processed[population_cols].fillna(0).astype(int)

print("Converted population columns to numeric and filled missing values with 0.")

# --- Reorder Columns ---
# Define the desired order, ensuring 'code' and 'name' are handled correctly
desired_order = ['code', 'name', 'all']
# Dynamically get the remaining age columns and sort them
age_cols = sorted([col for col in df_processed.columns if col not in desired_order],
                  key=lambda x: int(re.search(r'\d+', x.split('-')[0].replace('+', '999')).group())) # Improved sorting key

final_column_order = desired_order + age_cols
final_column_order = [col for col in final_column_order if col in df_processed.columns] # Ensure all exist
df_final = df_processed[final_column_order]


# --- Display Results ---
print("\n--- Processed Data Sample (First 5 Rows) ---")
print(df_final.head())

print("\n--- Column Names and Data Types ---")
print(df_final.info())

# --- Save to CSV (Optional) ---
output_filename = 'us_congressional_districts_age_groups_2023.csv'
# df_final.to_csv(output_filename, index=False)
# print(f"\nProcessed data saved to {output_filename}")

print("\nNote: The age columns represent groups (e.g., 0-4, 5-9) based on available ACS S0101 data,")
print("not single years of age as in the original UK example.")

from pathlib import Path

df_final.to_csv(
    Path('/mnt/c/devl/policyengine-us-data/policyengine_us_data/datasets/cps/local_areas/districts/targets') / 'age.csv'
)
