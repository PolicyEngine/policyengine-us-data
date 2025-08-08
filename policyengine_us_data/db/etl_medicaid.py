import requests
import pandas as pd



# This is from another file
#def extract_docs(year=2023):
#    docs_url = (
#        f"https://api.census.gov/data/{year}/acs/acs1/subject/variables.json"
#    )
#
#    try:
#        docs_response = requests.get(docs_url)
#        docs_response.raise_for_status()
#
#        docs = docs_response.json()
#        docs["year"] = year
#
#    except requests.exceptions.RequestException as e:
#        print(f"Error during API request: {e}")
#        raise
#    except Exception as e:
#        print(f"An error occurred: {e}")
#        raise
#    return docs



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



# I can get data from:

    "S2704_C02_006E": {
      "label": "Estimate!!Public Coverage!!COVERAGE ALONE OR IN COMBINATION!!Medicaid/means-tested public coverage alone or in combination",
      "concept": "Public Health Insurance Coverage by Type and Selected Characteristics",
      "predicateType": "int",
      "group": "S2704",
      "limit": 0,
      "attributes": "S2704_C02_006EA,S2704_C02_006M,S2704_C02_006MA"
    },


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


cd_survey_df, state_admin_df = extract_medicaid_data()

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

    return long_df

# YOU KNOW WHAT TO DO!

def load_medicaid_data():

    pass

















def _geo_clause_for(geo: str) -> str:
    if geo == "National":
        return "for=us:*"
    if geo == "State":
        return "for=state:*"
    if geo == "District":
        # Congressional districts; adding 'in=state:*' avoids API ambiguities
        return "for=congressional+district:*&in=state:*"
    raise ValueError("geo must be 'National', 'State', or 'District'")


def _group_meta(year: int, dataset: str, group: str) -> dict:
    url = f"https://api.census.gov/data/{year}/{dataset}/groups/{group}.json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def extract_medicaid_s2701(geo: str, year: int = 2023,
                           which: str = "estimate",
                           by_age: bool = True) -> pd.DataFrame:
    """
    Pulls ACS S2701 'With Medicaid/means-tested public coverage' for the requested geography.
      which: 'estimate' (counts) or 'percent'
      by_age: True -> Under 19, 19-64, 65+ ; False -> all ages combined
    Returns: tidy DataFrame with readable columns plus geo identifiers.
    """
    dataset = "acs/acs1/subject"
    group = "S2701"
    meta = _group_meta(year, dataset, group)["variables"]

    target_prefix = "Estimate" if which == "estimate" else "Percent"
    selected, rename = [], {}

    for vid, v in meta.items():
        pass

        if not vid.endswith("E"):  # just the estimates
            continue
        label = v["label"]
        if not label.startswith(target_prefix):
            continue
        ## Keep 'With Medicaid/means-tested public coverage'
        #if "COVERAGE TYPE!!With Medicaid/means-tested public coverage" not in label:
        #    continue

        has_age = "!!AGE!!" in label
        if by_age and not has_age:
            continue
        if not by_age and has_age:
            continue

        selected.append(vid)
        nice = label.split("!!")[-1] if by_age else "All ages"
        rename[vid] = f"Medicaid ({nice}) - {which}"

    if not selected:
        raise RuntimeError("No S2701 Medicaid variables matched. Check 'which' or 'by_age' options.")

    get_vars = ["NAME"] + selected
    url = f"https://api.census.gov/data/{year}/{dataset}?get={','.join(get_vars)}&{_geo_clause_for(geo)}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    raw = r.json()

    df = pd.DataFrame(raw[1:], columns=raw[0])
    for vid in selected:
        df[vid] = pd.to_numeric(df[vid], errors="coerce")
    df = df.rename(columns=rename)

    # Reorder: geo columns first, then NAME, then our measures
    geo_cols = [c for c in ["us", "state", "congressional district"] if c in df.columns]
    measure_cols = [rename[v] for v in selected]
    return df[geo_cols + ["NAME"] + measure_cols]


df = extract_medicaid_s2701("District",
                            2023,
                            "estimate",
                            False)
