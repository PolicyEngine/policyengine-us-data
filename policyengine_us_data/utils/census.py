import pathlib
import requests

import pandas as pd
import numpy as np


def get_census_docs(year):
    docs_url = (
        f"https://api.census.gov/data/{year}/acs/acs1/subject/variables.json"
    )
    # TODO: Alternative: incorporate it!
    "https://api.census.gov/data/2023/acs/acs1/variables.json"

    docs_response = requests.get(docs_url)
    docs_response.raise_for_status()

    return docs_response.json()


def pull_acs_table(group: str, geo: str, year: int) -> pd.DataFrame:
    """
    "group": e.g., 'S2201'
    "geo": 'National' | 'State' | 'District'
    "year": e.g., 2023
    """
    base = f"https://api.census.gov/data/{year}/acs/acs1"
    
    if group[0] == 'S':
         base = base + "/subject"
    geo_q = {
        "National": "us:*",
        "State": "state:*",
        "District": "congressional+district:*",
    }[geo]

    url = f"{base}?get=group({group})&for={geo_q}"

    data = requests.get(url).json()
    headers, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=headers)
    return df
