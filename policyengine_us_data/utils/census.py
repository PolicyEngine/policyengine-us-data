import pathlib
import requests

import pandas as pd
import numpy as np


STATE_NAME_TO_FIPS = {
    "Alabama": "01",
    "Alaska": "02",
    "Arizona": "04",
    "Arkansas": "05",
    "California": "06",
    "Colorado": "08",
    "Connecticut": "09",
    "Delaware": "10",
    "District of Columbia": "11",
    "Florida": "12",
    "Georgia": "13",
    "Hawaii": "15",
    "Idaho": "16",
    "Illinois": "17",
    "Indiana": "18",
    "Iowa": "19",
    "Kansas": "20",
    "Kentucky": "21",
    "Louisiana": "22",
    "Maine": "23",
    "Maryland": "24",
    "Massachusetts": "25",
    "Michigan": "26",
    "Minnesota": "27",
    "Mississippi": "28",
    "Missouri": "29",
    "Montana": "30",
    "Nebraska": "31",
    "Nevada": "32",
    "New Hampshire": "33",
    "New Jersey": "34",
    "New Mexico": "35",
    "New York": "36",
    "North Carolina": "37",
    "North Dakota": "38",
    "Ohio": "39",
    "Oklahoma": "40",
    "Oregon": "41",
    "Pennsylvania": "42",
    "Rhode Island": "44",
    "South Carolina": "45",
    "South Dakota": "46",
    "Tennessee": "47",
    "Texas": "48",
    "Utah": "49",
    "Vermont": "50",
    "Virginia": "51",
    "Washington": "53",
    "West Virginia": "54",
    "Wisconsin": "55",
    "Wyoming": "56",
}


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
