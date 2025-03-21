import pytest
import os
import pandas as pd
from policyengine_us_data.geography import (
    COUNTY_FIPS_DATASET_PATH,
    COUNTY_FIPS_DATASET,
)


# Test that the dataset CSV exists in the repo
def test_county_fips_exists():
    assert os.path.exists(COUNTY_FIPS_DATASET_PATH)


# Test that the dataset CSV is not empty
def test_county_fips_not_empty():
    df = pd.read_csv(COUNTY_FIPS_DATASET_PATH, compression="gzip")
    assert len(df) > 0


# Test that read-in dataset contains no NaNs
def test_county_fips_no_nans():
    assert not COUNTY_FIPS_DATASET.isnull().values.any()


# Test that columns are of correct data type
@pytest.mark.parametrize(
    "column_name, dtype",
    [
        ("county_fips", str),
        ("county_name", str),
        ("state", str),
    ],
)
def test_county_fips_column_dtypes(column_name, dtype):
    assert all(isinstance(x, dtype) for x in COUNTY_FIPS_DATASET[column_name])


# Test that particularly relevant counties are correct
# Counties where we have API partners:
# * Los Angeles, CA
# * Denver, CO
# Counties with high populations:
# * New York, NY
# * Cook, IL
# * Harris, TX
# Edge-case counties:
# * District of Columbia, DC
# * Oglala Lakota, SD (recent name change)
# * Mayagüez, PR
# * Guam, GU
@pytest.mark.parametrize(
    "county_fips, county_name, state",
    [
        ("06037", "Los Angeles County", "CA"),
        ("08031", "Denver County", "CO"),
        ("36061", "New York County", "NY"),
        ("17031", "Cook County", "IL"),
        ("48201", "Harris County", "TX"),
        ("11001", "District of Columbia", "DC"),
        ("46102", "Oglala Lakota County", "SD"),
        ("72097", "Mayagüez Municipio", "PR"),
        ("66010", "Guam", "GU"),
    ],
)
def test_county_fips_values(county_fips, county_name, state):
    county = COUNTY_FIPS_DATASET[
        COUNTY_FIPS_DATASET["county_fips"] == county_fips
    ]
    assert len(county) == 1
    assert county["county_name"].values[0] == county_name
    assert county["state"].values[0] == state
