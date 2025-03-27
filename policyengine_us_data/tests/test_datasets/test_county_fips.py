# import pytest
# import os
# import pandas as pd
# from policyengine_us_data.geography import (
#     COUNTY_FIPS_DATASET_PATH,
#     COUNTY_FIPS_DATASET,
# )
# 
# 
# 
# 
# 
# # Test that the dataset CSV exists in the repo
# def test_county_fips_exists():
#     assert os.path.exists(COUNTY_FIPS_DATASET_PATH)
# 
# 
# # Test that the dataset CSV is not empty
# def test_county_fips_not_empty():
#     df = pd.read_csv(COUNTY_FIPS_DATASET_PATH, compression="gzip")
#     assert len(df) > 0
# 
# 
# # Test that read-in dataset contains no NaNs
# def test_county_fips_no_nans():
#     assert not COUNTY_FIPS_DATASET.isnull().values.any()
# 
# 
# # Test that columns are of correct data type
# @pytest.mark.parametrize(
#     "column_name, dtype",
#     [
#         ("county_fips", str),
#         ("county_name", str),
#         ("state", str),
#     ],
# )
# def test_county_fips_column_dtypes(column_name, dtype):
#     assert all(isinstance(x, dtype) for x in COUNTY_FIPS_DATASET[column_name])
# 
# 
# # Test that particularly relevant counties are correct
# # Counties where we have API partners:
# # * Los Angeles, CA
# # * Denver, CO
# # Counties with high populations:
# # * New York, NY
# # * Cook, IL
# # * Harris, TX
# # Edge-case counties:
# # * District of Columbia, DC
# # * Oglala Lakota, SD (recent name change)
# # * Mayagüez, PR
# # * Guam, GU
# @pytest.mark.parametrize(
#     "county_fips, county_name, state",
#     [
#         ("06037", "Los Angeles County", "CA"),
#         ("08031", "Denver County", "CO"),
#         ("36061", "New York County", "NY"),
#         ("17031", "Cook County", "IL"),
#         ("48201", "Harris County", "TX"),
#         ("11001", "District of Columbia", "DC"),
#         ("46102", "Oglala Lakota County", "SD"),
#         ("72097", "Mayagüez Municipio", "PR"),
#         ("66010", "Guam", "GU"),
#     ],
# )
# def test_county_fips_values(county_fips, county_name, state):
#     county = COUNTY_FIPS_DATASET[
#         COUNTY_FIPS_DATASET["county_fips"] == county_fips
#     ]
#     assert len(county) == 1
#     assert county["county_name"].values[0] == county_name
#     assert county["state"].values[0] == state




import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO, BytesIO
from pathlib import Path

# Import the function to test
from policyengine_us_data.geography.county_fips import generate_county_fips_2020_dataset, LOCAL_FOLDER


# Sample data that mimics the format from census.gov
SAMPLE_CENSUS_DATA = """STATE|STATEFP|COUNTYFP|COUNTYNAME
AL|01|001|Autauga County
AL|01|003|Baldwin County
NY|36|001|Albany County
NY|36|003|Bronx County
"""


@pytest.fixture
def mock_response():
    """Create a mock response object that mimics a successful requests.get"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_CENSUS_DATA
    return mock_resp


@pytest.fixture
def expected_dataframe():
    """Create the expected dataframe after processing"""
    data = {
        "state": ["AL", "AL", "NY", "NY"],
        "county_name": ["Autauga County", "Baldwin County", "Albany County", "Bronx County"],
        "county_fips": ["01001", "01003", "36001", "36003"],
    }
    return pd.DataFrame(data)


def test_successful_download_and_processing(mock_response, expected_dataframe):
    """Test the entire function with a successful download and processing"""
    
    with patch("requests.get", return_value=mock_response), \
         patch("policyengine_us_data.geography.county_fips.upload_to_hf") as mock_upload, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        
        # Run the function
        generate_county_fips_2020_dataset()
        
        # Check that upload_to_hf was called
        mock_upload.assert_called_once()
        
        # Check that to_csv was called with the right path
        local_csv_call = mock_to_csv.call_args_list[-1]
        assert str(LOCAL_FOLDER / "county_fips.csv.gz") in str(local_csv_call)


def test_download_failure():
    """Test handling of download failure"""
    
    # Create a mock response with error status code
    failed_response = MagicMock()
    failed_response.status_code = 404
    
    with patch("requests.get", return_value=failed_response), \
         pytest.raises(ValueError) as excinfo:
        
        # Run the function, expect ValueError
        generate_county_fips_2020_dataset()
    
    # Check error message contains status code
    assert "404" in str(excinfo.value)


def test_dataframe_transformation(mock_response, expected_dataframe):
    """Test the transformation of the raw data into the expected dataframe"""
    
    with patch("requests.get", return_value=mock_response), \
         patch("policyengine_us_data.geography.county_fips.upload_to_hf"), \
         patch("pandas.DataFrame.to_csv"):
        
        # Create a way to capture the dataframe before it's uploaded
        original_to_csv = pd.DataFrame.to_csv
        
        def capture_df(self, *args, **kwargs):
            # Store the dataframe for inspection
            capture_df.result_df = self.copy()
            return original_to_csv(self, *args, **kwargs)
        
        with patch("pandas.DataFrame.to_csv", capture_df):
            generate_county_fips_2020_dataset()
        
        # Get the captured dataframe
        result_df = capture_df.result_df
        
        # Check columns
        assert list(result_df.columns) == list(expected_dataframe.columns)
        
        # Check data content
        for col in result_df.columns:
            assert result_df[col].tolist() == expected_dataframe[col].tolist()
        
        # Ensure FIPS codes are correctly formatted (5 digits)
        assert all(len(fips) == 5 for fips in result_df['county_fips'])


def test_output_file_creation():
    """Test that the local output file is created correctly"""
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_CENSUS_DATA
    
    # Create a mock path object
    mock_path = MagicMock()
    
    with patch("requests.get", return_value=mock_resp), \
         patch("policyengine_us_data.geography.county_fips.upload_to_hf"), \
         patch("policyengine_us_data.geography.county_fips.LOCAL_FOLDER", mock_path), \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
        
        generate_county_fips_2020_dataset()
        
        # Check that to_csv was called with gzip compression
        kwargs = mock_to_csv.call_args_list[-1][1]  # Get kwargs of the last call
        assert kwargs.get('compression') == 'gzip'


def test_huggingface_upload():
    """Test that the upload to Hugging Face is called with the correct parameters"""
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_CENSUS_DATA
    
    with patch("requests.get", return_value=mock_resp), \
         patch("policyengine_us_data.geography.county_fips.upload_to_hf") as mock_upload, \
         patch("pandas.DataFrame.to_csv"):
        
        generate_county_fips_2020_dataset()
        
        # Check that upload_to_hf was called with the correct repo and file path
        call_kwargs = mock_upload.call_args[1]
        assert call_kwargs['repo'] == "policyengine/policyengine-us-data"
        assert call_kwargs['repo_file_path'] == "county_fips_2020.csv.gz"
        
        # Verify that the first parameter is a BytesIO instance
        assert isinstance(mock_upload.call_args[1]['local_file_path'], BytesIO)