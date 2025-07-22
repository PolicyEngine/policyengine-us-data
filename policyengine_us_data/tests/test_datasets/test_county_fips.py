import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO, BytesIO
from pathlib import Path

# Import the function to test
from policyengine_us_data.geography.county_fips import (
    generate_county_fips_2020_dataset,
    LOCAL_FOLDER,
)


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
def mock_requests_get(mock_response):
    """Mock requests.get to return our mock response"""
    with patch("requests.get", return_value=mock_response) as mock_get:
        yield mock_get


@pytest.fixture
def mock_upload_to_hf():
    """Mock the upload_to_hf function"""
    with patch(
        "policyengine_us_data.geography.county_fips.upload_to_hf"
    ) as mock_upload:
        yield mock_upload


@pytest.fixture
def mock_local_folder():
    """Mock the LOCAL_FOLDER"""
    mock_path = MagicMock()
    with patch(
        "policyengine_us_data.geography.county_fips.LOCAL_FOLDER", mock_path
    ):
        yield mock_path


@pytest.fixture
def mock_to_csv():
    """Mock pandas DataFrame.to_csv"""
    with patch("pandas.DataFrame.to_csv") as mock_csv:
        yield mock_csv


@pytest.fixture
def expected_dataframe():
    """Create the expected dataframe after processing"""
    data = {
        "state": ["AL", "AL", "NY", "NY"],
        "county_name": [
            "Autauga County",
            "Baldwin County",
            "Albany County",
            "Bronx County",
        ],
        "county_fips": ["01001", "01003", "36001", "36003"],
    }
    return pd.DataFrame(data)


def test_successful_download_and_processing(
    mock_response,
    mock_upload_to_hf,
    mock_to_csv,
    mock_requests_get,
    expected_dataframe,
):
    """Test the entire function with a successful download and processing"""

    # Run the function
    generate_county_fips_2020_dataset()

    # Check that upload_to_hf was called
    mock_upload_to_hf.assert_called_once()


def test_download_failure():
    """Test handling of download failure"""

    # Create a mock response with error status code
    failed_response = MagicMock()
    failed_response.status_code = 404

    with (
        patch("requests.get", return_value=failed_response),
        pytest.raises(ValueError) as excinfo,
    ):

        # Run the function, expect ValueError
        generate_county_fips_2020_dataset()

    # Check error message contains status code
    assert "404" in str(excinfo.value)


def test_dataframe_transformation(
    mock_response,
    mock_requests_get,
    mock_upload_to_hf,
    mock_to_csv,
    expected_dataframe,
):
    """Test the transformation of the raw data into the expected dataframe"""

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
    assert all(len(fips) == 5 for fips in result_df["county_fips"])


def test_output_file_creation(
    mock_upload_to_hf, mock_to_csv, mock_requests_get, mock_local_folder
):
    """Test that the local output file is created correctly"""

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_CENSUS_DATA

    # Create a mock path object
    mock_path = MagicMock()

    generate_county_fips_2020_dataset()

    # Check that to_csv was called with gzip compression
    kwargs = mock_to_csv.call_args_list[-1][1]  # Get kwargs of the last call
    assert kwargs.get("compression") == "gzip"


def test_huggingface_upload(mock_upload_to_hf, mock_to_csv, mock_requests_get):
    """Test that the upload to Hugging Face is called with the correct parameters"""

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_CENSUS_DATA

    generate_county_fips_2020_dataset()

    # Check that upload_to_hf was called with the correct repo and file path
    call_kwargs = mock_upload_to_hf.call_args[1]
    assert call_kwargs["repo"] == "policyengine/policyengine-us-data"
    assert call_kwargs["repo_file_path"] == "county_fips_2020.csv.gz"

    # Verify that the first parameter is a BytesIO instance
    assert isinstance(
        mock_upload_to_hf.call_args[1]["local_file_path"], BytesIO
    )
