from policyengine_core.data import Dataset
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.scf.fed_scf import (
    FedSCF,
    FedSCF_2016,
    FedSCF_2019,
    FedSCF_2022,
)
import pandas as pd
import numpy as np
import os
import h5py
from typing import List, Optional, Union, Type


class SCF(Dataset):
    """Dataset containing processed Survey of Consumer Finances data."""

    name = "scf"
    label = "SCF"
    raw_scf: Type[FedSCF] = None
    time_period: int = None
    data_format = Dataset.ARRAYS
    frac: float | None = 1

    def generate(self):
        """Generates the SCF dataset for PolicyEngine US microsimulations.

        Downloads the raw SCF data and processes it for use in PolicyEngine.
        """
        if self.raw_scf is None:
            raise ValueError("No raw SCF data class specified.")

        # Load raw SCF data
        raw_scf_instance = self.raw_scf(require=True)
        raw_data = raw_scf_instance.load()

        # Make sure raw_data is a DataFrame
        if not isinstance(raw_data, pd.DataFrame):
            raise TypeError(
                f"Expected DataFrame but got {type(raw_data)} from {self.raw_scf.name}"
            )

        # Initialize dictionary for arrays
        scf = {}

        # Process the SCF data - convert to dictionary
        add_variables_to_dict(scf, raw_data)
        rename_columns_to_match_cps(scf, raw_data)
        add_auto_loan_interest(scf, self.time_period)

        # Save the dataset - explicitly convert any arrays that aren't numpy arrays
        for key in scf:
            if not isinstance(scf[key], np.ndarray):
                try:
                    scf[key] = np.array(scf[key])
                except Exception as e:
                    print(
                        f"Warning: Could not convert {key} to numpy array: {e}"
                    )

        self.save_dataset(scf)

        # Downsample if needed
        if self.frac is not None and self.frac < 1.0:
            self.downsample(frac=self.frac)

    def load_dataset(self):
        """Loads the processed SCF dataset.

        Returns:
            dict: Dictionary of numpy arrays with preprocessed SCF data.
        """
        # Check if file exists
        if not os.path.exists(self.file_path):
            print(f"SCF dataset file not found. Generating it.")
            self.generate()

        # Open the HDF5 file and handle potential errors
        try:
            with h5py.File(self.file_path, "r") as f:
                # Convert to a dictionary of numpy arrays
                return {key: f[key][()] for key in f.keys()}
        except Exception as e:
            print(f"Error loading HDF5 file: {e}")
            # Alternative loading method - use pandas to load and then convert
            print("Trying alternative loading method...")
            try:
                with pd.HDFStore(self.file_path, mode="r") as store:
                    data = store["data"]
                    # Convert DataFrame to dict of arrays
                    return {col: data[col].values for col in data.columns}
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                # Last resort - regenerate the file
                print("Regenerating dataset...")
                if os.path.exists(self.file_path):
                    os.remove(self.file_path)
                self.generate()
                with h5py.File(self.file_path, "r") as f:
                    return {key: f[key][()] for key in f.keys()}

    def downsample(self, frac: float):
        """Downsamples the SCF dataset.

        Args:
            frac (float): Fraction of data to keep.
        """
        from policyengine_us import Microsimulation

        # Store original dtypes before modifying
        original_data: dict = self.load_dataset()
        original_dtypes = {
            key: original_data[key].dtype for key in original_data
        }

        sim = Microsimulation(dataset=self)
        sim.subsample(frac=frac)

        for key in original_data:
            if key not in sim.tax_benefit_system.variables:
                continue
            values = sim.calculate(key).values

            # Preserve the original dtype if possible
            if (
                key in original_dtypes
                and hasattr(values, "dtype")
                and values.dtype != original_dtypes[key]
            ):
                try:
                    original_data[key] = values.astype(original_dtypes[key])
                except:
                    # If conversion fails, log it but continue
                    print(
                        f"Warning: Could not convert {key} back to {original_dtypes[key]}"
                    )
                    original_data[key] = values
            else:
                original_data[key] = values

        self.save_dataset(original_data)


def add_variables_to_dict(scf: dict, raw_data: pd.DataFrame) -> None:
    """Add all variables directly from the SCF DataFrame to the SCF dictionary.

    Args:
        scf (dict): The SCF dataset dictionary to populate.
        raw_data (pd.DataFrame): The raw SCF data.
    """
    # Simply copy all columns from the DataFrame to the dictionary as numpy arrays
    for column in raw_data.columns:
        if pd.api.types.is_numeric_dtype(raw_data[column]):
            # Handle NaN values for numeric columns
            scf[column] = raw_data[column].fillna(0).values
        elif pd.api.types.is_categorical_dtype(raw_data[column]):
            # Convert categorical to numbers or strings as needed
            scf[column] = raw_data[column].cat.codes.values
        else:
            # Convert object/string columns to numpy arrays
            # For simplicity, convert to string
            scf[column] = raw_data[column].astype(str).values


def rename_columns_to_match_cps(scf: dict, raw_data: pd.DataFrame) -> None:
    """Renames SCF columns to match CPS column names.

    Args:
        scf (dict): The SCF data dictionary to populate.
        raw_data (pd.DataFrame): The raw SCF dataframe.
    """
    # Age variable - directly map
    if "age" in raw_data.columns:
        scf["age"] = raw_data["age"].values

    # Sex → is_female (SCF: hhsex: 1=male, 2=female)
    if "hhsex" in raw_data.columns:
        scf["is_female"] = (raw_data["hhsex"] == 2).values

    # Race → cps_race
    # SCF has multiple race variables: race, racecl, racecl4, racecl5
    if "racecl5" in raw_data.columns:
        # SCF's racecl5: 1=White, 2=Black, 3=Hispanic, 4=Asian, 7=Other
        race_map = {
            1: 1,  # White
            2: 2,  # Black
            3: 3,  # Hispanic
            4: 4,  # Asian
            5: 7,  # Other
        }
        scf["cps_race"] = (
            raw_data["racecl5"].map(race_map).fillna(6).astype(int).values
        )
        # Hispanic indicator
        scf["is_hispanic"] = (raw_data["racecl5"] == 3).values

    # Children in household
    if "kids" in raw_data.columns:
        scf["own_children_in_household"] = (
            raw_data["kids"].fillna(0).astype(int).values
        )

    # Employment & self-employment income
    if "wageinc" in raw_data.columns:
        scf["employment_income"] = raw_data["wageinc"].fillna(0).values
    if "bussefarminc" in raw_data.columns:
        scf["self_employment_income"] = (
            raw_data["bussefarminc"].fillna(0).values
        )
        # Farm income - SCF bundles with business income
        scf["farm_income"] = np.zeros_like(scf["self_employment_income"])

    # Rent
    if "rent" in raw_data.columns:
        scf["rent"] = raw_data["rent"].fillna(0).values

    # Vehicle loan (auto loan)
    if "veh_inst" in raw_data.columns:
        scf["auto_loan_balance"] = raw_data["veh_inst"].fillna(0).values

    # Household weights
    if "wgt" in raw_data.columns:
        scf["household_weight"] = raw_data["wgt"].fillna(0).values

    # Marital status
    if "married" in raw_data.columns:
        # In SCF, married is a binary flag
        scf["is_married"] = (raw_data["married"] == 1).values
        # Create placeholders for other marital statuses
        scf["is_widowed"] = np.zeros(len(raw_data), dtype=bool)
        scf["is_separated"] = np.zeros(len(raw_data), dtype=bool)

    # Additional variables if available in raw_data
    # Financial variables
    variable_mappings = {
        "intdivinc": "interest_income",
        "ssretinc": "social_security_retirement",
        "houses": "real_estate_value",
        "mrthel": "mortgage_debt",
        "edn_inst": "student_loan_debt",
        "ccbal": "credit_card_debt",
    }

    for scf_var, pe_var in variable_mappings.items():
        if scf_var in raw_data.columns:
            scf[pe_var] = raw_data[scf_var].fillna(0).values


def add_auto_loan_interest(scf: dict, year: int) -> None:
    """Adds auto loan interest to the summarized SCF dataset from the full SCF."""
    import requests
    import zipfile
    import io
    import logging
    from tqdm import tqdm

    logger = logging.getLogger(__name__)

    url = f"https://www.federalreserve.gov/econres/files/scf{year}s.zip"

    # Define columns of interest
    columns = ["yy1", "y1", "x2219", "x2319", "x2419", "x7170"]

    try:
        # Download zip file
        logger.info(f"Downloading SCF data for year {year}")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()  # Raise an error for bad responses
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Network error downloading SCF data for year {year}: {str(e)}"
            )
            raise RuntimeError(
                f"Failed to download SCF data for year {year}"
            ) from e

        # Process zip file
        try:
            logger.info("Creating zipfile from downloaded content")
            z = zipfile.ZipFile(io.BytesIO(response.content))

            # Find the .dta file in the zip
            dta_files = [f for f in z.namelist() if f.endswith(".dta")]
            if not dta_files:
                logger.error(f"No Stata files found in zip for year {year}")
                raise ValueError(
                    f"No Stata files found in zip for year {year}"
                )

            logger.info(f"Found Stata files: {dta_files}")

            # Read the Stata file
            try:
                logger.info(f"Reading Stata file: {dta_files[0]}")
                with z.open(dta_files[0]) as f:
                    df = pd.read_stata(io.BytesIO(f.read()), columns=columns)
                    logger.info(f"Read DataFrame with shape {df.shape}")
            except Exception as e:
                logger.error(
                    f"Error reading Stata file for year {year}: {str(e)}"
                )
                raise RuntimeError(
                    f"Failed to process Stata file for year {year}"
                ) from e

        except zipfile.BadZipFile as e:
            logger.error(f"Bad zip file for year {year}: {str(e)}")
            raise RuntimeError(
                f"Downloaded zip file is corrupt for year {year}"
            ) from e

        # Process the interest data and add to final SCF dictionary
        auto_int = df[columns].copy()
        auto_int["x2219"] = auto_int["x2219"].replace(-1, 0)
        auto_int["x2319"] = auto_int["x2319"].replace(-1, 0)
        auto_int["x2419"] = auto_int["x2419"].replace(-1, 0)
        auto_int["x7170"] = auto_int["x7170"].replace(-1, 0)
        # Calculate total auto loan interest (sum of all auto loan interest variables)
        auto_int["auto_loan_interest"] = auto_int[
            ["x2219", "x2319", "x2419", "x7170"]
        ].sum(axis=1)

        # Check if we have household identifiers (y1, yy1) in both datasets
        if (
            "y1" in scf
            and "yy1" in scf
            and "y1" in auto_int.columns
            and "yy1" in auto_int.columns
        ):
            logger.info(
                "Using household identifiers (y1, yy1) to ensure correct matching"
            )

            # Create unique identifier from y1 and yy1 for each dataset
            # In the original data
            auto_int["household_id"] = (
                auto_int["y1"].astype(str) + "_" + auto_int["yy1"].astype(str)
            )

            # In the SCF dictionary
            # Convert the arrays to a temporary DataFrame for easier handling
            temp_scf = pd.DataFrame({"y1": scf["y1"], "yy1": scf["yy1"]})
            temp_scf["household_id"] = (
                temp_scf["y1"].astype(str) + "_" + temp_scf["yy1"].astype(str)
            )

            # Create a mapping from household ID to auto loan interest
            id_to_interest = dict(
                zip(
                    auto_int["household_id"].values,
                    auto_int["auto_loan_interest"].values,
                )
            )

            # Create array for auto loan interest that matches SCF order
            interest_values = np.zeros(len(temp_scf), dtype=float)

            # Fill in interest values based on household ID
            for i, household_id in enumerate(temp_scf["household_id"]):
                if household_id in id_to_interest:
                    interest_values[i] = id_to_interest[household_id]

            # Add to SCF dictionary
            scf["auto_loan_interest"] = interest_values
            logger.info(
                f"Added auto loan interest data for year {year} with household matching"
            )
        else:
            # Fallback to simple assignment if identifiers aren't present
            logger.warning(
                "Household identifiers not found. Using direct array assignment (may not match households correctly)"
            )
            scf["auto_loan_interest"] = auto_int["auto_loan_interest"].values
            logger.info(
                f"Added auto loan interest data for year {year} without household matching"
            )

    except Exception as e:
        logger.error(f"Error processing year {year}: {str(e)}")
        raise


class SCF_2022(SCF):
    """SCF dataset for 2022."""

    name = "scf_2022"
    label = "SCF 2022"
    raw_scf = FedSCF_2022
    file_path = STORAGE_FOLDER / "scf_2022.h5"
    time_period = 2022
    frac = 1


class SCF_2019(SCF):
    """SCF dataset for 2019."""

    name = "scf_2019"
    label = "SCF 2019"
    raw_scf = FedSCF_2019
    file_path = STORAGE_FOLDER / "scf_2019.h5"
    time_period = 2019
    frac = 1


class SCF_2016(SCF):
    """SCF dataset for 2016."""

    name = "scf_2016"
    label = "SCF 2016"
    raw_scf = FedSCF_2016
    file_path = STORAGE_FOLDER / "scf_2016.h5"
    time_period = 2016
    frac = 1


if __name__ == "__main__":
    # Generate all SCF datasets
    SCF_2016().generate()
    SCF_2019().generate()
    SCF_2022().generate()
