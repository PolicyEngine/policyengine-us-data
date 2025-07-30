"""
Generate synthetic test data for reproducibility testing.

This script creates a small synthetic dataset that mimics the
structure of the Enhanced CPS for testing and demonstration.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_synthetic_cps(n_households=1000, seed=42):
    """Generate synthetic CPS-like data."""

    np.random.seed(seed)

    # Generate household structure
    households = []
    persons = []

    person_id = 0
    for hh_id in range(n_households):
        # Household size (1-6 people)
        hh_size = np.random.choice(
            [1, 2, 3, 4, 5, 6], p=[0.28, 0.34, 0.16, 0.13, 0.06, 0.03]
        )

        # Generate people in household
        for person_num in range(hh_size):
            # Determine role
            if person_num == 0:
                role = "head"
                age = np.random.randint(18, 85)
            elif person_num == 1 and hh_size >= 2:
                role = "spouse"
                age = np.random.randint(18, 85)
            else:
                role = "child"
                age = np.random.randint(0, 25)

            # Generate person data
            person = {
                "person_id": person_id,
                "household_id": hh_id,
                "age": age,
                "sex": np.random.choice([1, 2]),  # 1=male, 2=female
                "person_weight": np.random.uniform(1000, 3000),
                "employment_income": (
                    np.random.lognormal(10, 1.5) if age >= 18 else 0
                ),
                "is_disabled": np.random.random() < 0.15,
                "role": role,
            }

            persons.append(person)
            person_id += 1

        # Generate household data
        household = {
            "household_id": hh_id,
            "state_code": np.random.randint(1, 57),
            "household_weight": np.random.uniform(500, 2000),
            "household_size": hh_size,
            "housing_tenure": np.random.choice(["own", "rent", "other"]),
            "snap_reported": np.random.random() < 0.15,
            "medicaid_reported": np.random.random() < 0.20,
        }

        households.append(household)

    return pd.DataFrame(households), pd.DataFrame(persons)


def generate_synthetic_puf(n_returns=10000, seed=43):
    """Generate synthetic PUF-like data."""

    np.random.seed(seed)

    returns = []

    for i in range(n_returns):
        # Income components (log-normal distributions)
        wages = np.random.lognormal(10.5, 1.2)
        interest = (
            np.random.exponential(500) if np.random.random() < 0.3 else 0
        )
        dividends = (
            np.random.exponential(1000) if np.random.random() < 0.2 else 0
        )
        business = np.random.lognormal(9, 2) if np.random.random() < 0.1 else 0
        cap_gains = (
            np.random.exponential(5000) if np.random.random() < 0.15 else 0
        )

        # Deductions
        mortgage_int = (
            np.random.exponential(8000) if np.random.random() < 0.25 else 0
        )
        charity = (
            np.random.exponential(3000) if np.random.random() < 0.3 else 0
        )
        salt = min(10000, wages * 0.05 + np.random.normal(0, 1000))

        # Demographics (limited in PUF)
        filing_status = np.random.choice(
            [1, 2, 3, 4], p=[0.45, 0.40, 0.10, 0.05]
        )
        num_deps = np.random.choice(
            [0, 1, 2, 3, 4], p=[0.6, 0.15, 0.15, 0.08, 0.02]
        )

        return_data = {
            "return_id": i,
            "filing_status": filing_status,
            "num_dependents": num_deps,
            "age_primary": np.random.randint(18, 85),
            "age_secondary": (
                np.random.randint(18, 85) if filing_status == 2 else 0
            ),
            "wages": wages,
            "interest": interest,
            "dividends": dividends,
            "business_income": business,
            "capital_gains": cap_gains,
            "total_income": wages
            + interest
            + dividends
            + business
            + cap_gains,
            "mortgage_interest": mortgage_int,
            "charitable_deduction": charity,
            "salt_deduction": salt,
            "weight": np.random.uniform(10, 1000),
        }

        returns.append(return_data)

    return pd.DataFrame(returns)


def save_test_data():
    """Generate and save all test datasets."""

    print("Generating synthetic test data...")

    # Create directories
    data_dir = Path("data/test")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate CPS data
    print("- Generating synthetic CPS data...")
    households, persons = generate_synthetic_cps(n_households=1000)

    # Save CPS
    households.to_csv(data_dir / "synthetic_households.csv", index=False)
    persons.to_csv(data_dir / "synthetic_persons.csv", index=False)
    print(f"  Saved {len(households)} households, {len(persons)} persons")

    # Generate PUF data
    print("- Generating synthetic PUF data...")
    puf = generate_synthetic_puf(n_returns=5000)
    puf.to_csv(data_dir / "synthetic_puf.csv", index=False)
    print(f"  Saved {len(puf)} tax returns")

    # Generate expected outputs
    print("- Generating expected outputs...")

    # Simple imputation example
    # Match on age brackets
    age_brackets = [18, 25, 35, 45, 55, 65, 100]
    persons["age_bracket"] = pd.cut(persons["age"], age_brackets)

    # Average wages by age bracket from PUF
    puf["age_bracket"] = pd.cut(puf["age_primary"], age_brackets)
    wage_by_age = puf.groupby("age_bracket")["wages"].mean()

    # Impute to persons
    persons["imputed_wages"] = persons["age_bracket"].map(wage_by_age)
    persons["imputed_wages"] = persons["imputed_wages"].fillna(0)

    # Save enhanced version
    persons.to_csv(data_dir / "synthetic_enhanced_persons.csv", index=False)

    # Generate checksums
    print("- Generating checksums...")
    import hashlib

    checksums = {}
    for file in data_dir.glob("*.csv"):
        with open(file, "rb") as f:
            checksums[file.name] = hashlib.sha256(f.read()).hexdigest()

    with open(data_dir / "checksums.txt", "w") as f:
        for filename, checksum in checksums.items():
            f.write(f"{filename}: {checksum}\n")

    print(f"\nTest data saved to {data_dir}")
    print("Files created:")
    for file in data_dir.glob("*"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    save_test_data()
