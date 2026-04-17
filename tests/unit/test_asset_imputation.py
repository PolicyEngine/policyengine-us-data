import numpy as np
import pandas as pd

from policyengine_us_data.utils.asset_imputation import (
    build_household_vehicle_receiver,
)


def test_build_household_vehicle_receiver_aggregates_person_inputs():
    person_df = pd.DataFrame(
        {
            "household_id": [10, 10, 20],
            "employment_income": [20_000.0, 5_000.0, 30_000.0],
            "interest_income": [100.0, 50.0, 25.0],
            "dividend_income": [10.0, 0.0, 5.0],
            "rental_income": [0.0, 200.0, 0.0],
            "age": [42, 12, 35],
            "is_female": [1.0, 1.0, 0.0],
            "is_married": [1.0, 0.0, 0.0],
            "is_household_head": [True, False, True],
        }
    )

    receiver = build_household_vehicle_receiver(
        person_df,
        tenure_type=np.array([b"OWNED_WITH_MORTGAGE", b"RENTED"]),
    )

    assert receiver["household_id"].tolist() == [10, 20]
    assert receiver["household_employment_income"].tolist() == [25_000.0, 30_000.0]
    assert receiver["household_interest_income"].tolist() == [150.0, 25.0]
    assert receiver["household_dividend_income"].tolist() == [10.0, 5.0]
    assert receiver["household_rental_income"].tolist() == [200.0, 0.0]
    assert receiver["count_under_18"].tolist() == [1.0, 0.0]
    assert receiver["household_size"].tolist() == [2.0, 1.0]
    assert receiver["reference_age"].tolist() == [42.0, 35.0]
    assert receiver["reference_is_female"].tolist() == [1.0, 0.0]
    assert receiver["reference_is_married"].tolist() == [1.0, 0.0]
    assert receiver["is_homeowner"].tolist() == [1.0, 0.0]
