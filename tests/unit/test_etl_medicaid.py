import pandas as pd

from policyengine_us_data.db.etl_medicaid import transform_administrative_chip_data


def test_transform_administrative_chip_data_preserves_reported_zero_enrollment():
    source = pd.DataFrame(
        {
            "State Abbreviation": ["RI", "RI", "CA"],
            "Reporting Period": [202411, 202412, 202412],
            "Final Report": ["Y", "Y", "Y"],
            "Total CHIP Enrollment": [12_345, 0, 1_232_909],
        }
    )

    transformed = transform_administrative_chip_data(source, 2024)

    assert transformed.to_dict("records") == [
        {"ucgid_str": "0400000US44", "chip_enrollment": 0},
        {"ucgid_str": "0400000US06", "chip_enrollment": 1_232_909},
    ]
