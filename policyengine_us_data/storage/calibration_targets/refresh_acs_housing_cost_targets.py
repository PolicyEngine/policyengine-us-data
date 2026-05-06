import csv
import json
from urllib.request import urlopen

from policyengine_us_data.storage import CALIBRATION_FOLDER
from policyengine_us_data.storage.calibration_targets.pull_soi_targets import (
    STATE_ABBR_TO_FIPS,
)


YEAR = 2024
ACS_DATASET = "acs/acs1"
STATE_FIPS_TO_ABBR = {
    fips: state_code for state_code, fips in STATE_ABBR_TO_FIPS.items()
}


def fetch_acs_housing_cost_targets(year: int = YEAR) -> list[dict]:
    """Fetch ACS state rent and property-tax aggregates.

    B25060 is aggregate monthly contract rent for renter-occupied units
    paying cash rent. We annualize it to match the yearly `rent` variable.
    B25090 is aggregate real estate taxes paid by owner-occupied units.
    """
    variables = "NAME,B25060_001E,B25090_001E"
    url = (
        f"https://api.census.gov/data/{year}/{ACS_DATASET}?get={variables}&for=state:*"
    )
    with urlopen(url) as response:
        rows = json.load(response)

    header = rows[0]
    column_index = {column: index for index, column in enumerate(header)}

    targets = []
    for row in rows[1:]:
        state_fips = row[column_index["state"]]
        state_code = STATE_FIPS_TO_ABBR.get(state_fips)
        if state_code is None:
            continue

        monthly_contract_rent = float(row[column_index["B25060_001E"]])
        real_estate_taxes = float(row[column_index["B25090_001E"]])
        targets.append(
            {
                "state_code": state_code,
                "state_fips": state_fips,
                "annual_contract_rent": int(monthly_contract_rent * 12),
                "real_estate_taxes": int(real_estate_taxes),
            }
        )

    return sorted(targets, key=lambda target: target["state_code"])


def main() -> None:
    targets = fetch_acs_housing_cost_targets()
    output_path = CALIBRATION_FOLDER / f"acs_housing_costs_{YEAR}.csv"
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "state_code",
                "state_fips",
                "annual_contract_rent",
                "real_estate_taxes",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(targets)


if __name__ == "__main__":
    main()
