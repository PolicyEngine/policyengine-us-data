from policyengine_us_data.data_storage import STORAGE_FOLDER
from policyengine_core.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Type


def evaluate_dataset(dataset: Type[Dataset]):
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=dataset)

    KEY_VARIABLES = [
        "household_net_income",
        "income_tax",
        "snap",
        "ssi",
    ]

    KEY_TIME_PERIODS = [
        2024,
        2025,
        2026,
        2027,
        2028,
        2029,
        2030,
        2031,
        2032,
        2033,
    ]
    variables = []
    time_periods = []
    totals = []

    for time_period in KEY_TIME_PERIODS[:3]:
        for variable in KEY_VARIABLES:
            total = round(sim.calculate(variable, time_period).sum() / 1e9, 1)
            variables.append(variable)
            time_periods.append(time_period)
            totals.append(total)

    df = pd.DataFrame(
        {
            "Variable": variables,
            "Time period": time_periods,
            "Total": totals,
        }
    )

    df["Date"] = pd.Timestamp("now")
    df["Dataset"] = dataset.name

    file_path = Path(STORAGE_FOLDER / "evaluation.csv")
    if file_path.exists():
        existing_df = pd.read_csv(file_path)
        df = pd.concat([existing_df, df])

    df.to_csv(file_path, index=False)


def main():
    from policyengine_us_data.datasets import DATASETS

    for dataset in DATASETS:
        evaluate_dataset(dataset)


if __name__ == "__main__":
    main()
