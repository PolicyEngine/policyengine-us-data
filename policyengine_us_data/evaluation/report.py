from policyengine_us_data.data_storage import STORAGE_FOLDER
import argparse


def create_report():
    from policyengine_us import Microsimulation
    from policyengine_us_data import CPS_2024
    import pandas as pd

    sim = Microsimulation(dataset=CPS_2024)

    START_YEAR = 2024
    BUDGET_WINDOW = 10

    hnet_totals = []
    years = []
    for year in range(START_YEAR, START_YEAR + BUDGET_WINDOW):
        hnet_totals.append(
            round(sim.calculate("household_net_income", year).sum() / 1e9, 1)
        )
        years.append(year)

    df = pd.DataFrame(
        {"Year": years, "Household net income": hnet_totals}
    ).set_index("Year", drop=True)

    report = f"""# Economy summary

## Household net income
{df.T.to_markdown(index=False)}
"""

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="report.md")
    args = parser.parse_args()
    report = create_report()
    with open(STORAGE_FOLDER / args.output, "w") as f:
        f.write(report)
