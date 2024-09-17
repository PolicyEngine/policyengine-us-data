from policyengine_us_data.storage import STORAGE_FOLDER
import pandas as pd

START_YEAR = 2020
END_YEAR = 2034


def create_policyengine_uprating_factors_table():
    from policyengine_us.system import system

    df = pd.DataFrame()

    variable_names = []
    years = []
    index_values = []

    population_size = system.parameters.get_child(
        "calibration.gov.census.populations.total"
    )

    for variable in system.variables.values():
        if variable.uprating is not None:
            parameter = system.parameters.get_child(variable.uprating)
            start_value = parameter(START_YEAR)
            for year in range(START_YEAR, END_YEAR + 1):
                population_growth = population_size(year) / population_size(
                    START_YEAR
                )
                variable_names.append(variable.name)
                years.append(year)
                growth = parameter(year) / start_value
                if "_weight" not in variable.name:
                    per_capita_growth = growth / population_growth
                else:
                    per_capita_growth = growth
                index_values.append(round(per_capita_growth, 3))

    # Add population growth

    for year in range(START_YEAR, END_YEAR + 1):
        variable_names.append("population")
        years.append(year)
        index_values.append(
            round(population_size(year) / population_size(START_YEAR), 3)
        )

    df["Variable"] = variable_names
    df["Year"] = years
    df["Value"] = index_values

    # Convert to there is a column for each year
    df = df.pivot(index="Variable", columns="Year", values="Value")
    df = df.sort_values("Variable")
    df.to_csv(STORAGE_FOLDER / "uprating_factors.csv")

    # Create a table with growth factors by year

    df_growth = df.copy()
    for year in range(END_YEAR, START_YEAR, -1):
        df_growth[year] = round(df_growth[year] / df_growth[year - 1] - 1, 3)
    df_growth[START_YEAR] = 0

    df_growth.to_csv(STORAGE_FOLDER / "uprating_growth_factors.csv")
    return df
