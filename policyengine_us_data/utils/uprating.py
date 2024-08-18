from policyengine_core.model_api import Variable
import pandas as pd

START_YEAR = 2020
END_YEAR = 2034

def create_policyengine_uprating_factors_table():
    from policyengine_us.system import system

    df = pd.DataFrame()

    variable_names = []
    years = []
    index_values = []

    for variable in system.variables.values():
        if variable.uprating is not None:
            parameter = system.parameters.get_child(variable.uprating)
            start_value = parameter(START_YEAR)
            for year in range(START_YEAR, END_YEAR):
                variable_names.append(variable.name)
                years.append(year)
                index_values.append(round(parameter(year) / start_value, 3))
            
    df["Variable"] = variable_names
    df["Year"] = years
    df["Value"] = index_values

    # Convert to there is a column for each year
    df = df.pivot(index="Variable", columns="Year", values="Value")

    return df.sort_values("Variable")
            
