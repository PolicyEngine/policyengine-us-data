"""Build the per-variable uprating factor CSV from policyengine-us parameters.

For each uprated variable we emit a per-capita growth factor indexed at
``START_YEAR = 2020``. The factor is

    growth(variable, year) = parameter(year) / parameter(START_YEAR)

If the underlying parameter is a dollar aggregate (a "total"), we
divide by population growth to recover the per-capita factor. If the
parameter is *already* a per-capita index — CPI, the SSA uprating
index, per-capita spending, a benefit-per-recipient rate, etc. — we
must *not* divide by population growth or we would double-adjust and
introduce a small but compounding downward drift (roughly
0.5-1%/year) in the emitted factor.

The previous implementation only special-cased ``"_weight" in
variable.name`` and divided everything else by population growth,
which silently double-adjusted CPI/SSA-indexed variables. This module
now drives the per-capita divide from the parameter path rather than
the variable name, so parameters known to be per-capita indices (by
path substring match) skip the divisor.
"""

from policyengine_us_data.storage import STORAGE_FOLDER
import pandas as pd

START_YEAR = 2020
END_YEAR = 2034


# Parameter-path substrings that mark a parameter as already-per-capita
# or an index (i.e., growth(year) is already a per-capita ratio, and
# dividing by population growth would double-adjust it).
#
# Rule: if any of these substrings appears in the parameter path, we
# skip the population-growth divisor. Ordered roughly by frequency.
PER_CAPITA_PARAMETER_PATH_MARKERS: tuple[str, ...] = (
    "gov.bls.cpi",  # BLS CPI series (already an index)
    "per_capita",  # per-capita series (spending, moop, etc.)
    "gov.ssa.uprating",  # SSA benefit-uprating COLA (an index)
    ".uprating.",  # any other explicit ``uprating`` parameter (an index)
    ".index.",  # paths under an ``index`` sub-tree
    "per_recipient",  # benefit-per-recipient rates
    "per_worker",  # wage-per-worker indices
)


def is_per_capita_parameter(parameter_path: str) -> bool:
    """Return True if the uprating parameter is already per-capita.

    Path-substring heuristic so new parameter additions can opt in by
    name without touching this module (any path containing any of
    :data:`PER_CAPITA_PARAMETER_PATH_MARKERS`). Parameter names that
    are *total* aggregates (e.g., `calibration.gov.irs.soi.*`,
    `calibration.gov.census.populations.total`) do not match any of
    the markers and therefore still get divided by population growth.
    """
    return any(marker in parameter_path for marker in PER_CAPITA_PARAMETER_PATH_MARKERS)


def create_policyengine_uprating_factors_table():
    from policyengine_us.system import system

    df = pd.DataFrame()

    variable_names = []
    years = []
    index_values = []

    population_size = system.parameters.get_child(
        "calibration.gov.census.populations.total"
    )

    # Cache population growth factors outside the variable loop — they
    # do not depend on ``variable``.
    population_growth_by_year = {
        year: population_size(year) / population_size(START_YEAR)
        for year in range(START_YEAR, END_YEAR + 1)
    }

    for variable in system.variables.values():
        if variable.uprating is None:
            continue
        parameter_path = variable.uprating
        parameter = system.parameters.get_child(parameter_path)
        start_value = parameter(START_YEAR)
        skip_population_divisor = is_per_capita_parameter(parameter_path) or (
            "_weight" in variable.name
        )
        for year in range(START_YEAR, END_YEAR + 1):
            variable_names.append(variable.name)
            years.append(year)
            growth = parameter(year) / start_value
            if skip_population_divisor:
                per_capita_growth = growth
            else:
                per_capita_growth = growth / population_growth_by_year[year]
            index_values.append(round(per_capita_growth, 3))

    # Add population growth

    for year in range(START_YEAR, END_YEAR + 1):
        variable_names.append("population")
        years.append(year)
        index_values.append(round(population_growth_by_year[year], 3))

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
