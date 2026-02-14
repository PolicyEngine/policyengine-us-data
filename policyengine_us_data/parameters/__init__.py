"""
Take-up rate parameters for stochastic simulation.

These parameters are stored in the data package to keep the country package
as a purely deterministic rules engine.
"""

import yaml
from pathlib import Path

PARAMETERS_DIR = Path(__file__).parent


def load_take_up_rate(variable_name: str, year: int = 2018):
    """Load take-up rate from YAML parameter files.

    Args:
        variable_name: Name of the take-up parameter file (without .yaml)
        year: Year for which to get the rate

    Returns:
        float, dict (EITC rates_by_children), or dict (Medicaid/TANF
        rates_by_state)
    """
    yaml_path = PARAMETERS_DIR / "take_up" / f"{variable_name}.yaml"

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # EITC: rates by number of children
    if "rates_by_children" in data:
        return data["rates_by_children"]

    # State-specific rates (Medicaid, TANF, etc.)
    if "rates_by_state" in data:
        return data["rates_by_state"]

    # WIC-style: rates by category (each category has a time series)
    if "rates_by_category" in data:
        result = {}
        for category, time_series in data["rates_by_category"].items():
            applicable_value = None
            for y, value in sorted(time_series.items()):
                if int(y) <= year:
                    applicable_value = value
                else:
                    break
            if applicable_value is not None:
                result[category] = applicable_value
        return result

    # Standard time-series values
    values = data["values"]
    applicable_value = None

    for date_key, value in sorted(values.items()):
        if hasattr(date_key, "year"):
            date_year = date_key.year
        else:
            date_year = int(date_key.split("-")[0])

        if date_year <= year:
            applicable_value = value
        else:
            break

    if applicable_value is None:
        raise ValueError(
            f"No take-up rate found for {variable_name} in {year}"
        )

    return applicable_value
