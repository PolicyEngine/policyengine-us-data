"""
Take-up rate parameters for stochastic simulation.

These parameters are stored in the data package to keep the country package
as a purely deterministic rules engine.
"""
import yaml
from pathlib import Path

PARAMETERS_DIR = Path(__file__).parent


def load_take_up_rate(variable_name: str, year: int = 2018) -> float:
    """Load take-up rate from YAML parameter files.

    Args:
        variable_name: Name of the take-up parameter file (without .yaml)
        year: Year for which to get the rate

    Returns:
        Take-up rate as a float between 0 and 1
    """
    yaml_path = PARAMETERS_DIR / "take_up" / f"{variable_name}.yaml"

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Handle EITC special case (has rates_by_children instead of values)
    if "rates_by_children" in data:
        return data["rates_by_children"]  # Return the dict

    # Find the applicable value for the year
    values = data["values"]
    applicable_value = None

    for date_key, value in sorted(values.items()):
        # Handle both string and datetime.date objects from YAML
        if hasattr(date_key, "year"):
            # It's a datetime.date object
            date_year = date_key.year
        else:
            # It's a string
            date_year = int(date_key.split("-")[0])

        if date_year <= year:
            applicable_value = value
        else:
            break

    if applicable_value is None:
        raise ValueError(f"No take-up rate found for {variable_name} in {year}")

    return applicable_value
