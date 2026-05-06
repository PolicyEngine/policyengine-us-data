from __future__ import annotations

import pandas as pd

from policyengine_us_data.datasets.acs.acs_to_cps_columns import (
    acs_person_to_cps_tax_unit_columns,
)
from policyengine_us_data.datasets.cps.tax_unit_construction import (
    POLICYENGINE_MODE,
    construct_tax_units,
)


def construct_tax_units_acs(
    person: pd.DataFrame,
    year: int,
    mode: str = POLICYENGINE_MODE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cps_like_person = acs_person_to_cps_tax_unit_columns(person)
    assignments, tax_unit = construct_tax_units(
        cps_like_person,
        year=year,
        mode=mode,
    )
    for column in ("acs_spouse_link_imputed", "acs_parent_link_imputed"):
        assignments[column] = cps_like_person[column].values
    return assignments, tax_unit
