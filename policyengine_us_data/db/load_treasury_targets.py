import logging
import requests
from pathlib import Path
import io

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)


logger = logging.getLogger(__name__)


def extract_eitc_data():
    # IRS Table 2.5, Tax Year 2020S
    url = "https://www.irs.gov/pub/irs-soi/20in25ic.xls"
    r   = requests.get(url, timeout=30)
    r.raise_for_status()
    
    # Pandas uses xlrd to open .xls
    xls = pd.ExcelFile(io.BytesIO(r.content), engine="xlrd")
    sheets = {name: xls.parse(name, header=None) for name in xls.sheet_names}
    
    raw = sheets[xls.sheet_names[0]]
    return raw


def transform_eitc_data(raw_data):
    # This is not ideal from a data processing standpoint, but it's too much
    # effort to fully parse this hierarchical XLS for a few data points
    # At least the full lineage is represented from the source

    zero_children_returns = raw_data.iloc[8, 25]
    zero_children_amount = raw_data.iloc[8, 26] * 1000
    
    one_child_returns = raw_data.iloc[8, 39]
    one_child_amount = raw_data.iloc[8, 40] * 1000
    
    two_children_returns = raw_data.iloc[8, 57]
    two_children_amount = raw_data.iloc[8, 58] * 1000

    three_plus_children_returns = raw_data.iloc[8, 73]
    three_plus_children_amount = raw_data.iloc[8, 74] * 1000

    assert zero_children_returns == 7636714 
    assert zero_children_amount ==  2255068000

    df_long = pd.DataFrame([
        ["0100000US", "children_equal_to", 0, "tax_unit_count", zero_children_returns],
        ["0100000US", "children_equal_to", 1, "tax_unit_count", one_child_returns],
        ["0100000US", "children_equal_to", 2, "tax_unit_count", two_children_returns],
        ["0100000US", "children_greater_or_equal_to", 3, "tax_unit_count", three_plus_children_returns],
        ["0100000US", "children_equal_to", 0, "eitc", zero_children_amount],
        ["0100000US", "children_equal_to", 1, "eitc", one_child_returns],
        ["0100000US", "children_equal_to", 2, "eitc", two_children_returns],
        ["0100000US", "children_greater_or_equal_to", 3, "eitc", three_plus_children_returns],
    ])

    df_long.columns = ["ucgid", "constraint", "constraint_value", "variable", "value"] 
   
    df_long["period"] = 2020
    df_long["reform_id"] = 0
    df_long["source_id"] = 2
    df_long["active"] = True

    return df_long


def load_eitc_data(df_long):

    DATABASE_URL = "sqlite:///policy_data.db"
    engine = create_engine(DATABASE_URL)

    Session = sessionmaker(bind=engine)
    session = Session()

    ucgid = df_long.iloc[0]['ucgid']
    for num_children in [0, 1, 2, 3]:
        note = f"eitc_child_count: {num_children}, Geo: {ucgid}"
        new_stratum = Stratum(
            parent_stratum_id=None, stratum_group_id=0, notes=note
        )

        new_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="ucgid",
                operation="equals",
                value=ucgid,
            ),
        ]

        if num_children <= 2:
            new_stratum.constraints_rel.append(
                StratumConstraint(
                        constraint_variable="eitc_child_count",
                        operation="equals",
                        value=str(num_children),
                ),
            )
        elif num_children > 2:
            new_stratum.constraints_rel.append(
                StratumConstraint(
                        constraint_variable="eitc_child_count",
                        operation="greater_or_equal_than",
                        value=str(3),
                ),
            )

        rows = df_long.loc[df_long['constraint_value'] == num_children]
        count_target = rows.loc[rows.variable == 'tax_unit_count']['value'].values[0]
        amount_target = rows.loc[rows.variable == 'eitc']['value'].values[0]

        # Avoiding magic numbers in the load step
        count_active = rows.loc[rows.variable == 'tax_unit_count']['active'].values[0]
        amount_active = rows.loc[rows.variable == 'eitc']['active'].values[0]

        period = rows.iloc[0]['period']
        source_id = rows.iloc[0]['source_id']

        new_stratum.targets_rel = [
            Target(
                variable="eitc",
                period=period,
                value=amount_target,
                source_id=source_id,
                active=amount_active,
            ),
            Target(
                variable="tax_unit_count",
                period=period,
                value=amount_target,
                source_id=source_id,
                active=count_active,
            ),
        ]

        session.add(new_stratum)
        session.flush()
        print(new_stratum.stratum_id)

    session.commit()


if __name__ == "__main__":

    # --- ETL: Extract, Transform, Load ----

    # ---- Extract ----------
    national_df = extract_eitc_data()

    # --- Transform ----------
    long_national_df = transform_eitc_data(national_df)

    # --- Load --------
    state_strata_lku = load_eitc_data(long_national_df)
