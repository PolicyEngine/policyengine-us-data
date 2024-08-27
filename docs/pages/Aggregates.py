import streamlit as st

st.title("Aggregates")

st.write(
    """The table below shows the totals for calendar year 2024 for the Enhanced CPS dataset variables."""
)

from policyengine_us import Microsimulation
from policyengine_us_data import EnhancedCPS_2024
from policyengine_us_data.datasets.cps.extended_cps import IMPUTED_VARIABLES as FINANCE_VARIABLES
import pandas as pd

sim = Microsimulation(dataset=EnhancedCPS_2024)

df = pd.DataFrame({
    "Variable": FINANCE_VARIABLES,
    "Total ($bn)": [round(sim.calculate(variable, map_to="household").sum()/1e9, 1) for variable in FINANCE_VARIABLES],
}).sort_values("Total ($bn)", ascending=False).set_index("Variable")

st.dataframe(df, use_container_width=True)