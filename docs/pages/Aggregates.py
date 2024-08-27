import streamlit as st

st.title("Aggregates")

from policyengine_us import Microsimulation
from policyengine_us_data import EnhancedCPS_2024

sim = Microsimulation(dataset=EnhancedCPS_2024)
