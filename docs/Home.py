import streamlit as st
from policyengine_us_data.utils.docs_prerequisites_download import (
    download_data,
)

download_data()

st.title("PolicyEngine-US-Data")

st.write(
    """PolicyEngine-US-Data is a package to create representative microdata for the US, designed for input in the PolicyEngine tax-benefit microsimulation model."""
)

st.subheader("What does this repo do?")

st.write(
    """Principally, this package creates a (partly synthetic) dataset of households (with incomes, demographics and more) that describes the U.S. household sector. This dataset synthesises multiple sources of data (the Current Population Survey, the IRS Public Use File, and administrative statistics) to improve upon the accuracy of **any** of them."""
)

st.subheader("What does this dataset look like?")

st.write(
    "The below table shows an extract of the person records in one household in the dataset."
)


@st.cache_data
def sample_household():
    import pandas as pd
    from policyengine_us_data.datasets import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    df = Microsimulation(dataset=EnhancedCPS_2024).to_input_dataframe()

    household_id = df.person_household_id__2024.values[10]
    people_in_household = df[df.person_household_id__2024 == household_id]
    return people_in_household


people_in_household = sample_household()

st.dataframe(people_in_household.T, use_container_width=True)
