import streamlit as st

st.title("PolicyEngine-US-Data")

st.write(
    """PolicyEngine-US-Data is a package to create representative microdata for the US, designed for input in the PolicyEngine tax-benefit microsimulation model."""
)

st.subheader("What does this repo do?")

st.write(
    """Principally, this package creates a (partly synthetic) dataset of households (with incomes, demographics and more) that describes the U.S. household sector. This dataset synthesises multiple sources of data (the Current Population Survey, the IRS Public Use File, and administrative statistics) to improve upon the accuracy of **any** of them."""
)
