import streamlit as st
from policyengine_us_data.utils.docs_prerequisites_download import (
    download_data,
)

download_data()

st.title("Distributions")

st.write(
    "This page has several visualisations of the distributions of different variables in the PE-compatible datasets."
)

from policyengine_us_data import CPS_2024, EnhancedCPS_2024, PUF_2024
from policyengine_us_data.utils.soi import (
    pe_to_soi,
    get_soi,
    compare_soi_replication_to_soi,
)
from policyengine_us_data.utils.loss import fmt
import pandas as pd
import plotly.express as px
import numpy as np

st.subheader("IRS SOI")

st.write(
    "Use these controls to see how the different PE-compatible datasets compare to (extrapolated from 2021 to 2024) IRS SOI data."
)


@st.cache_data
def _get_soi(year):
    return get_soi(year)


soi = _get_soi(2024)


@st.cache_data
def get_soi_replication(dataset, year):
    df = compare_soi_replication_to_soi(pe_to_soi(dataset, year), soi)
    return df


variable = st.selectbox("Variable", soi.Variable.unique())
filing_status = st.selectbox("Filing status", soi["Filing status"].unique())
taxable = False
count = st.checkbox("Count")


def get_bar_chart(variable, filing_status, taxable, count):
    df = soi[soi.Variable == variable]
    df["Dataset"] = "SOI"
    for dataset in [EnhancedCPS_2024, PUF_2024, CPS_2024]:
        replication = get_soi_replication(dataset, 2024)
        replication["Dataset"] = dataset.label
        df = pd.concat([df, replication[replication.Variable == variable]])

    df = df[df["Filing status"] == filing_status]
    df = df[df["Taxable only"] == taxable]
    df = df[df["Count"] == count]
    df = df[
        ~(
            (df["AGI lower bound"] == -np.inf)
            & (df["AGI upper bound"] == np.inf)
        )
    ]

    df["AGI lower bound"] = df["AGI lower bound"].apply(fmt)

    return px.bar(
        df,
        x="AGI lower bound",
        y="Value",
        color="Dataset",
        barmode="group",
    )


st.plotly_chart(get_bar_chart(variable, filing_status, taxable, count))
