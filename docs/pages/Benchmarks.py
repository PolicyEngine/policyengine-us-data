import streamlit as st

st.title("Benchmarks")

from policyengine_us_data.datasets import CPS_2024, PUF_2024, EnhancedCPS_2024
from policyengine_us_data.utils import build_loss_matrix
from policyengine_us import Microsimulation
import pandas as pd
import plotly.express as px
import numpy as np


@st.cache_data
def compare_datasets():
    comparison_combined = pd.DataFrame()
    for dataset in [CPS_2024, PUF_2024, EnhancedCPS_2024]:
        sim = Microsimulation(dataset=dataset)
        weights = sim.calculate("household_weight").values
        loss_matrix, targets_array = build_loss_matrix(dataset, 2024)
        target_names = loss_matrix.columns
        estimates = weights @ loss_matrix.values
        comparison = pd.DataFrame(
            {
                "Target": target_names,
                "Estimate": estimates,
                "Actual": targets_array,
            }
        )
        comparison["Error"] = comparison["Estimate"] - comparison["Actual"]
        comparison["Abs. Error"] = comparison["Error"].abs()
        comparison["Abs. Error %"] = (
            (comparison["Abs. Error"] / comparison["Actual"].abs())
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        comparison["Dataset"] = dataset.label
        comparison_combined = pd.concat([comparison_combined, comparison])

    comparison_combined.to_csv("comparisons.csv", index=False)

    return comparison_combined


df = compare_datasets()

mean_relative_error_by_dataset = (
    df.groupby("Dataset")["Abs. Error %"]
    .mean()
    .sort_values(ascending=False)
    .apply(lambda x: round(x, 3))
)

st.write(
    f"PolicyEngine uses **{len(df.Target.unique())}** targets for calibration in the Enhanced CPS. This page compares the estimates and errors for these targets across the three datasets."
)

st.dataframe(mean_relative_error_by_dataset, use_container_width=True)

metric = st.selectbox(
    "Metric", ["Estimate", "Error", "Abs. Error", "Abs. Error %"]
)
target = st.selectbox("Target", df["Target"].unique())

fig = px.bar(
    df[df["Target"] == target],
    x="Dataset",
    y=metric,
    title=f"{metric} for {target}",
)

if metric == "Estimate":
    # Add a dashed line at the target
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=2.5,
        y0=df.loc[df["Target"] == target, "Actual"].values[0],
        y1=df.loc[df["Target"] == target, "Actual"].values[0],
        line=dict(dash="dash"),
    )

st.subheader("Dataset comparisons")
st.write(
    "The chart below, for a selected target and metric, shows the estimates and errors for each dataset."
)

st.plotly_chart(fig, use_container_width=True)

ecps_df = df[df["Dataset"] == "Enhanced CPS 2024"]

st.subheader("Enhanced CPS 2024")
st.write(
    "The table below shows the error for each target in the Enhanced CPS 2024 dataset."
)

st.dataframe(ecps_df, use_container_width=True)
