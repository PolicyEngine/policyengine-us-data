import streamlit as st
from policyengine_us_data.utils.docs_prerequisites_download import (
    download_data,
)

download_data()

st.title("Reforms")

from policyengine_us import Microsimulation
from policyengine_core.reforms import Reform
from pathlib import Path
import pandas as pd

FOLDER = Path(__file__).parent
scores = (
    pd.read_csv(FOLDER / "scores.csv")
    if (FOLDER / "scores.csv").exists()
    else pd.DataFrame(
        {
            "reform_id": [],
            "dataset": [],
            "year": [],
        }
    )
)


@st.cache_data
def get_budget(dataset: str, year: int, reform_id: int = None) -> float:
    from policyengine_us_data import EnhancedCPS_2024, CPS_2024, PUF_2024

    dataset = {ds.name: ds for ds in [EnhancedCPS_2024, CPS_2024, PUF_2024]}[
        dataset
    ]

    if reform_id is None:
        reform = None
    else:
        reform = Reform.from_api(reform_id, "us")

    sim = Microsimulation(dataset=dataset, reform=reform)
    tax_revenues = (
        sim.calculate(
            "household_tax_before_refundable_credits", period=year
        ).sum()
        - sim.calculate("household_refundable_tax_credits", period=year).sum()
    )
    benefit_spending = sim.calculate("household_benefits", period=year).sum()
    govt_balance = tax_revenues - benefit_spending

    return govt_balance


@st.cache_data
def get_budgetary_impact(dataset: str, year: int, reform_id: int) -> float:
    baseline = get_budget(dataset, year)
    with_reform = get_budget(dataset, year, reform_id)
    scores = (
        pd.read_csv(FOLDER / "scores.csv")
        if (FOLDER / "scores.csv").exists()
        else pd.DataFrame(
            {
                "reform_id": [],
                "dataset": [],
                "year": [],
                "budgetary_impact": [],
            }
        )
    )

    if not scores[scores.reform_id == reform_id][scores.dataset == dataset][
        scores.year == year
    ].empty:
        scores = scores.drop(
            scores[scores.reform_id == reform_id][scores.dataset == dataset][
                scores.year == year
            ].index
        )
    scores = pd.concat(
        [
            scores,
            pd.DataFrame(
                {
                    "reform_id": [reform_id],
                    "dataset": [dataset],
                    "year": [year],
                    "budgetary_impact": [
                        round((with_reform - baseline) / 1e9, 1)
                    ],
                }
            ),
        ]
    )
    scores.to_csv(FOLDER / "scores.csv", index=False)


st.write(
    "Use this page to compare the computed budgetary impacts of reforms by dataset."
)

dataset = st.selectbox(
    "Dataset", ["enhanced_cps_2024", "cps_2024", "puf_2024"]
)
num_years = st.slider("Number of years", 1, 11, 3)
reform_id = st.text_input("Reform ID", "1")
reform = Reform.from_api(reform_id, "us")
if reform is not None:
    st.info(reform.name)
    compute = st.button("Compute")
    if compute:
        for year in range(2024, 2024 + num_years):
            get_budgetary_impact(dataset, year, reform_id)

scores = (
    pd.read_csv(FOLDER / "scores.csv")
    if (FOLDER / "scores.csv").exists()
    else pd.DataFrame(
        {"reform_id": [], "dataset": [], "year": [], "budgetary_impact": []}
    )
)
scores.year = scores.year.astype(int)
scores.reform_id = scores.reform_id.astype(int)

# Convert to a table restricted to the given reform with a row for each dataset in scores.csv and a column for each year.

scores_wide = (
    scores[scores.reform_id == int(reform_id)]
    .pivot(index="dataset", columns="year", values="budgetary_impact")
    .fillna(0)
)
st.dataframe(scores_wide, use_container_width=True)
