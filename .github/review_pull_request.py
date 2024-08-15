import pandas as pd
from policyengine_us_data.data_storage import STORAGE_FOLDER
from policyengine_us_data.utils.github import set_pr_auto_review_comment

def main():
    df = pd.read_csv(STORAGE_FOLDER / "evaluation.csv")

    most_recent_rows = df[df.Date == df.Date.max()].sort_values(["Variable", "Time period"]).set_index(["Variable", "Time period"]).Total
    second_most_recent_rows = df[df.Date == df.Date.sort_values().unique()[-2]].reset_index(drop=True).sort_values(["Variable", "Time period"]).set_index(["Variable", "Time period"]).Total

    diff = (most_recent_rows - second_most_recent_rows)
    # Convert to df
    diff = diff.reset_index()
    diff = diff[diff.Variable == "household_net_income"].T
    table = diff.to_markdown(index=False)

    review_text = f"""## National projection changes\n\nThis pull request makes the following changes to economic estimates.\n\n{table}"""

    print(review_text)

    set_pr_auto_review_comment(review_text)

if __name__ == "__main__":
    main()