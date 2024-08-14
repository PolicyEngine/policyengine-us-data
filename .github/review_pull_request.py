import pandas as pd
from policyengine_us_data.data_storage import STORAGE_FOLDER
from IPython.display import Markdown

def main():
    df = pd.read_csv(STORAGE_FOLDER / "evaluation.csv")

    most_recent_rows = df[df.Date == df.Date.max()].sort_values(["Variable", "Time period"]).set_index(["Variable", "Time period"]).Total
    second_most_recent_rows = df[df.Date == df.Date.sort_values().unique()[-2]].reset_index(drop=True).sort_values(["Variable", "Time period"]).set_index(["Variable", "Time period"]).Total

    diff = (most_recent_rows - second_most_recent_rows)
    # Convert to df
    diff = diff.reset_index()
    table = diff.to_markdown(index=False)

    review_text = f"""## National projection changes\n\n{table}"""

    print(review_text)

if __name__ == "__main__":
    main()