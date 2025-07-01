import requests
import zipfile
import io
import pandas as pd
import us

from policyengine_us_data.storage import STORAGE_FOLDER


def extract_usda_snap_data(year=2023):
    """
    Downloads and extracts annual state-level SNAP data from the USDA FNS zip file.
    """
    url = "https://www.fns.usda.gov/sites/default/files/resource-files/snap-zip-fy69tocurrent-6.zip"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    filename = f"FY{str(year)[-2:]}.xlsx"
    with zip_file.open(filename) as f:
        xls = pd.ExcelFile(f)
        tab_results = []
        for sheet_name in [
            "NERO",
            "MARO",
            "SERO",
            "MWRO",
            "SWRO",
            "MPRO",
            "WRO",
        ]:
            df_raw = pd.read_excel(
                xls, sheet_name=sheet_name, header=None, dtype={0: str}
            )

            state_row_mask = (
                df_raw[0].notna()
                & df_raw[1].isna()
                & ~df_raw[0].str.contains("Total", na=False)
                & ~df_raw[0].str.contains("Footnote", na=False)
            )

            df_raw["State"] = df_raw.loc[state_row_mask, 0]
            df_raw["State"] = df_raw["State"].ffill()
            total_rows = df_raw[df_raw[0].eq("Total")].copy()
            total_rows = total_rows.rename(
                columns={
                    1: "Households",
                    2: "Persons",
                    3: "Cost",
                    4: "CostPerHousehold",
                    5: "CostPerPerson",
                }
            )

            state_totals = total_rows[
                [
                    "State",
                    "Households",
                    "Persons",
                    "Cost",
                    "CostPerHousehold",
                    "CostPerPerson",
                ]
            ]

            tab_results.append(state_totals)

    results_df = pd.concat(tab_results)

    state_equivs = us.states.STATES + [us.states.DC]
    fips_lookup = {state.name: state.fips.zfill(2) for state in state_equivs}

    df_states = results_df.loc[
        results_df["State"].isin(fips_lookup.keys())
    ].copy()
    df_states["STATE_FIPS"] = df_states["State"].map(fips_lookup)
    df_states = (
        df_states.loc[~df_states["STATE_FIPS"].isna()]
        .sort_values("STATE_FIPS")
        .reset_index(drop=True)
    )
    df_states["GEO_ID"] = "0400000US" + df_states["STATE_FIPS"]

    return df_states[["GEO_ID", "Households", "Cost"]]


def main() -> None:
    out_dir = STORAGE_FOLDER
    state_df = extract_usda_snap_data(2023)
    state_df.to_csv(out_dir / "snap_state.csv", index=False)


if __name__ == "__main__":
    main()
