"""
Compare PE-US tax outcomes under Census TAX_ID vs constructed CPS tax units.

This harness keeps all non-tax-unit staged CPS arrays fixed and swaps only the
tax-unit graph. It intentionally does not pass tax-unit role or filing-status
inputs: those are PE-US tax rules, and the outcome benchmark shows constructed
IDs perform best when PE-US infers those quantities from the new unit graph.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from policyengine_core.data import Dataset

from policyengine_us_data.datasets.cps.tax_unit_construction import construct_tax_units
from policyengine_us_data.utils.soi import (
    compare_soi_replication_to_soi,
    get_soi,
    pe_to_soi,
)


CONSTRUCTION_USECOLS = [
    "PH_SEQ",
    "P_SEQ",
    "A_LINENO",
    "A_AGE",
    "A_MARITL",
    "A_SPOUSE",
    "PECOHAB",
    "PEPAR1",
    "PEPAR2",
    "A_EXPRRP",
    "A_ENRLW",
    "A_FTPT",
    "A_HSCOL",
    "WSAL_VAL",
    "SEMP_VAL",
    "FRSE_VAL",
    "INT_VAL",
    "DIV_VAL",
    "RNT_VAL",
    "CAP_VAL",
    "UC_VAL",
    "OI_VAL",
    "ANN_VAL",
    "PNSN_VAL",
    "PTOTVAL",
    "SS_VAL",
    "PEDISDRS",
    "PEDISEAR",
    "PEDISEYE",
    "PEDISOUT",
    "PEDISPHY",
    "PEDISREM",
    "TAX_ID",
]


def load_public_person_file(input_path: Path, csv_name: str | None) -> pd.DataFrame:
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as zf:
            selected_name = csv_name
            if selected_name is None:
                matches = [
                    name
                    for name in zf.namelist()
                    if name.lower().startswith("pppub")
                    and name.lower().endswith(".csv")
                ]
                if not matches:
                    raise FileNotFoundError(
                        f"No pppub*.csv person file found in {input_path}."
                    )
                selected_name = sorted(matches)[0]
            with zf.open(selected_name) as f:
                return pd.read_csv(
                    f,
                    usecols=lambda col: col in CONSTRUCTION_USECOLS,
                    low_memory=False,
                )
    return pd.read_csv(
        input_path,
        usecols=lambda col: col in CONSTRUCTION_USECOLS,
        low_memory=False,
    )


def load_staged_arrays(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {key: np.array(value) for key, value in f.items()}


def make_constructed_id_dataset(
    base: dict[str, np.ndarray],
    public_person: pd.DataFrame,
    year: int,
) -> dict[str, np.ndarray]:
    raw_person_id = (
        public_person.PH_SEQ.astype(int) * 100 + public_person.P_SEQ.astype(int)
    ).to_numpy()
    if not np.array_equal(raw_person_id, base["person_id"]):
        raise ValueError(
            "Public CPS person file does not align with staged CPS person order."
        )

    assignments, tax_unit = construct_tax_units(public_person, year)
    data = {
        key: value
        for key, value in base.items()
        if key
        not in {
            "person_tax_unit_id",
            "tax_unit_id",
            "tax_unit_role_input",
            "filing_status_input",
            "is_related_to_head_or_spouse",
        }
    }
    data["person_tax_unit_id"] = assignments["TAX_ID"].to_numpy(dtype=np.int64)
    data["tax_unit_id"] = tax_unit["TAX_ID"].to_numpy(dtype=np.int64)
    return data


def dataset_class(
    name: str,
    data: dict[str, np.ndarray],
    file_path: Path,
    year: int,
) -> type[Dataset]:
    class InMemoryDataset(Dataset):
        def load(self, key: str | None = None, mode: str = "r"):
            del mode
            if key is None:
                return data
            return data[key]

        def load_dataset(self):
            return data

    InMemoryDataset.name = name
    InMemoryDataset.label = name
    InMemoryDataset.data_format = Dataset.ARRAYS
    InMemoryDataset.time_period = year
    InMemoryDataset.file_path = file_path
    return InMemoryDataset


def _metric_summary(frame: pd.DataFrame) -> dict:
    relative_error = frame["Relative error"].to_numpy(dtype=float)
    absolute_relative_error = np.abs(relative_error)
    return {
        "rows": int(len(frame)),
        "mean_abs_relative_error": float(absolute_relative_error.mean()),
        "median_abs_relative_error": float(np.median(absolute_relative_error)),
        "rmse_relative_error": float(np.sqrt(np.mean(relative_error**2))),
        "p90_abs_relative_error": float(np.quantile(absolute_relative_error, 0.9)),
    }


def summarize_soi_fit(
    data: dict[str, np.ndarray],
    name: str,
    staged_path: Path,
    year: int,
) -> dict:
    soi_df = pe_to_soi(dataset_class(name, data, staged_path, year), year)
    comparison = compare_soi_replication_to_soi(soi_df, get_soi(year))

    taxable = comparison[
        comparison["Taxable only"]
        & (comparison["AGI upper bound"] > 10_000)
        & comparison["SOI Value"].ne(0)
    ]
    selected = comparison[
        comparison["Variable"].isin(
            [
                "adjusted_gross_income",
                "count",
                "income_tax_before_credits",
                "income_tax_after_credits",
            ]
        )
        & (comparison["AGI upper bound"] > 10_000)
        & comparison["SOI Value"].ne(0)
    ]
    aggregate = comparison[
        comparison["Full population"] & comparison["SOI Value"].ne(0)
    ]
    filers = soi_df[soi_df["is_tax_filer"].astype(bool)]

    return {
        "tax_units": int(len(soi_df)),
        "weighted_filers": float(filers["weight"].sum()),
        "filing_status_counts": {
            status: float(group["weight"].sum())
            for status, group in filers.groupby("filing_status")
        },
        "taxable_soi_rows": _metric_summary(taxable),
        "selected_tax_rows": _metric_summary(selected),
        "aggregate_soi_rows": _metric_summary(aggregate),
    }


def compute_outcome_comparison(
    staged_path: Path,
    public_person_path: Path,
    year: int,
    csv_name: str | None = None,
) -> dict:
    base = load_staged_arrays(staged_path)
    public_person = load_public_person_file(public_person_path, csv_name)
    constructed = make_constructed_id_dataset(base, public_person, year)

    census = summarize_soi_fit(base, "census_tax_units", staged_path, year)
    policyengine = summarize_soi_fit(
        constructed,
        "policyengine_tax_units",
        staged_path,
        year,
    )

    return {
        "year": year,
        "staged_dataset": str(staged_path),
        "public_person_file": str(public_person_path),
        "census_tax_units": census,
        "policyengine_tax_units": policyengine,
        "deltas_policyengine_minus_census": {
            "tax_units": policyengine["tax_units"] - census["tax_units"],
            "weighted_filers": policyengine["weighted_filers"]
            - census["weighted_filers"],
            "taxable_soi_rows_mean_abs_relative_error": policyengine[
                "taxable_soi_rows"
            ]["mean_abs_relative_error"]
            - census["taxable_soi_rows"]["mean_abs_relative_error"],
            "taxable_soi_rows_rmse_relative_error": policyengine["taxable_soi_rows"][
                "rmse_relative_error"
            ]
            - census["taxable_soi_rows"]["rmse_relative_error"],
            "selected_tax_rows_mean_abs_relative_error": policyengine[
                "selected_tax_rows"
            ]["mean_abs_relative_error"]
            - census["selected_tax_rows"]["mean_abs_relative_error"],
            "selected_tax_rows_rmse_relative_error": policyengine["selected_tax_rows"][
                "rmse_relative_error"
            ]
            - census["selected_tax_rows"]["rmse_relative_error"],
            "aggregate_soi_rows_mean_abs_relative_error": policyengine[
                "aggregate_soi_rows"
            ]["mean_abs_relative_error"]
            - census["aggregate_soi_rows"]["mean_abs_relative_error"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PE-US SOI tax outcome fit under Census TAX_ID versus "
            "constructed CPS tax units."
        )
    )
    parser.add_argument("staged_path", type=Path)
    parser.add_argument("public_person_path", type=Path)
    parser.add_argument("--csv-name", default=None)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compute_outcome_comparison(
        staged_path=args.staged_path,
        public_person_path=args.public_person_path,
        year=args.year,
        csv_name=args.csv_name,
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
