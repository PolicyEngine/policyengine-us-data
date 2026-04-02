from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


FILING_STATUSES = [
    "SINGLE",
    "JOINT",
    "SEPARATE",
    "HEAD_OF_HOUSEHOLD",
    "SURVIVING_SPOUSE",
]


def round_down(amount: float, interval: float) -> float:
    return math.floor(amount / interval) * interval


def create_wage_indexed_brackets_reform(
    start_year: int = 2035,
    end_year: int = 2100,
):
    from policyengine_us.model_api import Reform

    def modify_parameters(parameters):
        nawi = parameters.gov.ssa.nawi
        thresholds = parameters.gov.irs.income.bracket.thresholds

        for bracket in map(str, range(1, 7)):
            bracket_node = thresholds.get_child(bracket)
            for filing_status in FILING_STATUSES:
                parameter = bracket_node.get_child(filing_status)
                interval = float(
                    parameter.metadata["uprating"]["rounding"]["interval"]
                )

                for year in range(start_year, end_year + 1):
                    previous_value = float(parameter(f"{year - 1}-01-01"))
                    wage_growth = float(nawi(f"{year - 1}-01-01")) / float(
                        nawi(f"{year - 2}-01-01")
                    )
                    updated_value = round_down(previous_value * wage_growth, interval)
                    parameter.update(
                        period=f"year:{year}-01-01:1",
                        value=updated_value,
                    )
        return parameters

    class reform(Reform):
        def apply(self):
            self.modify_parameters(modify_parameters)

    return reform


def _coerce_h5_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_dir():
        matches = sorted(path.glob("*.h5"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one .h5 file in {path}, found {len(matches)}"
            )
        return matches[0]
    if path.suffix == ".metadata.json":
        return path.with_suffix("").with_suffix(".h5")
    return path


def _load_metadata(h5_path: Path) -> dict | None:
    metadata_path = h5_path.with_suffix(".h5.metadata.json")
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _baseline_record(h5_path: Path, metadata: dict | None) -> dict:
    if metadata is None:
        return {
            "year": int(h5_path.stem),
            "baseline_oasdi_share_pct": None,
            "baseline_combined_share_pct": None,
            "target_oasdi_share_pct": None,
        }

    audit = metadata["calibration_audit"]
    ss_total = float(audit["constraints"]["ss_total"]["achieved"])
    ss_target = float(audit["constraints"]["ss_total"]["target"])
    oasdi_actual = float(audit["benchmarks"]["oasdi_tob"]["achieved"])
    hi_actual = float(audit["benchmarks"]["hi_tob"]["achieved"])
    oasdi_target = float(audit["benchmarks"]["oasdi_tob"]["target"])

    return {
        "year": int(metadata["year"]),
        "baseline_oasdi_share_pct": 100 * oasdi_actual / ss_total,
        "baseline_combined_share_pct": 100 * (oasdi_actual + hi_actual) / ss_total,
        "target_oasdi_share_pct": 100 * oasdi_target / ss_target,
    }


def _compute_reformed_shares(
    h5_path: Path,
    start_year: int,
    end_year: int,
) -> dict:
    from policyengine_us import Microsimulation

    reform = create_wage_indexed_brackets_reform(
        start_year=start_year,
        end_year=end_year,
    )
    sim = Microsimulation(dataset=str(h5_path), reform=reform)
    ss_total = float(sim.calculate("social_security").sum())
    oasdi_tob = float(sim.calculate("tob_revenue_oasdi").sum())
    hi_tob = float(sim.calculate("tob_revenue_medicare_hi").sum())
    return {
        "reformed_oasdi_share_pct": 100 * oasdi_tob / ss_total,
        "reformed_combined_share_pct": 100 * (oasdi_tob + hi_tob) / ss_total,
    }


def _format_markdown(records: list[dict]) -> str:
    header = (
        "| Year | Trustees OASDI target | Baseline OASDI | Wage-indexed OASDI | "
        "OASDI delta | Baseline combined | Wage-indexed combined |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = []
    for record in records:
        baseline_oasdi = record["baseline_oasdi_share_pct"]
        reformed_oasdi = record["reformed_oasdi_share_pct"]
        baseline_combined = record["baseline_combined_share_pct"]
        reformed_combined = record["reformed_combined_share_pct"]
        rows.append(
            "| {year} | {target:.2f}% | {base_o:.2f}% | {reform_o:.2f}% | {delta:+.2f} pp | "
            "{base_c:.2f}% | {reform_c:.2f}% |".format(
                year=record["year"],
                target=record["target_oasdi_share_pct"],
                base_o=baseline_oasdi,
                reform_o=reformed_oasdi,
                delta=reformed_oasdi - baseline_oasdi,
                base_c=baseline_combined,
                reform_c=reformed_combined,
            )
        )
    return "\n".join([header, *rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark calibrated long-run H5s under a wage-indexed ordinary "
            "income-tax bracket sensitivity."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="H5 files, metadata files, or directories containing a single H5.",
    )
    parser.add_argument(
        "--policyengine-us-path",
        help=(
            "Optional local policyengine-us checkout to prepend to sys.path. "
            "Use this when the required tax-side fix is not yet released."
        ),
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2035,
        help="First year to switch ordinary federal brackets from CPI to wages.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2100,
        help="Last year to extend the wage-indexed bracket sensitivity through.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.policyengine_us_path:
        sys.path.insert(0, str(Path(args.policyengine_us_path).expanduser()))

    h5_paths = [_coerce_h5_path(path) for path in args.paths]
    records = []

    for h5_path in h5_paths:
        metadata = _load_metadata(h5_path)
        baseline = _baseline_record(h5_path, metadata)
        print(
            f"[{baseline['year']}] benchmarking wage-indexed federal brackets on {h5_path}",
            file=sys.stderr,
            flush=True,
        )
        reformed = _compute_reformed_shares(
            h5_path,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        baseline.update(reformed)
        records.append(baseline)

    records.sort(key=lambda record: record["year"])
    if args.format == "json":
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        print(_format_markdown(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
