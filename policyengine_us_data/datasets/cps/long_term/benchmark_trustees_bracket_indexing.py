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


def round_amount(amount: float, rounding: dict | None) -> float:
    if not rounding:
        return amount

    interval = float(rounding["interval"])
    rounding_type = rounding["type"]

    if rounding_type == "downwards":
        return math.floor(amount / interval) * interval
    if rounding_type == "nearest":
        return math.floor(amount / interval + 0.5) * interval

    raise ValueError(f"Unsupported rounding type: {rounding_type}")


def _uprating_parameter_name(parameter) -> str | None:
    metadata = getattr(parameter, "metadata", {})
    uprating = metadata.get("uprating")
    if isinstance(uprating, dict):
        return uprating.get("parameter")
    return uprating


def _iter_updatable_parameters(root, uprating_parameter: str | None = None) -> list:
    candidates = [root]
    if hasattr(root, "get_descendants"):
        candidates.extend(root.get_descendants())

    result = []
    for candidate in candidates:
        if candidate.__class__.__name__ != "Parameter":
            continue
        uprating_name = _uprating_parameter_name(candidate)
        if uprating_name is None:
            continue
        if uprating_parameter is not None and uprating_name != uprating_parameter:
            continue
        result.append(candidate)
    return result


def _apply_wage_growth_to_parameter(parameter, nawi, start_year: int, end_year: int):
    metadata = getattr(parameter, "metadata", {})
    uprating = metadata.get("uprating")
    rounding = uprating.get("rounding") if isinstance(uprating, dict) else None

    for year in range(start_year, end_year + 1):
        previous_value = float(parameter(f"{year - 1}-01-01"))
        wage_growth = float(nawi(f"{year - 1}-01-01")) / float(
            nawi(f"{year - 2}-01-01")
        )
        updated_value = round_amount(previous_value * wage_growth, rounding)
        parameter.update(
            period=f"year:{year}-01-01:1",
            value=updated_value,
        )


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


def create_wage_indexed_core_thresholds_reform(
    start_year: int = 2035,
    end_year: int = 2100,
):
    from policyengine_us.model_api import Reform

    def modify_parameters(parameters):
        nawi = parameters.gov.ssa.nawi
        roots = [
            parameters.gov.irs.income.bracket.thresholds,
            parameters.gov.irs.deductions.standard.amount,
            parameters.gov.irs.deductions.standard.aged_or_blind.amount,
            parameters.gov.irs.capital_gains.thresholds,
            parameters.gov.irs.income.amt.brackets,
            parameters.gov.irs.income.amt.exemption.amount,
            parameters.gov.irs.income.amt.exemption.phase_out.start,
            parameters.gov.irs.income.amt.exemption.separate_limit,
        ]

        seen = set()
        for root in roots:
            for parameter in _iter_updatable_parameters(root):
                if parameter.name in seen:
                    continue
                seen.add(parameter.name)
                _apply_wage_growth_to_parameter(
                    parameter,
                    nawi=nawi,
                    start_year=start_year,
                    end_year=end_year,
                )
        return parameters

    class reform(Reform):
        def apply(self):
            self.modify_parameters(modify_parameters)

    return reform


def create_wage_indexed_irs_uprating_reform(
    start_year: int = 2035,
    end_year: int = 2100,
):
    from policyengine_us.model_api import Reform

    def modify_parameters(parameters):
        nawi = parameters.gov.ssa.nawi
        seen = set()
        for parameter in _iter_updatable_parameters(
            parameters.gov.irs,
            uprating_parameter="gov.irs.uprating",
        ):
            if parameter.name in seen:
                continue
            seen.add(parameter.name)
            _apply_wage_growth_to_parameter(
                parameter,
                nawi=nawi,
                start_year=start_year,
                end_year=end_year,
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
    scenario: str,
) -> dict:
    from policyengine_us import Microsimulation

    if scenario == "brackets":
        reform = create_wage_indexed_brackets_reform(
            start_year=start_year,
            end_year=end_year,
        )
    elif scenario == "core-thresholds":
        reform = create_wage_indexed_core_thresholds_reform(
            start_year=start_year,
            end_year=end_year,
        )
    elif scenario == "irs-uprating":
        reform = create_wage_indexed_irs_uprating_reform(
            start_year=start_year,
            end_year=end_year,
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

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
        "| Year | Scenario | Trustees OASDI target | Baseline OASDI | "
        "Reformed OASDI | OASDI delta | Baseline combined | Reformed combined |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = []
    for record in records:
        baseline_oasdi = record["baseline_oasdi_share_pct"]
        reformed_oasdi = record["reformed_oasdi_share_pct"]
        baseline_combined = record["baseline_combined_share_pct"]
        reformed_combined = record["reformed_combined_share_pct"]
        rows.append(
            "| {year} | {scenario} | {target:.2f}% | {base_o:.2f}% | {reform_o:.2f}% | {delta:+.2f} pp | "
            "{base_c:.2f}% | {reform_c:.2f}% |".format(
                year=record["year"],
                scenario=record["scenario"],
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
        help=(
            "Last year to extend the wage-indexed sensitivity through. "
            "Defaults to the maximum year among the input H5s."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--scenario",
        choices=("brackets", "core-thresholds", "irs-uprating"),
        default="brackets",
        help=(
            "Tax-side sensitivity to run: wage-index only ordinary bracket "
            "thresholds, wage-index a core threshold set, or wage-index the "
            "full IRS uprating path."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.policyengine_us_path:
        sys.path.insert(0, str(Path(args.policyengine_us_path).expanduser()))

    h5_paths = [_coerce_h5_path(path) for path in args.paths]
    end_year = args.end_year or max(int(path.stem) for path in h5_paths)
    records = []

    for h5_path in h5_paths:
        metadata = _load_metadata(h5_path)
        baseline = _baseline_record(h5_path, metadata)
        print(
            f"[{baseline['year']}] benchmarking {args.scenario} on {h5_path}",
            file=sys.stderr,
            flush=True,
        )
        reformed = _compute_reformed_shares(
            h5_path,
            start_year=args.start_year,
            end_year=end_year,
            scenario=args.scenario,
        )
        baseline.update(reformed)
        baseline["scenario"] = args.scenario
        records.append(baseline)

    records.sort(key=lambda record: record["year"])
    if args.format == "json":
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        print(_format_markdown(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
