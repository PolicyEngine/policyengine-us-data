from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from policyengine_us_data.storage import STORAGE_FOLDER


SOURCES_DIR = STORAGE_FOLDER / "long_term_target_sources"
LEGACY_AUX_PATH = STORAGE_FOLDER / "social_security_aux.csv"
OACT_DELTA_PATH = SOURCES_DIR / "oasdi_oact_20250805_nominal_delta.csv"
TRUSTEES_OUTPUT_PATH = SOURCES_DIR / "trustees_2025_current_law.csv"
OACT_OUTPUT_PATH = SOURCES_DIR / "oact_2025_08_05_provisional.csv"
MANIFEST_PATH = SOURCES_DIR / "sources.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_trustees_source() -> pd.DataFrame:
    trustees = pd.read_csv(LEGACY_AUX_PATH).copy()
    trustees.to_csv(TRUSTEES_OUTPUT_PATH, index=False)
    return trustees


def build_oact_source(trustees: pd.DataFrame) -> pd.DataFrame:
    delta = pd.read_csv(OACT_DELTA_PATH).copy()
    if 2100 not in set(delta.year):
        delta = pd.concat(
            [
                delta,
                pd.DataFrame(
                    {
                        "year": [2100],
                        "oasdi_nominal_delta_billions": [
                            float(delta.iloc[-1]["oasdi_nominal_delta_billions"])
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )

    merged = trustees.merge(delta, on="year", how="left")
    if merged["oasdi_nominal_delta_billions"].isna().any():
        missing_years = merged.loc[
            merged["oasdi_nominal_delta_billions"].isna(), "year"
        ].tolist()
        raise ValueError(f"Missing OACT OASDI deltas for years: {missing_years}")

    merged["oasdi_tob_billions_nominal_usd"] = (
        merged["oasdi_tob_billions_nominal_usd"]
        + merged["oasdi_nominal_delta_billions"]
    )
    ratio = (
        merged["oasdi_tob_billions_nominal_usd"]
        / trustees["oasdi_tob_billions_nominal_usd"]
    )
    merged["hi_tob_billions_nominal_usd"] = (
        trustees["hi_tob_billions_nominal_usd"] * ratio
    )
    merged["oasdi_tob_pct_of_taxable_payroll"] = (
        merged["oasdi_tob_billions_nominal_usd"]
        / merged["taxable_payroll_in_billion_nominal_usd"]
        * 100
    )
    merged = merged.drop(columns=["oasdi_nominal_delta_billions"])
    merged.to_csv(OACT_OUTPUT_PATH, index=False)
    return merged


def write_manifest() -> None:
    trustees_sha256 = file_sha256(TRUSTEES_OUTPUT_PATH)
    oact_sha256 = file_sha256(OACT_OUTPUT_PATH)
    manifest = {
        "default_source": "trustees_2025_current_law",
        "sources": {
            "trustees_2025_current_law": {
                "name": "trustees_2025_current_law",
                "file": TRUSTEES_OUTPUT_PATH.name,
                "type": "trustees_current_law",
                "description": (
                    "2025 Trustees current-law baseline used by the legacy "
                    "long-term calibration stack."
                ),
                "source_urls": [
                    "https://www.ssa.gov/oact/tr/2025/lrIndex.html",
                    "https://www.ssa.gov/oact/solvency/provisions/tables/table_run133.html",
                ],
                "notes": [
                    "Generated from social_security_aux.csv for explicit source selection.",
                ],
                "baseline_kind": "current_law_comparator",
                "not_law": False,
                "sha256": trustees_sha256,
            },
            "oact_2025_08_05_provisional": {
                "name": "oact_2025_08_05_provisional",
                "file": OACT_OUTPUT_PATH.name,
                "type": "oact_override",
                "description": (
                    "Post-OBBBA SSA OACT baseline overlay with provisional HI "
                    "bridge for long-term calibration experiments."
                ),
                "source_urls": [
                    "https://www.ssa.gov/OACT/solvency/RWyden_20250805.pdf",
                    "https://www.ssa.gov/oact/tr/2025/lrIndex.html",
                ],
                "notes": [
                    "OASDI TOB nominal deltas are taken from the August 5, 2025 OACT letter.",
                    "2100 OASDI delta is carried forward from 2099 because the published delta table ends at 2099.",
                    "HI TOB series is provisional: it applies the same percentage change as OASDI TOB to preserve the OASDI/HI share split until a published annual HI replacement series is available.",
                ],
                "derived_from": "trustees_2025_current_law",
                "hi_method": "match_oasdi_pct_change",
                "scenario_id": "crfb_post_obbba_tob_75y",
                "baseline_kind": "calibration_target",
                "calibration_target_id": "post_obbba_calibrated_tob_75y",
                "law_mode": "trustees-2025-core-thresholds-v1",
                "not_law": True,
                "sha256": oact_sha256,
                "artifact_contract": {
                    "must_consume_baseline_sha256": oact_sha256,
                    "must_expose_scenario_id": "crfb_post_obbba_tob_75y",
                    "reject_raw_current_law_substitution": True,
                },
            },
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build explicit long-term target source packages.",
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    trustees = build_trustees_source()
    build_oact_source(trustees)
    write_manifest()
    print(f"Wrote {TRUSTEES_OUTPUT_PATH}")
    print(f"Wrote {OACT_OUTPUT_PATH}")
    print(f"Wrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
