"""
Validate staging .h5 files by running sim.calculate() and comparing
against calibration targets from policy_data.db.

Usage:
    python -m policyengine_us_data.calibration.validate_staging \
        --area-type states,districts --areas NC \
        --period 2024 --output validation_results.csv

    python -m policyengine_us_data.calibration.validate_staging \
        --sanity-only --area-type states --areas NC
"""

import argparse
import csv
import gc
import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.calibration.unified_calibration import (
    load_target_config,
    _match_rules,
)
from policyengine_us_data.calibration.unified_matrix_builder import (
    UnifiedMatrixBuilder,
    _calculate_target_values_standalone,
    _GEO_VARS,
)
from policyengine_us_data.calibration.calibration_utils import (
    STATE_CODES,
)
from policyengine_us_data.calibration.sanity_checks import (
    run_sanity_checks,
)

logger = logging.getLogger(__name__)

DEFAULT_HF_PREFIX = "hf://policyengine/policyengine-us-data/staging"
DEFAULT_DB_PATH = str(STORAGE_FOLDER / "calibration" / "policy_data.db")
DEFAULT_TARGET_CONFIG = str(Path(__file__).parent / "target_config_full.yaml")
TRAINING_TARGET_CONFIG = str(Path(__file__).parent / "target_config.yaml")

SANITY_CEILINGS = {
    "national": {
        "dollar": 30e12,
        "person_count": 340e6,
        "household_count": 135e6,
        "count": 340e6,
    },
    "state": {
        "dollar": 5e12,
        "person_count": 40e6,
        "household_count": 15e6,
        "count": 40e6,
    },
    "district": {
        "dollar": 500e9,
        "person_count": 1e6,
        "household_count": 400e3,
        "count": 1e6,
    },
}

FIPS_TO_ABBR = {str(k): v for k, v in STATE_CODES.items()}
ABBR_TO_FIPS = {v: str(k) for k, v in STATE_CODES.items()}

CSV_COLUMNS = [
    "area_type",
    "area_id",
    "variable",
    "target_name",
    "period",
    "target_value",
    "sim_value",
    "error",
    "rel_error",
    "abs_error",
    "rel_abs_error",
    "sanity_check",
    "sanity_reason",
    "in_training",
]


def _classify_variable(variable: str) -> str:
    if "household_count" in variable:
        return "household_count"
    if "person_count" in variable:
        return "person_count"
    if variable.endswith("_count"):
        return "count"
    return "dollar"


def _run_sanity_check(
    sim_value: float,
    variable: str,
    geo_level: str,
) -> tuple:
    if not math.isfinite(sim_value):
        return "FAIL", "non-finite value"
    vtype = _classify_variable(variable)
    ceilings = SANITY_CEILINGS.get(geo_level, SANITY_CEILINGS["state"])
    ceiling = ceilings.get(vtype, ceilings["dollar"])
    if abs(sim_value) > ceiling:
        return (
            "FAIL",
            f"|{sim_value:.2e}| > {ceiling:.0e} ceiling "
            f"({vtype} @ {geo_level})",
        )
    return "PASS", ""


def _query_all_active_targets(engine, period: int) -> pd.DataFrame:
    query = """
    WITH best_periods AS (
        SELECT stratum_id, variable,
            CASE
                WHEN MAX(CASE WHEN period <= :period
                         THEN period END) IS NOT NULL
                THEN MAX(CASE WHEN period <= :period
                         THEN period END)
                ELSE MIN(period)
            END as best_period
        FROM target_overview
        WHERE active = 1
        GROUP BY stratum_id, variable
    )
    SELECT tv.target_id, tv.stratum_id, tv.variable,
           tv.value, tv.period, tv.geo_level,
           tv.geographic_id, tv.domain_variable
    FROM target_overview tv
    JOIN best_periods bp
        ON tv.stratum_id = bp.stratum_id
        AND tv.variable = bp.variable
        AND tv.period = bp.best_period
    WHERE tv.active = 1
    ORDER BY tv.target_id
    """
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"period": period})


def _get_stratum_constraints(engine, stratum_id: int) -> list:
    query = """
    SELECT constraint_variable AS variable, operation, value
    FROM stratum_constraints
    WHERE stratum_id = :stratum_id
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"stratum_id": int(stratum_id)})
    return df.to_dict("records")


def _geoid_to_district_filename(geoid: str) -> str:
    """Convert DB geographic_id like '3701' to filename 'NC-01'."""
    geoid = geoid.zfill(4)
    state_fips = geoid[:-2]
    district_num = geoid[-2:]
    abbr = FIPS_TO_ABBR.get(state_fips)
    if abbr is None:
        return geoid
    return f"{abbr}-{district_num}"


def _geoid_to_display(geoid: str) -> str:
    """Convert DB geographic_id like '3701' to 'NC-01'."""
    return _geoid_to_district_filename(geoid)


def _resolve_state_fips(areas_str: Optional[str]) -> list:
    """Resolve --areas to state FIPS codes."""
    if not areas_str:
        return [str(f) for f in sorted(STATE_CODES.keys())]
    resolved = []
    for a in areas_str.split(","):
        a = a.strip()
        if a in ABBR_TO_FIPS:
            resolved.append(ABBR_TO_FIPS[a])
        elif a.isdigit():
            resolved.append(a)
        else:
            logger.warning("Unknown area '%s', skipping", a)
    return resolved


def _resolve_district_ids(engine, areas_str: Optional[str]) -> list:
    """Resolve --areas to district geographic_ids from DB."""
    state_fips_list = _resolve_state_fips(areas_str)
    with engine.connect() as conn:
        df = pd.read_sql(
            "SELECT DISTINCT geographic_id FROM target_overview "
            "WHERE geo_level = 'district'",
            conn,
        )
    all_geoids = df["geographic_id"].tolist()
    result = []
    for geoid in all_geoids:
        padded = str(geoid).zfill(4)
        sfips = padded[:-2]
        if sfips in state_fips_list:
            result.append(str(geoid))
    return sorted(result)


def _build_variable_entity_map(sim) -> dict:
    tbs = sim.tax_benefit_system
    mapping = {}
    for var_name in tbs.variables:
        var = tbs.get_variable(var_name)
        if var is not None:
            mapping[var_name] = var.entity.key
    count_entities = {
        "person_count": "person",
        "household_count": "household",
        "tax_unit_count": "tax_unit",
        "spm_unit_count": "spm_unit",
    }
    mapping.update(count_entities)
    return mapping


def _build_entity_rel(sim) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": sim.calculate("person_id", map_to="person").values,
            "household_id": sim.calculate(
                "household_id", map_to="person"
            ).values,
            "tax_unit_id": sim.calculate(
                "tax_unit_id", map_to="person"
            ).values,
            "spm_unit_id": sim.calculate(
                "spm_unit_id", map_to="person"
            ).values,
        }
    )


def validate_area(
    sim,
    targets_df: pd.DataFrame,
    engine,
    area_type: str,
    area_id: str,
    display_id: str,
    period: int,
    training_mask: np.ndarray,
    variable_entity_map: dict,
) -> list:
    entity_rel = _build_entity_rel(sim)
    household_ids = sim.calculate("household_id", map_to="household").values
    n_households = len(household_ids)

    hh_weight = sim.calculate(
        "household_weight",
        map_to="household",
        period=period,
    ).values.astype(np.float64)

    hh_vars_cache = {}
    person_vars_cache = {}

    training_arr = np.asarray(training_mask, dtype=bool)

    geo_level = "state" if area_type == "states" else "district"

    results = []
    for i, (idx, row) in enumerate(targets_df.iterrows()):
        variable = row["variable"]
        target_value = float(row["value"])
        stratum_id = int(row["stratum_id"])

        constraints = _get_stratum_constraints(engine, stratum_id)
        non_geo = [c for c in constraints if c["variable"] not in _GEO_VARS]

        needed_vars = set()
        needed_vars.add(variable)
        for c in non_geo:
            needed_vars.add(c["variable"])

        for vname in needed_vars:
            if vname not in hh_vars_cache:
                entity = variable_entity_map.get(vname)
                if entity == "household" or (
                    entity is None and not vname.endswith("_count")
                ):
                    try:
                        hh_vars_cache[vname] = sim.calculate(
                            vname,
                            map_to="household",
                            period=period,
                        ).values
                    except Exception:
                        pass
            if vname not in person_vars_cache:
                try:
                    person_vars_cache[vname] = sim.calculate(
                        vname,
                        map_to="person",
                        period=period,
                    ).values
                except Exception:
                    pass

        per_hh = _calculate_target_values_standalone(
            target_variable=variable,
            non_geo_constraints=non_geo,
            n_households=n_households,
            hh_vars=hh_vars_cache,
            person_vars=person_vars_cache,
            entity_rel=entity_rel,
            household_ids=household_ids,
            variable_entity_map=variable_entity_map,
        )

        sim_value = float(np.dot(per_hh, hh_weight))

        error = sim_value - target_value
        abs_error = abs(error)
        if target_value != 0:
            rel_error = error / target_value
            rel_abs_error = abs_error / abs(target_value)
        else:
            rel_error = float("inf") if error != 0 else 0.0
            rel_abs_error = float("inf") if abs_error != 0 else 0.0

        target_name = UnifiedMatrixBuilder._make_target_name(
            variable,
            constraints,
        )

        sanity_check, sanity_reason = _run_sanity_check(
            sim_value,
            variable,
            geo_level,
        )

        in_training = bool(training_arr[i])

        results.append(
            {
                "area_type": area_type,
                "area_id": display_id,
                "variable": variable,
                "target_name": target_name,
                "period": int(row["period"]),
                "target_value": target_value,
                "sim_value": sim_value,
                "error": error,
                "rel_error": rel_error,
                "abs_error": abs_error,
                "rel_abs_error": rel_abs_error,
                "sanity_check": sanity_check,
                "sanity_reason": sanity_reason,
                "in_training": in_training,
            }
        )

    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate staging .h5 files against "
        "calibration targets via sim.calculate()"
    )
    parser.add_argument(
        "--area-type",
        default="states",
        help="Comma-separated geo levels to validate: "
        "states, districts (default: states)",
    )
    parser.add_argument(
        "--areas",
        default=None,
        help="Comma-separated state abbreviations or FIPS "
        "(applies to all area types; all if omitted)",
    )
    parser.add_argument(
        "--hf-prefix",
        default=DEFAULT_HF_PREFIX,
        help="HuggingFace path prefix for .h5 files",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=2024,
        help="Tax year to validate (default: 2024)",
    )
    parser.add_argument(
        "--target-config",
        default=DEFAULT_TARGET_CONFIG,
        help="YAML config with exclude rules "
        "(default: target_config_full.yaml)",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Path to policy_data.db",
    )
    parser.add_argument(
        "--output",
        default="validation_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Run only structural sanity checks (fast, " "no database needed)",
    )
    return parser.parse_args(argv)


def _run_area_type(
    area_type,
    area_ids,
    level_targets,
    level_training,
    engine,
    args,
    Microsimulation,
):
    """Validate all areas for a single area_type.

    Loads one sim at a time to keep memory low.
    """
    results = []
    total_weighted_pop = 0.0
    current_h5 = None
    sim = None

    for area_id in area_ids:
        if area_type == "states":
            abbr = FIPS_TO_ABBR.get(area_id, area_id)
            h5_name = abbr
            display_id = abbr
        else:
            h5_name = _geoid_to_district_filename(area_id)
            display_id = h5_name

        h5_path = f"{args.hf_prefix}/{area_type}/{h5_name}.h5"

        if h5_path != current_h5:
            current_h5 = h5_path
            del sim
            gc.collect()
            logger.info("Loading sim from %s", h5_path)
            try:
                sim = Microsimulation(dataset=h5_path)
            except Exception as e:
                logger.error("Failed to load %s: %s", h5_path, e)
                sim = None

            if sim is not None and area_type == "states":
                person_weight = sim.calculate(
                    "person_weight",
                    map_to="person",
                    period=args.period,
                ).values.astype(np.float64)
                area_pop = float(person_weight.sum())
                total_weighted_pop += area_pop
                logger.info(
                    "  %s population: %,.0f",
                    display_id,
                    area_pop,
                )

        if sim is None:
            continue

        area_mask = (level_targets["geographic_id"] == area_id).values
        area_targets = level_targets[area_mask].reset_index(drop=True)
        area_training = level_training[area_mask]

        if len(area_targets) == 0:
            logger.warning("No targets for %s, skipping", display_id)
            continue

        logger.info(
            "Validating %d targets for %s",
            len(area_targets),
            display_id,
        )

        variable_entity_map = _build_variable_entity_map(sim)

        area_results = validate_area(
            sim=sim,
            targets_df=area_targets,
            engine=engine,
            area_type=area_type,
            area_id=area_id,
            display_id=display_id,
            period=args.period,
            training_mask=area_training,
            variable_entity_map=variable_entity_map,
        )
        results.extend(area_results)

        n_fail = sum(1 for r in area_results if r["sanity_check"] == "FAIL")
        logger.info(
            "  %s: %d results, %d sanity failures",
            display_id,
            len(area_results),
            n_fail,
        )

    if area_type == "states" and total_weighted_pop > 0:
        logger.info(
            "TOTAL WEIGHTED POPULATION: %,.0f (expect ~340M)",
            total_weighted_pop,
        )

    return results


def _run_sanity_only(args):
    """Run structural sanity checks on staging H5 files."""
    area_types = [t.strip() for t in args.area_type.split(",")]
    state_fips_list = _resolve_state_fips(args.areas)

    total_failures = 0

    for area_type in area_types:
        if area_type == "states":
            for fips in state_fips_list:
                abbr = FIPS_TO_ABBR.get(fips, fips)
                h5_url = f"{args.hf_prefix}/{area_type}/{abbr}.h5"
                logger.info("Sanity-checking %s", h5_url)

                if h5_url.startswith("hf://"):
                    from huggingface_hub import hf_hub_download
                    import tempfile

                    parts = h5_url[5:].split("/", 2)
                    repo = f"{parts[0]}/{parts[1]}"
                    path = parts[2]
                    local = hf_hub_download(
                        repo_id=repo,
                        filename=path,
                        repo_type="model",
                        token=os.environ.get("HUGGING_FACE_TOKEN"),
                    )
                else:
                    local = h5_url

                results = run_sanity_checks(local, args.period)
                n_fail = sum(1 for r in results if r["status"] == "FAIL")
                total_failures += n_fail

                for r in results:
                    if r["status"] != "PASS":
                        detail = f" — {r['detail']}" if r["detail"] else ""
                        logger.warning(
                            "  %s [%s] %s%s",
                            abbr,
                            r["status"],
                            r["check"],
                            detail,
                        )

                if n_fail == 0:
                    logger.info("  %s: all checks passed", abbr)
        else:
            logger.info(
                "Sanity-only mode for %s not yet implemented",
                area_type,
            )

    if total_failures > 0:
        logger.error("%d total sanity failures", total_failures)
    else:
        logger.info("All sanity checks passed")


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args(argv)
    logger.info("CLI args: %s", vars(args))

    if args.sanity_only:
        _run_sanity_only(args)
        return

    from policyengine_us import Microsimulation

    engine = create_engine(f"sqlite:///{args.db_path}")

    all_targets = _query_all_active_targets(engine, args.period)
    logger.info("Loaded %d active targets from DB", len(all_targets))

    exclude_config = load_target_config(args.target_config)
    exclude_rules = exclude_config.get("exclude", [])
    if exclude_rules:
        exc_mask = _match_rules(all_targets, exclude_rules)
        all_targets = all_targets[~exc_mask].reset_index(drop=True)
        logger.info("After exclusions: %d targets", len(all_targets))

    include_rules = exclude_config.get("include", [])
    if include_rules:
        inc_mask = _match_rules(all_targets, include_rules)
        all_targets = all_targets[inc_mask].reset_index(drop=True)
        logger.info("After inclusions: %d targets", len(all_targets))

    training_config = load_target_config(TRAINING_TARGET_CONFIG)
    training_include = training_config.get("include", [])
    if training_include:
        training_mask = np.asarray(
            _match_rules(all_targets, training_include),
            dtype=bool,
        )
    else:
        training_mask = np.ones(len(all_targets), dtype=bool)

    area_types = [t.strip() for t in args.area_type.split(",")]
    valid_types = {"states", "districts"}
    for t in area_types:
        if t not in valid_types:
            logger.error(
                "Unknown area-type '%s'. Use: %s",
                t,
                ", ".join(sorted(valid_types)),
            )
            return

    all_results = []

    for area_type in area_types:
        geo_level = "state" if area_type == "states" else "district"
        geo_mask = (all_targets["geo_level"] == geo_level).values
        level_targets = all_targets[geo_mask].reset_index(drop=True)
        level_training = training_mask[geo_mask]

        logger.info(
            "%d targets at geo_level=%s",
            len(level_targets),
            geo_level,
        )

        if area_type == "states":
            area_ids = _resolve_state_fips(args.areas)
        else:
            area_ids = _resolve_district_ids(engine, args.areas)

        logger.info(
            "%s: %d areas to validate",
            area_type,
            len(area_ids),
        )

        results = _run_area_type(
            area_type=area_type,
            area_ids=area_ids,
            level_targets=level_targets,
            level_training=level_training,
            engine=engine,
            args=args,
            Microsimulation=Microsimulation,
        )
        all_results.extend(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_results)

    logger.info("Wrote %d rows to %s", len(all_results), output_path)

    n_total_fail = sum(1 for r in all_results if r["sanity_check"] == "FAIL")
    if n_total_fail > 0:
        logger.warning(
            "%d SANITY FAILURES across all areas",
            n_total_fail,
        )


if __name__ == "__main__":
    main()
