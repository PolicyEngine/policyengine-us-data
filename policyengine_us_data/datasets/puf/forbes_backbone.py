"""Forbes-backed synthesis for the PUF's $100M+ aggregate record.

This module follows the public-model pattern used by PWBM and TPC:
append a fixed rich-list backbone, keep those records at unit weight,
and calibrate the resulting microdata back to published tax aggregates.

The current implementation is a two-stage donor model:

1. Use Forbes to identify the tax units most likely to populate the
   IRS $100M+ aggregate row.
2. Draw joint wealth-to-income regimes from top-tail SCF donors.
3. Use top-tail PUF donors only for tax-line detail that SCF does not
   observe directly, then calibrate back to the published aggregate row.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from filelock import FileLock
import numpy as np
import pandas as pd
import requests

from policyengine_us_data.datasets.puf import aggregate_record_utils as utils
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.census import (
    STATE_ABBREV_TO_FIPS,
    STATE_NAME_TO_FIPS,
)

logger = logging.getLogger(__name__)

FORBES_DEFAULT_SNAPSHOT_DATE = "2024-09-01"
FORBES_RTB_API_REF = "04e6948f0b3097de28f8af080e3a6ae577bd8956"
FORBES_TOP_400_SIZE = 400
FORBES_DEFAULT_REPLICATES = 10
FORBES_TOP_TAIL_SCF_YEAR = 2022
FORBES_TOP_TAIL_AGI_THRESHOLD = 100_000_000

FORBES_RTB_API_BASE_URL = (
    f"https://raw.githubusercontent.com/komed3/rtb-api/{FORBES_RTB_API_REF}"
)
FORBES_US_ALIASES_URL = f"{FORBES_RTB_API_BASE_URL}/api/filter/country/us"
FORBES_LIST_URL_TEMPLATE = f"{FORBES_RTB_API_BASE_URL}/api/list/rtb/{{snapshot_date}}"
FORBES_PROFILE_INFO_URL = f"{FORBES_RTB_API_BASE_URL}/api/profile/{{alias}}/info"
FORBES_PACKAGED_SNAPSHOT_NAME = (
    f"forbes_us_top_400_{FORBES_DEFAULT_SNAPSHOT_DATE}_{FORBES_RTB_API_REF[:12]}.json"
)
SCF_PACKAGED_DONOR_NAME = f"scf_forbes_donors_{FORBES_TOP_TAIL_SCF_YEAR}.json.gz"
FORBES_TOP_TAIL_METADATA_DEFAULTS = {
    "forbes_alias": "",
    "forbes_name": "",
    "forbes_snapshot_date": "",
    "forbes_marital_status": "",
    "forbes_rank": 0,
    "forbes_unit_id": -1,
    "forbes_replicate_id": -1,
    "forbes_age": 0,
    "forbes_children": 0,
    "forbes_state_fips": 0,
}
FORBES_STRING_METADATA_COLUMNS = {
    "forbes_alias",
    "forbes_name",
    "forbes_snapshot_date",
    "forbes_marital_status",
}

SCF_JOINT_INCOME_COLUMNS = (
    "wageinc",
    "kginc",
    "intdivinc",
    "bussefarminc",
    "ssretinc",
)
SCF_BALANCE_COLUMNS = ("houses", "mrthel", "ccbal", "edn_inst")
PUF_AGI_COMPONENTS = (
    "E00200",
    "P23250",
    "P22250",
    "E00300",
    "E00600",
    "E26270",
    "E00900",
    "E02100",
)
PUF_PENSION_COLUMNS = ("E01500", "E01700", "E02400")
ARTIFACT_NUMERIC_TOL = 1e-6
FORBES_DIAGNOSTIC_COMPONENTS = (
    ("agi", ("E00100",)),
    ("wages", ("E00200",)),
    ("capital_gains", ("P23250", "P22250")),
    ("interest_dividends", ("E00300", "E00400", "E00600")),
    ("business_farm", ("E26270", "E00900", "E02100")),
    ("pension_social_security", ("E01500", "E02400")),
    ("taxable_pensions", ("E01700",)),
    ("charitable_contributions", ("E19800",)),
    ("state_local_taxes", ("E18400",)),
)
SCF_DIAGNOSTIC_COMPONENTS = {
    "wages": ("employment_income",),
    "capital_gains": ("capital_gains",),
    "interest_dividends": ("interest_dividend_income",),
    "business_farm": ("business_farm_income",),
    "pension_social_security": ("pension_income",),
}

_SCF_FORBES_DONOR_CACHE: dict[int, pd.DataFrame] = {}


@dataclass(frozen=True)
class ForbesIndustryProfile:
    archetype: str
    agi_ratio: float


@dataclass(frozen=True)
class ForbesTopTailConfig:
    """Configuration for one reproducible Forbes top-tail build."""

    snapshot_date: str = FORBES_DEFAULT_SNAPSHOT_DATE
    source_ref: str = FORBES_RTB_API_REF
    replicate_count: int = FORBES_DEFAULT_REPLICATES
    scf_year: int = FORBES_TOP_TAIL_SCF_YEAR
    cache_path: Path | None = None
    force_refresh: bool = False

    def validate(self) -> None:
        if self.source_ref != FORBES_RTB_API_REF:
            raise ValueError(
                "ForbesTopTailConfig.source_ref must match the pinned rtb-api ref."
            )
        if self.replicate_count <= 0 or 100 % self.replicate_count != 0:
            raise ValueError(
                "Forbes replicate_count must be a positive divisor of 100."
            )

    @property
    def unit_weight_hundredths(self) -> int:
        return int(100 / self.replicate_count)


@dataclass(frozen=True)
class ForbesTopTailArtifact:
    """Staged top-tail build artifact before it is upserted into the PUF."""

    source_forbes: pd.DataFrame
    selected_forbes: pd.DataFrame
    scf_donors: pd.DataFrame
    scf_draws: pd.DataFrame
    puf_templates: pd.DataFrame
    puf_priors: pd.DataFrame
    synthetic: pd.DataFrame
    diagnostics: dict[str, int | float | str]


DEFAULT_PROFILE = ForbesIndustryProfile(archetype="mixed", agi_ratio=0.022)

FORBES_INDUSTRY_PROFILES = {
    "technology": ForbesIndustryProfile(archetype="ltcg", agi_ratio=0.018),
    "automotive": ForbesIndustryProfile(archetype="ltcg", agi_ratio=0.020),
    "finance-investments": ForbesIndustryProfile(
        archetype="partnership",
        agi_ratio=0.034,
    ),
    "real-estate": ForbesIndustryProfile(archetype="partnership", agi_ratio=0.028),
    "diversified": ForbesIndustryProfile(archetype="dividend", agi_ratio=0.020),
    "energy": ForbesIndustryProfile(archetype="dividend", agi_ratio=0.023),
    "fashion-retail": DEFAULT_PROFILE,
    "food-beverage": DEFAULT_PROFILE,
    "healthcare": DEFAULT_PROFILE,
    "construction-engineering": DEFAULT_PROFILE,
    "media-entertainment": DEFAULT_PROFILE,
    "sports": DEFAULT_PROFILE,
    "logistics": DEFAULT_PROFILE,
    "gambling-casinos": DEFAULT_PROFILE,
}


def build_forbes_top_tail_bucket(
    row: pd.Series,
    regular: pd.DataFrame,
    amount_columns: list[str],
    donor_scores: pd.Series,
    next_recid: int,
    rng: np.random.Generator,
    snapshot_date: str = FORBES_DEFAULT_SNAPSHOT_DATE,
    cache_path: Path | None = None,
    replicate_count: int = FORBES_DEFAULT_REPLICATES,
) -> pd.DataFrame | None:
    """Build a fixed-weight Forbes backbone for the $100M+ PUF bucket."""

    config = ForbesTopTailConfig(
        snapshot_date=snapshot_date,
        replicate_count=replicate_count,
        cache_path=cache_path,
    )
    try:
        return build_forbes_top_tail_artifact(
            row=row,
            regular=regular,
            amount_columns=amount_columns,
            donor_scores=donor_scores,
            next_recid=next_recid,
            rng=rng,
            config=config,
        ).synthetic
    except Exception as exc:  # pragma: no cover - pipeline fallback
        logger.warning("Forbes backbone unavailable; using donor synthesis: %s", exc)
        return None


def build_forbes_top_tail_artifact(
    row: pd.Series,
    regular: pd.DataFrame,
    amount_columns: list[str],
    donor_scores: pd.Series,
    next_recid: int,
    rng: np.random.Generator,
    config: ForbesTopTailConfig | None = None,
) -> ForbesTopTailArtifact:
    """Build the staged Forbes/SCF/PUF top-tail artifact."""

    config = config or ForbesTopTailConfig()
    config.validate()

    pop_weight, _, target_total_agi = utils._get_bucket_targets(row)
    target_n = int(round(pop_weight))
    if target_n <= 0:
        raise ValueError("The Forbes top-tail target has non-positive weight.")

    source_forbes = load_forbes_us_top_400(
        snapshot_date=config.snapshot_date,
        cache_path=config.cache_path,
        force_refresh=config.force_refresh,
    )
    scf_donors = load_scf_forbes_donor_pool(
        scf_year=config.scf_year,
        force_refresh=config.force_refresh,
    )

    selected_forbes = select_forbes_extreme_tail(
        forbes=source_forbes,
        target_n=target_n,
        scf_donors=scf_donors,
    )
    if len(selected_forbes) < target_n:
        raise ValueError(
            "Forbes backbone produced only "
            f"{len(selected_forbes)} eligible units for target {target_n}."
        )

    forbes_draws = expand_forbes_replicates(
        selected_forbes=selected_forbes,
        replicate_count=config.replicate_count,
    )
    scf_draws = sample_scf_joint_profiles(
        forbes_draws=forbes_draws,
        scf_donors=scf_donors,
        rng=rng,
    )

    donor_bucket = utils._get_donor_bucket(regular, 999999)
    donor_templates = sample_puf_detail_templates(
        forbes_draws=scf_draws,
        donor_bucket=donor_bucket,
        donor_scores=donor_scores,
        rng=rng,
    )
    donor_templates = utils._coerce_amount_columns(donor_templates, amount_columns)

    puf_priors = donor_templates.copy()
    apply_forbes_joint_amount_bases(
        selected=puf_priors,
        forbes_draws=scf_draws,
        donor_templates=donor_templates,
    )

    synthetic = donor_templates.copy()
    synthetic["RECID"] = np.arange(
        next_recid,
        next_recid + len(scf_draws),
        dtype=int,
    )
    synthetic["S006"] = config.unit_weight_hundredths

    utils._apply_structural_templates(synthetic, donor_templates)
    apply_forbes_structural_overrides(synthetic, scf_draws)

    synthetic_weights = np.full(
        len(scf_draws),
        1.0 / config.replicate_count,
        dtype=float,
    )
    utils._calibrate_amount_columns(
        synthetic=synthetic,
        selected=puf_priors,
        row=row,
        recid=999999,
        pop_weight=pop_weight,
        target_total_agi=target_total_agi,
        amount_columns=amount_columns,
        synthetic_weights=synthetic_weights,
    )
    synthetic["S006"] = config.unit_weight_hundredths

    artifact = ForbesTopTailArtifact(
        source_forbes=source_forbes,
        selected_forbes=selected_forbes,
        scf_donors=scf_donors,
        scf_draws=scf_draws,
        puf_templates=donor_templates,
        puf_priors=puf_priors,
        synthetic=synthetic,
        diagnostics=build_forbes_top_tail_diagnostics(
            config=config,
            pop_weight=pop_weight,
            target_total_agi=target_total_agi,
            source_forbes=source_forbes,
            selected_forbes=selected_forbes,
            scf_donors=scf_donors,
            scf_draws=scf_draws,
            synthetic=synthetic,
        ),
    )
    validate_forbes_top_tail_artifact(
        artifact=artifact,
        config=config,
        row=row,
        pop_weight=pop_weight,
        amount_columns=amount_columns,
    )
    return artifact


def build_forbes_top_tail_diagnostics(
    config: ForbesTopTailConfig,
    pop_weight: float,
    target_total_agi: float,
    source_forbes: pd.DataFrame,
    selected_forbes: pd.DataFrame,
    scf_donors: pd.DataFrame,
    scf_draws: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> dict[str, int | float | str]:
    """Summarize one staged top-tail build for validation and logging."""

    return {
        "snapshot_date": config.snapshot_date,
        "source_ref": config.source_ref,
        "replicate_count": config.replicate_count,
        "target_units": int(round(pop_weight)),
        "source_forbes_rows": len(source_forbes),
        "selected_forbes_units": len(selected_forbes),
        "scf_donor_rows": len(scf_donors),
        "scf_draw_rows": len(scf_draws),
        "synthetic_rows": len(synthetic),
        "synthetic_weight": float((synthetic["S006"] / 100).sum()),
        "target_total_agi": float(target_total_agi),
    }


def build_forbes_top_tail_diagnostic_tables(
    artifact: ForbesTopTailArtifact,
    row: pd.Series,
    amount_columns: list[str],
) -> dict[str, pd.DataFrame]:
    """Return PR/CI-ready diagnostic tables for one Forbes top-tail artifact."""

    pop_weight, _, target_total_agi = utils._get_bucket_targets(row)
    weights = artifact.synthetic["S006"].to_numpy(dtype=float) / 100
    calibration = _build_calibration_diagnostics(
        synthetic=artifact.synthetic,
        row=row,
        amount_columns=amount_columns,
        pop_weight=pop_weight,
        weights=weights,
    )
    composition = _build_composition_diagnostics(
        artifact=artifact,
        row=row,
        pop_weight=pop_weight,
        weights=weights,
    )
    selection = _build_selection_diagnostics(artifact.selected_forbes)
    draws = _build_draw_diagnostics(artifact.scf_draws, weights)

    max_abs_error = float(calibration["absolute_error"].abs().max())
    rel_error = calibration["relative_error"].replace([np.inf, -np.inf], np.nan)
    max_rel_error = rel_error.abs().max(skipna=True)
    summary = pd.DataFrame(
        [
            {
                **artifact.diagnostics,
                "synthetic_total_agi": _weighted_columns_total(
                    artifact.synthetic,
                    ("E00100",),
                    weights,
                ),
                "target_total_agi": float(target_total_agi),
                "max_calibration_abs_error": max_abs_error,
                "max_calibration_rel_error": (
                    float(max_rel_error) if pd.notna(max_rel_error) else np.nan
                ),
                "mean_calibration_loss": float(
                    calibration["squared_relative_error"].mean()
                ),
                "total_calibration_loss": float(
                    calibration["squared_relative_error"].sum()
                ),
                "selected_networth_min_millions": float(
                    pd.to_numeric(
                        artifact.selected_forbes["networth_millions"],
                        errors="coerce",
                    ).min()
                ),
                "selected_networth_median_millions": float(
                    pd.to_numeric(
                        artifact.selected_forbes["networth_millions"],
                        errors="coerce",
                    ).median()
                ),
                "selected_networth_max_millions": float(
                    pd.to_numeric(
                        artifact.selected_forbes["networth_millions"],
                        errors="coerce",
                    ).max()
                ),
                "top_selected_aliases": ", ".join(
                    artifact.selected_forbes["alias"].head(5).astype(str)
                ),
            }
        ]
    )

    return {
        "summary": summary,
        "calibration": calibration,
        "composition": composition,
        "selection": selection,
        "draws": draws,
    }


def format_forbes_top_tail_diagnostics(
    tables: dict[str, pd.DataFrame],
    max_rows: int = 12,
) -> str:
    """Format diagnostic tables as compact Markdown with ASCII bar visuals."""

    summary = tables["summary"].iloc[0]
    source_ref = str(summary["source_ref"])
    lines = [
        "# Forbes top-tail diagnostics",
        (
            f"Snapshot: {summary['snapshot_date']} @ {source_ref[:12]} | "
            f"Rows: {int(summary['synthetic_rows']):,} synthetic from "
            f"{int(summary['selected_forbes_units']):,} Forbes units x "
            f"{int(summary['replicate_count']):,} draws"
        ),
        (
            "Calibration: max absolute error "
            f"{_format_dollars(summary['max_calibration_abs_error'])}; "
            "max relative error "
            f"{_format_percent(summary['max_calibration_rel_error'])}; "
            f"loss={summary['total_calibration_loss']:.3g}"
        ),
        "",
        "## Composition",
    ]

    composition = tables["composition"].copy()
    composition = composition.sort_values(
        "synthetic_total",
        key=lambda values: values.abs(),
        ascending=False,
    ).head(max_rows)
    max_share = composition["share_of_agi"].abs().max()
    for row in composition.itertuples(index=False):
        lines.append(
            f"{row.component:<26} "
            f"{_format_bar(row.share_of_agi, max_share):<30} "
            f"{_format_percent(row.share_of_agi):>7} "
            f"target={_format_dollars(row.target_total):>8} "
            f"synthetic={_format_dollars(row.synthetic_total):>8}"
        )

    lines.extend(["", "## Largest calibration errors"])
    calibration = tables["calibration"].copy()
    calibration = calibration.sort_values(
        "absolute_error",
        key=lambda values: values.abs(),
        ascending=False,
    ).head(max_rows)
    for row in calibration.itertuples(index=False):
        lines.append(
            f"{row.column:<10} "
            f"target={_format_dollars(row.target_total):>8} "
            f"synthetic={_format_dollars(row.synthetic_total):>8} "
            f"error={_format_dollars(row.absolute_error):>8}"
        )

    lines.extend(["", "## Selected Forbes units"])
    selection = tables["selection"].head(max_rows)
    for row in selection.itertuples(index=False):
        tail_probability = getattr(row, "scf_tail_probability", np.nan)
        expected_agi = getattr(row, "scf_expected_agi", np.nan)
        lines.append(
            f"{int(row.rank):>3} {str(row.alias):<28} "
            f"net_worth={_format_dollars(row.networth_millions * 1_000_000):>8} "
            f"tail_prob={_format_percent(tail_probability):>7} "
            f"scf_agi={_format_dollars(expected_agi):>8}"
        )

    return "\n".join(lines)


def validate_forbes_top_tail_artifact(
    artifact: ForbesTopTailArtifact,
    config: ForbesTopTailConfig,
    row: pd.Series,
    pop_weight: float,
    amount_columns: list[str],
) -> None:
    """Validate the staged artifact before it is upserted into the PUF."""

    config.validate()
    expected_units = int(round(pop_weight))
    expected_draws = expected_units * config.replicate_count

    if len(artifact.selected_forbes) != expected_units:
        raise ValueError(
            "Forbes artifact selected "
            f"{len(artifact.selected_forbes)} units for target {expected_units}."
        )
    for name, frame in {
        "scf_draws": artifact.scf_draws,
        "puf_templates": artifact.puf_templates,
        "puf_priors": artifact.puf_priors,
        "synthetic": artifact.synthetic,
    }.items():
        if len(frame) != expected_draws:
            raise ValueError(
                f"Forbes artifact {name} has {len(frame)} rows; "
                f"expected {expected_draws}."
            )

    synthetic_weight = float((artifact.synthetic["S006"] / 100).sum())
    if abs(synthetic_weight - expected_units) > ARTIFACT_NUMERIC_TOL:
        raise ValueError(
            "Forbes synthetic weights sum to "
            f"{synthetic_weight}; expected {expected_units}."
        )
    if not artifact.synthetic["S006"].eq(config.unit_weight_hundredths).all():
        raise ValueError("Forbes synthetic replicate weights are not uniform.")

    required_columns = {"RECID", "S006", "E00100", *amount_columns}
    missing_columns = required_columns.difference(artifact.synthetic.columns)
    if missing_columns:
        raise ValueError(
            f"Forbes synthetic output is missing columns: {sorted(missing_columns)}."
        )

    numeric_columns = [column for column in required_columns if column != "RECID"]
    values = artifact.synthetic[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Forbes synthetic output contains non-finite numeric values.")

    if artifact.synthetic["RECID"].duplicated().any():
        raise ValueError("Forbes synthetic output contains duplicate RECIDs.")

    weights = artifact.synthetic["S006"].to_numpy(dtype=float) / 100
    for column in amount_columns:
        target_total = pop_weight * float(row.get(column, 0.0))
        actual_total = float(
            np.dot(artifact.synthetic[column].to_numpy(dtype=float), weights)
        )
        tolerance = max(ARTIFACT_NUMERIC_TOL, abs(target_total) * 1e-9)
        if abs(actual_total - target_total) > tolerance:
            raise ValueError(
                "Forbes calibrated total mismatch for "
                f"{column}: {actual_total} != {target_total}."
            )


def load_forbes_us_top_400(
    snapshot_date: str = FORBES_DEFAULT_SNAPSHOT_DATE,
    cache_path: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load a cached Forbes-style US top-400 backbone, refreshing if needed."""

    if not force_refresh and cache_path is None:
        packaged_records = read_packaged_forbes_records(snapshot_date)
        if packaged_records is not None:
            return pd.DataFrame(packaged_records)

    cache_path = cache_path or forbes_cache_path(snapshot_date)

    with FileLock(f"{cache_path}.lock", timeout=600):
        cached_records = None
        if cache_path.exists():
            try:
                cached_records = read_cached_forbes_records(cache_path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Ignoring unreadable Forbes cache at %s: %s", cache_path, exc
                )
                cached_records = None
            else:
                if not force_refresh and cache_is_fresh(cache_path):
                    return pd.DataFrame(cached_records)

        try:
            records = fetch_forbes_us_top_400(snapshot_date=snapshot_date)
        except Exception:
            if cached_records is not None:
                logger.warning(
                    "Using stale cached Forbes backbone from %s after refresh failure.",
                    cache_path,
                )
                return pd.DataFrame(cached_records)
            raise

        write_cached_forbes_records(records, cache_path)
        return pd.DataFrame(records)


def load_scf_forbes_donor_pool(
    scf_year: int = FORBES_TOP_TAIL_SCF_YEAR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load and prepare the SCF donor pool used for Forbes top-tail draws."""

    cached = _SCF_FORBES_DONOR_CACHE.get(scf_year)
    if cached is not None and not force_refresh:
        return cached.copy()

    if not force_refresh:
        packaged = read_packaged_scf_forbes_donor_pool(scf_year)
        if packaged is not None:
            _SCF_FORBES_DONOR_CACHE[scf_year] = packaged
            return packaged.copy()

    if scf_year != 2022:
        raise ValueError(f"Unsupported SCF year for Forbes backbone: {scf_year}")

    from policyengine_us_data.datasets.scf.fed_scf import SummarizedFedSCF_2022

    raw_scf = SummarizedFedSCF_2022().load()
    prepared = prepare_scf_forbes_donor_pool(raw_scf)
    _SCF_FORBES_DONOR_CACHE[scf_year] = prepared
    return prepared.copy()


def read_packaged_scf_forbes_donor_pool(
    scf_year: int,
) -> pd.DataFrame | None:
    """Read the committed normalized SCF donor pool if it matches the request."""

    if scf_year != FORBES_TOP_TAIL_SCF_YEAR:
        return None

    donor_path = packaged_scf_forbes_donor_path()
    if not donor_path.exists():
        return None

    donor = pd.read_json(donor_path, orient="split", compression="gzip")
    if "is_married" in donor.columns:
        donor["is_married"] = donor["is_married"].astype(bool)
    return donor


def packaged_scf_forbes_donor_path() -> Path:
    """Return the committed normalized SCF donor artifact path."""

    return Path(__file__).with_name(SCF_PACKAGED_DONOR_NAME)


def prepare_scf_forbes_donor_pool(raw_scf: pd.DataFrame) -> pd.DataFrame:
    """Construct a top-tail SCF donor pool with joint wealth-income ratios."""

    donor = pd.DataFrame(
        {
            "age": _numeric_series(raw_scf, "age"),
            "is_married": _numeric_series(raw_scf, "married").eq(1),
            "wgt": _numeric_series(raw_scf, "household_weight", "wgt").clip(lower=0.0),
            "net_worth": _numeric_series(raw_scf, "networth", "net_worth"),
            "wageinc": _numeric_series(raw_scf, "wageinc"),
            "kginc": _numeric_series(raw_scf, "kginc"),
            "intdivinc": _numeric_series(raw_scf, "intdivinc"),
            "bussefarminc": _numeric_series(raw_scf, "bussefarminc"),
            "ssretinc": _numeric_series(raw_scf, "ssretinc"),
            "houses": _numeric_series(raw_scf, "houses"),
            "mrthel": _numeric_series(raw_scf, "mrthel"),
            "ccbal": _numeric_series(raw_scf, "ccbal"),
            "edn_inst": _numeric_series(raw_scf, "edn_inst"),
        }
    )
    donor = donor.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    donor = donor[(donor["net_worth"] > 0) & (donor["wgt"] > 0)].copy()
    if donor.empty:
        raise ValueError("No eligible SCF donors with positive net worth.")

    tail_cutoff = _weighted_quantile(
        donor["net_worth"].to_numpy(dtype=float),
        donor["wgt"].to_numpy(dtype=float),
        0.90,
    )
    tail = donor[donor["net_worth"] >= tail_cutoff].copy()
    if len(tail) < min(100, len(donor)):
        tail = donor.copy()

    wealth = np.maximum(tail["net_worth"].to_numpy(dtype=float), 1.0)
    for column in SCF_JOINT_INCOME_COLUMNS + SCF_BALANCE_COLUMNS:
        tail[f"{column}_ratio"] = tail[column].to_numpy(dtype=float) / wealth

    tail["major_income_total"] = (
        tail["wageinc"] + tail["kginc"] + tail["intdivinc"] + tail["bussefarminc"]
    )
    tail["wealth_score"] = np.log1p(tail["net_worth"].to_numpy(dtype=float))
    tail["archetype"] = classify_scf_archetypes(tail)
    return tail.reset_index(drop=True)


def cache_is_fresh(cache_path: Path) -> bool:
    """Return whether an immutable Forbes cache exists."""

    return cache_path.exists()


def read_cached_forbes_records(cache_path: Path) -> list[dict]:
    """Read cached Forbes records from disk."""

    return json.loads(cache_path.read_text(encoding="utf-8"))


def read_packaged_forbes_records(snapshot_date: str) -> list[dict] | None:
    """Read the committed normalized Forbes snapshot if it matches the request."""

    if snapshot_date != FORBES_DEFAULT_SNAPSHOT_DATE:
        return None

    snapshot_path = packaged_forbes_snapshot_path()
    if not snapshot_path.exists():
        return None

    return read_cached_forbes_records(snapshot_path)


def packaged_forbes_snapshot_path() -> Path:
    """Return the committed normalized Forbes snapshot path."""

    return Path(__file__).with_name(FORBES_PACKAGED_SNAPSHOT_NAME)


def forbes_cache_path(snapshot_date: str) -> Path:
    """Return the cache path for one dated Forbes snapshot."""

    ref = FORBES_RTB_API_REF[:12]
    return STORAGE_FOLDER / f"forbes_us_top_400_{snapshot_date}_{ref}.json"


def write_cached_forbes_records(records: list[dict], cache_path: Path) -> None:
    """Write the Forbes cache atomically so partial writes do not corrupt it."""

    temp_path = cache_path.with_name(f".{cache_path.name}.tmp")
    temp_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temp_path.replace(cache_path)


def fetch_forbes_us_top_400(
    session: requests.Session | None = None,
    snapshot_date: str = FORBES_DEFAULT_SNAPSHOT_DATE,
) -> list[dict]:
    """Fetch and normalize the US top 400 richest people from the public API."""

    owns_session = session is None
    session = session or requests.Session()
    try:
        us_aliases = set(fetch_json(session, FORBES_US_ALIASES_URL))
        latest_payload = fetch_json(
            session,
            FORBES_LIST_URL_TEMPLATE.format(snapshot_date=snapshot_date),
        )
        latest_rows = latest_payload["list"]
        latest_us = [
            row
            for row in latest_rows
            if row.get("uri") in us_aliases or row.get("citizenship") == "us"
        ]
        latest_us = sorted(
            latest_us,
            key=lambda row: (-float(row.get("networth", 0.0)), row.get("rank", 10**9)),
        )[:FORBES_TOP_400_SIZE]

        info_by_alias = fetch_profile_info_batch(
            aliases=[row["uri"] for row in latest_us],
        )
        snapshot_date = latest_payload.get("date")

        records = []
        for latest in latest_us:
            alias = latest["uri"]
            info = info_by_alias.get(alias, {})
            self_made = info.get("selfMade") or {}
            residence = info.get("residence") or {}
            industries = latest.get("industry") or info.get("industry") or []
            sources = latest.get("source") or info.get("source") or []
            records.append(
                {
                    "alias": alias,
                    "rank": latest.get("rank"),
                    "snapshot_date": snapshot_date,
                    "name": latest.get("name") or info.get("name"),
                    "age": latest.get("age"),
                    "birth_date": info.get("birthDate"),
                    "networth_millions": float(latest.get("networth", 0.0)),
                    "citizenship": latest.get("citizenship") or info.get("citizenship"),
                    "industry": "|".join(industries),
                    "source": "|".join(sources),
                    "residence_country": residence.get("country"),
                    "residence_state": residence.get("state"),
                    "marital_status": info.get("maritalStatus"),
                    "children": info.get("children"),
                    "deceased": bool(info.get("deceased", False)),
                    "family": bool(info.get("family", False)),
                    "self_made": bool(self_made.get("_is", False)),
                    "self_made_type": self_made.get("type"),
                }
            )
        return records
    finally:
        if owns_session:
            session.close()


def fetch_profile_info_batch(
    aliases: list[str],
    max_workers: int = 16,
) -> dict[str, dict]:
    """Fetch profile metadata for a batch of Forbes aliases."""

    results: dict[str, dict] = {}
    failures: list[tuple[str, Exception]] = []

    def load(alias: str) -> tuple[str, dict]:
        url = FORBES_PROFILE_INFO_URL.format(alias=alias)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return alias, response.json()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load, alias): alias for alias in aliases}
        for future in as_completed(futures):
            alias = futures[future]
            try:
                loaded_alias, payload = future.result()
            except Exception as exc:  # pragma: no cover - network/cache fallback
                failures.append((alias, exc))
                continue
            results[loaded_alias] = payload

    if failures:
        failed_aliases = ", ".join(alias for alias, _ in failures[:5])
        if len(failures) > 5:
            failed_aliases += f", ... ({len(failures)} total)"
        raise RuntimeError(
            f"Failed to fetch Forbes profile metadata for {failed_aliases}."
        ) from failures[0][1]
    return results


def fetch_json(session: requests.Session, url: str) -> dict | list:
    """Fetch JSON from the public Forbes backbone source."""

    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def select_forbes_extreme_tail(
    forbes: pd.DataFrame,
    target_n: int,
    scf_donors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Choose the Forbes records most likely to populate the $100M+ AGI bucket."""

    prepared = prepare_forbes_selection_frame(forbes)
    prepared["is_eligible"] = (
        ~prepared["deceased"].fillna(False).astype(bool)
        & (
            prepared["citizenship"].fillna("").eq("us")
            | prepared["residence_country"].fillna("").eq("us")
        )
        & (prepared["networth_dollars"] > 0)
    )
    prepared = prepared[prepared["is_eligible"]].copy()
    if prepared.empty:
        return prepared

    if scf_donors is not None and not scf_donors.empty:
        prepared = score_forbes_selection_with_scf(prepared, scf_donors)
        prepared = prepared.sort_values(
            [
                "scf_tail_probability",
                "scf_expected_agi",
                "networth_dollars",
                "rank",
            ],
            ascending=[False, False, False, True],
        ).head(target_n)
    else:
        prepared = prepared.sort_values(
            ["estimated_agi", "networth_dollars", "rank"],
            ascending=[False, False, True],
        ).head(target_n)
    return prepared.reset_index(drop=True)


def prepare_forbes_selection_frame(forbes: pd.DataFrame) -> pd.DataFrame:
    """Normalize Forbes records for selection and SCF matching."""

    prepared = forbes.copy()
    prepared["networth_dollars"] = (
        pd.to_numeric(prepared["networth_millions"], errors="coerce").fillna(0.0)
        * 1_000_000
    )
    prepared["industry_key"] = prepared["industry"].fillna("").str.split("|").str[0]
    profile_keys = prepared["industry_key"].where(
        prepared["industry_key"].isin(FORBES_INDUSTRY_PROFILES),
        "default",
    )
    profiles = profile_keys.map(
        lambda key: FORBES_INDUSTRY_PROFILES.get(key, DEFAULT_PROFILE)
    )
    prepared["archetype"] = [profile.archetype for profile in profiles]
    prepared["agi_ratio"] = [profile.agi_ratio for profile in profiles]
    prepared["age"] = pd.to_numeric(prepared["age"], errors="coerce").fillna(
        pd.to_numeric(prepared["age"], errors="coerce").median()
    )
    prepared["is_married"] = (
        prepared["marital_status"]
        .fillna("")
        .str.lower()
        .str.contains("married", regex=False)
    )
    prepared["children"] = pd.to_numeric(prepared["children"], errors="coerce").fillna(
        0.0
    )
    prepared["self_made_flag"] = prepared["self_made"].fillna(False).astype(bool)

    self_made_scale = np.where(
        prepared["self_made_flag"].to_numpy(),
        1.05,
        0.95,
    )
    prepared["estimated_agi"] = (
        prepared["networth_dollars"].to_numpy(dtype=float)
        * prepared["agi_ratio"].to_numpy(dtype=float)
        * self_made_scale
    )
    return prepared


def score_forbes_selection_with_scf(
    prepared_forbes: pd.DataFrame,
    scf_donors: pd.DataFrame,
) -> pd.DataFrame:
    """Score Forbes units for top-tail inclusion using the SCF donor model."""

    tail_probabilities = []
    expected_agi = []
    for row in prepared_forbes.itertuples(index=False):
        candidates = scf_candidates_for_receiver(scf_donors, row)
        probabilities = scf_match_probabilities(candidates, row)
        agi_values = scf_implied_agi_values(candidates, row)
        tail_probabilities.append(
            float(probabilities[agi_values >= FORBES_TOP_TAIL_AGI_THRESHOLD].sum())
        )
        expected_agi.append(float(np.dot(probabilities, agi_values)))

    scored = prepared_forbes.copy()
    scored["scf_tail_probability"] = tail_probabilities
    scored["scf_expected_agi"] = expected_agi
    scored["estimated_agi"] = np.maximum(
        scored["scf_expected_agi"].to_numpy(dtype=float),
        1.0,
    )
    return scored


def expand_forbes_replicates(
    selected_forbes: pd.DataFrame,
    replicate_count: int,
) -> pd.DataFrame:
    """Expand each Forbes unit into multiple uncertainty draws."""

    if selected_forbes.empty:
        return selected_forbes.copy()

    repeated = np.repeat(selected_forbes.index.to_numpy(), replicate_count)
    expanded = selected_forbes.loc[repeated].reset_index(drop=True).copy()
    expanded["forbes_unit_id"] = np.repeat(
        np.arange(len(selected_forbes), dtype=int),
        replicate_count,
    )
    expanded["replicate_id"] = np.tile(
        np.arange(replicate_count, dtype=int),
        len(selected_forbes),
    )
    expanded["networth_dollars"] = (
        pd.to_numeric(expanded["networth_millions"], errors="coerce").fillna(0.0)
        * 1_000_000
    )
    return expanded


def sample_scf_joint_profiles(
    forbes_draws: pd.DataFrame,
    scf_donors: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw joint wealth-income vectors from the SCF donor pool."""

    if forbes_draws.empty:
        return forbes_draws.copy()
    if scf_donors.empty:
        raise ValueError("SCF donor pool is empty.")

    drawn_rows: list[dict] = []
    for row in forbes_draws.itertuples(index=False):
        candidates = scf_candidates_for_receiver(scf_donors, row)
        probabilities = scf_match_probabilities(candidates, row)
        component_values = scf_implied_component_values(candidates, row)
        selected_position = rng.choice(len(candidates), p=probabilities)
        selected_idx = candidates.index.to_numpy()[selected_position]
        donor = candidates.loc[selected_idx]

        drawn_rows.append(
            {
                **row._asdict(),
                "scf_donor_index": int(selected_idx),
                "employment_income": float(
                    component_values["employment_income"][selected_position]
                ),
                "capital_gains": float(
                    component_values["capital_gains"][selected_position]
                ),
                "interest_dividend_income": float(
                    component_values["interest_dividend_income"][selected_position]
                ),
                "business_farm_income": float(
                    component_values["business_farm_income"][selected_position]
                ),
                "pension_income": float(
                    component_values["pension_income"][selected_position]
                ),
                "real_estate_share": float(donor["houses_ratio"]),
                "mortgage_share": float(donor["mrthel_ratio"]),
                "estimated_agi": float(
                    max(component_values["agi"][selected_position], 1.0)
                ),
            }
        )

    return pd.DataFrame(drawn_rows)


def scf_candidates_for_receiver(
    scf_donors: pd.DataFrame,
    receiver,
) -> pd.DataFrame:
    """Return the SCF donor neighborhood for one Forbes receiver."""

    candidates = scf_donors[
        scf_donors["archetype"] == getattr(receiver, "archetype", "")
    ]
    if len(candidates) < 20:
        return scf_donors
    return candidates


def scf_implied_component_values(
    candidates: pd.DataFrame,
    receiver,
) -> dict[str, np.ndarray]:
    """Scale SCF donor ratios up to one Forbes receiver's wealth level."""

    networth = float(getattr(receiver, "networth_dollars", 0.0))
    employment_income = np.maximum(
        0.0,
        networth * candidates["wageinc_ratio"].to_numpy(dtype=float),
    )
    capital_gains = networth * candidates["kginc_ratio"].to_numpy(dtype=float)
    interest_dividend_income = np.maximum(
        0.0,
        networth * candidates["intdivinc_ratio"].to_numpy(dtype=float),
    )
    business_farm_income = networth * candidates["bussefarminc_ratio"].to_numpy(
        dtype=float
    )
    pension_income = np.maximum(
        0.0,
        networth * candidates["ssretinc_ratio"].to_numpy(dtype=float),
    )
    agi = np.maximum(
        employment_income
        + capital_gains
        + interest_dividend_income
        + business_farm_income
        + 0.5 * pension_income,
        0.0,
    )
    return {
        "employment_income": employment_income,
        "capital_gains": capital_gains,
        "interest_dividend_income": interest_dividend_income,
        "business_farm_income": business_farm_income,
        "pension_income": pension_income,
        "agi": agi,
    }


def scf_implied_agi_values(
    candidates: pd.DataFrame,
    receiver,
) -> np.ndarray:
    """Return the implied positive AGI values for one receiver over SCF donors."""

    return scf_implied_component_values(candidates, receiver)["agi"]


def scf_match_probabilities(
    candidates: pd.DataFrame,
    receiver,
) -> np.ndarray:
    """Create SCF donor probabilities for one Forbes receiver draw."""

    base = candidates["wgt"].to_numpy(dtype=float)
    if not np.isfinite(base).all() or base.sum() <= 0:
        base = np.ones(len(candidates), dtype=float)

    wealth_tilt = np.exp(
        0.45
        * (
            candidates["wealth_score"].to_numpy(dtype=float)
            - candidates["wealth_score"].mean()
        )
    )
    age = float(getattr(receiver, "age", np.nan))
    if np.isfinite(age):
        age_diff = np.abs(candidates["age"].to_numpy(dtype=float) - age)
        age_mass = np.exp(-age_diff / 18.0)
    else:
        age_mass = np.ones(len(candidates), dtype=float)

    marital_target = bool(getattr(receiver, "is_married", False))
    marital_mass = np.where(
        candidates["is_married"].to_numpy(dtype=bool) == marital_target,
        1.35,
        0.80,
    )

    industry_key = getattr(receiver, "industry_key", "")
    if industry_key == "real-estate":
        industry_mass = 1.0 + np.clip(
            candidates["houses_ratio"].to_numpy(dtype=float),
            0.0,
            2.0,
        )
    elif industry_key == "finance-investments":
        industry_mass = 1.0 + np.clip(
            np.abs(candidates["kginc_ratio"].to_numpy(dtype=float))
            + np.abs(candidates["intdivinc_ratio"].to_numpy(dtype=float)),
            0.0,
            2.5,
        )
    else:
        industry_mass = np.ones(len(candidates), dtype=float)

    self_made = bool(getattr(receiver, "self_made_flag", False))
    if self_made:
        self_made_mass = 1.0 + np.clip(
            np.abs(candidates["wageinc_ratio"].to_numpy(dtype=float))
            + np.abs(candidates["kginc_ratio"].to_numpy(dtype=float)),
            0.0,
            2.0,
        )
    else:
        self_made_mass = 1.0 + np.clip(
            np.abs(candidates["intdivinc_ratio"].to_numpy(dtype=float)),
            0.0,
            1.5,
        )

    probabilities = (
        base * wealth_tilt * age_mass * marital_mass * industry_mass * self_made_mass
    )
    if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
        probabilities = np.ones(len(candidates), dtype=float)
    return probabilities / probabilities.sum()


def sample_puf_detail_templates(
    forbes_draws: pd.DataFrame,
    donor_bucket: pd.DataFrame,
    donor_scores: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample PUF donors to fill detailed tax lines not directly observed in SCF."""

    donor_pool = donor_bucket.copy()
    donor_pool["forbes_archetype"] = classify_donor_archetypes(donor_pool)
    donor_pool["puf_capital_gains"] = donor_pool.get("P23250", 0).to_numpy(
        dtype=float
    ) + donor_pool.get("P22250", 0).to_numpy(dtype=float)
    donor_pool["puf_investment"] = (
        donor_pool.get("E00300", 0).to_numpy(dtype=float)
        + donor_pool.get("E00400", 0).to_numpy(dtype=float)
        + donor_pool.get("E00600", 0).to_numpy(dtype=float)
    )
    donor_pool["puf_business"] = (
        donor_pool.get("E26270", 0).to_numpy(dtype=float)
        + donor_pool.get("E00900", 0).to_numpy(dtype=float)
        + donor_pool.get("E02100", 0).to_numpy(dtype=float)
    )

    matched_rows: list[pd.Series] = []
    for row in forbes_draws.itertuples(index=False):
        candidates = donor_pool[donor_pool["forbes_archetype"] == row.archetype]
        if len(candidates) < 20:
            candidates = donor_pool

        probabilities = puf_template_match_probabilities(
            candidates=candidates,
            donor_scores=donor_scores,
            receiver=row,
        )
        selected_idx = rng.choice(candidates.index.to_numpy(), p=probabilities)
        matched_rows.append(donor_bucket.loc[selected_idx].copy())

    return pd.DataFrame(matched_rows).reset_index(drop=True)


def puf_template_match_probabilities(
    candidates: pd.DataFrame,
    donor_scores: pd.Series,
    receiver,
) -> np.ndarray:
    """Create PUF donor probabilities for one SCF-conditioned Forbes draw."""

    score_mass = donor_scores.loc[candidates.index].to_numpy(dtype=float)
    score_mass = np.clip(score_mass, 1e-6, None) ** 8

    candidate_scale = np.maximum(
        np.abs(candidates["E00100"].to_numpy(dtype=float)), 1.0
    )
    candidate_components = np.column_stack(
        [
            candidates["E00200"].to_numpy(dtype=float),
            candidates["puf_capital_gains"].to_numpy(dtype=float),
            candidates["puf_investment"].to_numpy(dtype=float),
            candidates["puf_business"].to_numpy(dtype=float),
        ]
    )
    candidate_ratio = np.arcsinh(candidate_components / candidate_scale[:, None])

    receiver_scale = max(abs(float(receiver.estimated_agi)), 1.0)
    receiver_ratio = np.arcsinh(
        np.array(
            [
                float(receiver.employment_income),
                float(receiver.capital_gains),
                float(receiver.interest_dividend_income),
                float(receiver.business_farm_income),
            ],
            dtype=float,
        )
        / receiver_scale
    )
    composition_distance = np.abs(candidate_ratio - receiver_ratio).sum(axis=1)
    scale_distance = np.abs(np.log1p(candidate_scale) - np.log1p(receiver_scale))

    marital_target = 2 if bool(getattr(receiver, "is_married", False)) else 1
    marital_mass = np.where(
        candidates["MARS"].to_numpy(dtype=float) == marital_target,
        1.35,
        0.90,
    )

    probabilities = (
        score_mass
        * np.exp(-0.70 * composition_distance - 0.35 * scale_distance)
        * marital_mass
    )
    if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
        probabilities = np.ones(len(candidates), dtype=float)
    return probabilities / probabilities.sum()


def classify_donor_archetypes(donor_bucket: pd.DataFrame) -> pd.Series:
    """Classify donor records by their dominant high-income source."""

    scores = pd.DataFrame(
        {
            "ltcg": np.abs(donor_bucket.get("P23250", 0)).to_numpy(dtype=float)
            + 0.5 * np.abs(donor_bucket.get("P22250", 0)).to_numpy(dtype=float),
            "partnership": np.abs(donor_bucket.get("E26270", 0)).to_numpy(dtype=float)
            + np.abs(donor_bucket.get("E00900", 0)).to_numpy(dtype=float)
            + np.abs(donor_bucket.get("E02100", 0)).to_numpy(dtype=float),
            "dividend": np.abs(donor_bucket.get("E00600", 0)).to_numpy(dtype=float)
            + np.abs(donor_bucket.get("E00650", 0)).to_numpy(dtype=float)
            + np.abs(donor_bucket.get("E00300", 0)).to_numpy(dtype=float)
            + np.abs(donor_bucket.get("E00400", 0)).to_numpy(dtype=float),
            "mixed": np.abs(donor_bucket.get("E00200", 0)).to_numpy(dtype=float),
        },
        index=donor_bucket.index,
    )
    archetype = scores.idxmax(axis=1)
    return archetype.where(scores.max(axis=1) > 0, "mixed")


def classify_scf_archetypes(donor_pool: pd.DataFrame) -> pd.Series:
    """Classify SCF donors by the dominant wealth-to-income regime."""

    scores = pd.DataFrame(
        {
            "ltcg": np.abs(donor_pool["kginc"].to_numpy(dtype=float)),
            "partnership": np.abs(donor_pool["bussefarminc"].to_numpy(dtype=float))
            + 0.25 * np.abs(donor_pool["houses"].to_numpy(dtype=float)),
            "dividend": np.abs(donor_pool["intdivinc"].to_numpy(dtype=float)),
            "mixed": np.abs(donor_pool["wageinc"].to_numpy(dtype=float)),
        },
        index=donor_pool.index,
    )
    archetype = scores.idxmax(axis=1)
    return archetype.where(scores.max(axis=1) > 0, "mixed")


def apply_forbes_joint_amount_bases(
    selected: pd.DataFrame,
    forbes_draws: pd.DataFrame,
    donor_templates: pd.DataFrame,
) -> None:
    """Build major PUF amount priors from SCF draws plus PUF donor splits."""

    employment = np.maximum(
        forbes_draws["employment_income"].to_numpy(dtype=float),
        0.0,
    )
    capital_gains = forbes_draws["capital_gains"].to_numpy(dtype=float)
    investment = np.maximum(
        forbes_draws["interest_dividend_income"].to_numpy(dtype=float),
        0.0,
    )
    business = forbes_draws["business_farm_income"].to_numpy(dtype=float)
    pension_agi_component = apply_pension_amount_bases(
        selected=selected,
        forbes_draws=forbes_draws,
        donor_templates=donor_templates,
    )

    gains_split = allocate_rowwise_totals(
        totals=capital_gains,
        template_parts=donor_templates[["P23250", "P22250"]].to_numpy(dtype=float),
    )
    selected["P23250"] = gains_split[:, 0]
    selected["P22250"] = gains_split[:, 1]

    investment_split = allocate_rowwise_totals(
        totals=investment,
        template_parts=donor_templates[["E00300", "E00400", "E00600"]].to_numpy(
            dtype=float
        ),
        enforce_nonnegative=True,
    )
    selected["E00300"] = investment_split[:, 0]
    selected["E00400"] = investment_split[:, 1]
    selected["E00600"] = investment_split[:, 2]

    qdiv_denominator = np.maximum(
        donor_templates["E00600"].to_numpy(dtype=float),
        1e-9,
    )
    qdiv_share = np.divide(
        np.maximum(donor_templates["E00650"].to_numpy(dtype=float), 0.0),
        qdiv_denominator,
        out=np.zeros(len(donor_templates), dtype=float),
        where=qdiv_denominator > 0,
    )
    selected["E00650"] = selected["E00600"].to_numpy(dtype=float) * np.clip(
        qdiv_share,
        0.0,
        1.0,
    )

    business_split = allocate_rowwise_totals(
        totals=business,
        template_parts=donor_templates[["E26270", "E00900", "E02100"]].to_numpy(
            dtype=float
        ),
    )
    selected["E26270"] = business_split[:, 0]
    selected["E00900"] = business_split[:, 1]
    selected["E02100"] = business_split[:, 2]
    selected["E00200"] = employment

    template_major = donor_templates[list(PUF_AGI_COMPONENTS)].to_numpy(dtype=float)
    template_major_total = template_major.sum(axis=1)
    template_residual = (
        donor_templates["E00100"].to_numpy(dtype=float) - template_major_total
    )
    scaled_major_total = (
        selected["E00200"].to_numpy(dtype=float)
        + selected["P23250"].to_numpy(dtype=float)
        + selected["P22250"].to_numpy(dtype=float)
        + selected["E00300"].to_numpy(dtype=float)
        + selected["E00600"].to_numpy(dtype=float)
        + selected["E26270"].to_numpy(dtype=float)
        + selected["E00900"].to_numpy(dtype=float)
        + selected["E02100"].to_numpy(dtype=float)
        + pension_agi_component
    )
    scale = np.divide(
        np.maximum(np.abs(scaled_major_total), 1.0),
        np.maximum(np.abs(template_major_total), 1.0),
        out=np.ones(len(selected), dtype=float),
        where=np.maximum(np.abs(template_major_total), 1.0) > 0,
    )
    selected["E00100"] = np.maximum(
        np.abs(scaled_major_total + template_residual * scale),
        1.0,
    )


def apply_pension_amount_bases(
    selected: pd.DataFrame,
    forbes_draws: pd.DataFrame,
    donor_templates: pd.DataFrame,
) -> np.ndarray:
    """Map SCF retirement income draws to available PUF pension fields."""

    pension = np.maximum(
        forbes_draws["pension_income"].to_numpy(dtype=float),
        0.0,
    )
    gross_columns = [
        column
        for column in ("E01500", "E02400")
        if column in selected.columns and column in donor_templates.columns
    ]
    if gross_columns:
        gross_split = allocate_rowwise_totals(
            totals=pension,
            template_parts=donor_templates[gross_columns].to_numpy(dtype=float),
            enforce_nonnegative=True,
        )
        for index, column in enumerate(gross_columns):
            selected[column] = gross_split[:, index]

    if "E01700" in selected.columns and "E01700" in donor_templates.columns:
        if "E01500" in selected.columns and "E01500" in donor_templates.columns:
            donor_gross = np.maximum(
                donor_templates["E01500"].to_numpy(dtype=float),
                1e-9,
            )
            taxable_share = np.divide(
                np.maximum(donor_templates["E01700"].to_numpy(dtype=float), 0.0),
                donor_gross,
                out=np.zeros(len(donor_templates), dtype=float),
                where=donor_gross > 0,
            )
            selected["E01700"] = selected["E01500"].to_numpy(dtype=float) * np.clip(
                taxable_share,
                0.0,
                1.0,
            )
        else:
            selected["E01700"] = pension

    return 0.5 * pension


def allocate_rowwise_totals(
    totals: np.ndarray,
    template_parts: np.ndarray,
    enforce_nonnegative: bool = False,
) -> np.ndarray:
    """Allocate one total per row using donor template magnitudes."""

    totals = np.asarray(totals, dtype=float)
    template_parts = np.asarray(template_parts, dtype=float)
    if template_parts.ndim != 2:
        raise ValueError("template_parts must be 2-dimensional.")

    n_rows, n_cols = template_parts.shape
    if len(totals) != n_rows:
        raise ValueError("totals and template_parts must align row-wise.")

    magnitude = np.abs(totals)
    base = np.abs(template_parts)
    zero_rows = base.sum(axis=1) <= 1e-12
    if zero_rows.any():
        base[zero_rows] = 1.0
    shares = base / base.sum(axis=1, keepdims=True)
    allocated = shares * magnitude[:, None]

    if enforce_nonnegative:
        return allocated
    return np.sign(totals)[:, None] * allocated


def apply_forbes_structural_overrides(
    synthetic: pd.DataFrame,
    forbes: pd.DataFrame,
) -> None:
    """Set tax-unit structure and known state from Forbes metadata."""

    if "MARS" in synthetic.columns:
        married = forbes["is_married"].fillna(False).to_numpy(dtype=bool)
        synthetic["MARS"] = np.where(married, 2, 1).astype(int)

    if "XTOT" in synthetic.columns:
        if "MARS" in synthetic.columns:
            mars = synthetic["MARS"].to_numpy()
        else:
            mars = np.ones(len(synthetic), dtype=int)
        base_people = np.where(mars == 2, 2, 1)
        children = (
            pd.to_numeric(forbes["children"], errors="coerce").fillna(0).clip(0, 3)
        )
        synthetic["XTOT"] = np.clip(base_people + children.to_numpy(dtype=int), 1, 5)

    if "DSI" in synthetic.columns:
        synthetic["DSI"] = 0
    if "EIC" in synthetic.columns:
        synthetic["EIC"] = 0

    _apply_forbes_metadata(synthetic, forbes)


def _apply_forbes_metadata(
    synthetic: pd.DataFrame,
    forbes: pd.DataFrame,
) -> None:
    """Carry source Forbes metadata as household-level sidecar columns."""

    string_sources = {
        "forbes_alias": "alias",
        "forbes_name": "name",
        "forbes_snapshot_date": "snapshot_date",
        "forbes_marital_status": "marital_status",
    }
    for target, source in string_sources.items():
        if source in forbes.columns:
            synthetic[target] = forbes[source].fillna("").astype(str)
        else:
            synthetic[target] = FORBES_TOP_TAIL_METADATA_DEFAULTS[target]

    numeric_sources = {
        "forbes_rank": "rank",
        "forbes_unit_id": "forbes_unit_id",
        "forbes_replicate_id": "replicate_id",
        "forbes_age": "age",
        "forbes_children": "children",
    }
    for target, source in numeric_sources.items():
        if source in forbes.columns:
            synthetic[target] = (
                pd.to_numeric(forbes[source], errors="coerce")
                .fillna(FORBES_TOP_TAIL_METADATA_DEFAULTS[target])
                .astype(int)
            )
        else:
            synthetic[target] = FORBES_TOP_TAIL_METADATA_DEFAULTS[target]

    if "residence_state" in forbes.columns:
        synthetic["forbes_state_fips"] = forbes["residence_state"].map(
            _resolve_state_fips,
        )
    else:
        synthetic["forbes_state_fips"] = FORBES_TOP_TAIL_METADATA_DEFAULTS[
            "forbes_state_fips"
        ]


def _resolve_state_fips(value) -> int:
    """Resolve a Forbes residence state name/abbreviation to integer FIPS."""

    if value is None or pd.isna(value):
        return 0

    text = str(value).strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)

    fips = STATE_NAME_TO_FIPS.get(text)
    if fips is not None:
        return int(fips)

    fips = STATE_ABBREV_TO_FIPS.get(text.upper())
    if fips is not None:
        return int(fips)

    return 0


def _build_calibration_diagnostics(
    synthetic: pd.DataFrame,
    row: pd.Series,
    amount_columns: list[str],
    pop_weight: float,
    weights: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for column in amount_columns:
        target_total = pop_weight * float(row.get(column, 0.0))
        synthetic_total = _weighted_columns_total(synthetic, (column,), weights)
        absolute_error = synthetic_total - target_total
        if abs(target_total) > ARTIFACT_NUMERIC_TOL:
            relative_error = absolute_error / target_total
        else:
            relative_error = (
                0.0 if abs(absolute_error) <= ARTIFACT_NUMERIC_TOL else np.nan
            )
        squared_relative_error = (
            relative_error**2 if np.isfinite(relative_error) else np.nan
        )
        rows.append(
            {
                "column": column,
                "target_mean": float(row.get(column, 0.0)),
                "target_total": target_total,
                "synthetic_total": synthetic_total,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "squared_relative_error": squared_relative_error,
            }
        )
    return pd.DataFrame(rows)


def _build_composition_diagnostics(
    artifact: ForbesTopTailArtifact,
    row: pd.Series,
    pop_weight: float,
    weights: np.ndarray,
) -> pd.DataFrame:
    target_total_agi = pop_weight * float(row.get("E00100", 0.0))
    synthetic_total_agi = _weighted_columns_total(
        artifact.synthetic,
        ("E00100",),
        weights,
    )
    rows = []
    for component, columns in FORBES_DIAGNOSTIC_COMPONENTS:
        available_columns = tuple(
            column for column in columns if column in artifact.synthetic.columns
        )
        if not available_columns:
            continue

        target_total = pop_weight * sum(
            float(row.get(column, 0.0)) for column in columns
        )
        synthetic_total = _weighted_columns_total(
            artifact.synthetic,
            available_columns,
            weights,
        )
        puf_prior_total = _weighted_columns_total(
            artifact.puf_priors,
            available_columns,
            weights,
        )
        scf_columns = SCF_DIAGNOSTIC_COMPONENTS.get(component)
        scf_prior_total = (
            _weighted_columns_total(artifact.scf_draws, scf_columns, weights)
            if scf_columns
            else np.nan
        )
        synthetic_relative_error = _relative_error(synthetic_total, target_total)
        puf_prior_relative_error = _relative_error(puf_prior_total, target_total)
        scf_prior_relative_error = _relative_error(scf_prior_total, target_total)
        rows.append(
            {
                "component": component,
                "columns": ", ".join(available_columns),
                "target_total": target_total,
                "scf_prior_total": scf_prior_total,
                "puf_prior_total": puf_prior_total,
                "synthetic_total": synthetic_total,
                "absolute_error": synthetic_total - target_total,
                "synthetic_loss": _squared_relative_error(synthetic_relative_error),
                "puf_prior_relative_error": puf_prior_relative_error,
                "puf_prior_loss": _squared_relative_error(puf_prior_relative_error),
                "scf_prior_relative_error": scf_prior_relative_error,
                "scf_prior_loss": _squared_relative_error(scf_prior_relative_error),
                "share_of_agi": (
                    synthetic_total / synthetic_total_agi
                    if abs(synthetic_total_agi) > ARTIFACT_NUMERIC_TOL
                    else np.nan
                ),
                "target_share_of_agi": (
                    target_total / target_total_agi
                    if abs(target_total_agi) > ARTIFACT_NUMERIC_TOL
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _relative_error(actual: float, target: float) -> float:
    if not np.isfinite(actual) or not np.isfinite(target):
        return np.nan
    if abs(target) > ARTIFACT_NUMERIC_TOL:
        return (actual - target) / target
    if abs(actual) <= ARTIFACT_NUMERIC_TOL:
        return 0.0
    return np.nan


def _squared_relative_error(relative_error: float) -> float:
    return relative_error**2 if np.isfinite(relative_error) else np.nan


def _build_selection_diagnostics(selected_forbes: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank",
        "alias",
        "name",
        "networth_millions",
        "industry",
        "archetype",
        "scf_tail_probability",
        "scf_expected_agi",
        "estimated_agi",
        "is_married",
        "children",
        "self_made_flag",
        "residence_state",
    ]
    available_columns = [
        column for column in columns if column in selected_forbes.columns
    ]
    selection = selected_forbes[available_columns].copy()
    if "rank" in selection.columns:
        selection = selection.sort_values("rank")
    return selection.reset_index(drop=True)


def _build_draw_diagnostics(
    scf_draws: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    if len(scf_draws) != len(weights):
        raise ValueError("SCF draw diagnostics require one weight per draw.")
    if scf_draws.empty:
        return pd.DataFrame(
            columns=[
                "archetype",
                "draw_rows",
                "weighted_units",
                "forbes_units",
                "estimated_agi_weighted_mean",
            ]
        )

    draws = scf_draws.copy()
    draws["_diagnostic_weight"] = weights
    rows = []
    for archetype, group in draws.groupby("archetype", dropna=False):
        group_weights = group["_diagnostic_weight"].to_numpy(dtype=float)
        weighted_units = float(group_weights.sum())
        estimated_agi = group.get("estimated_agi", pd.Series(0.0, index=group.index))
        row = {
            "archetype": archetype,
            "draw_rows": len(group),
            "weighted_units": weighted_units,
            "forbes_units": (
                int(group["forbes_unit_id"].nunique())
                if "forbes_unit_id" in group
                else np.nan
            ),
            "estimated_agi_weighted_mean": (
                float(np.dot(estimated_agi.to_numpy(dtype=float), group_weights))
                / weighted_units
                if weighted_units > 0
                else np.nan
            ),
        }
        for column in (
            "employment_income",
            "capital_gains",
            "interest_dividend_income",
            "business_farm_income",
            "pension_income",
        ):
            if column in group:
                row[f"{column}_weighted_total"] = _weighted_columns_total(
                    group,
                    (column,),
                    group_weights,
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("weighted_units", ascending=False)


def _weighted_columns_total(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    weights: np.ndarray,
) -> float:
    if len(frame) != len(weights):
        raise ValueError("Weighted diagnostics require one weight per row.")

    available_columns = [column for column in columns if column in frame.columns]
    if not available_columns or frame.empty:
        return 0.0

    values = frame[available_columns].sum(axis=1).to_numpy(dtype=float)
    return float(np.dot(values, weights))


def _format_dollars(value) -> str:
    value = _safe_float(value)
    if not np.isfinite(value):
        return "n/a"

    sign = "-" if value < 0 else ""
    value = abs(value)
    for scale, suffix in (
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ):
        if value >= scale:
            return f"{sign}${value / scale:.1f}{suffix}"
    return f"{sign}${value:.0f}"


def _format_percent(value) -> str:
    value = _safe_float(value)
    if not np.isfinite(value):
        return "n/a"
    return f"{100 * value:.2f}%"


def _format_bar(value, max_value, width: int = 28) -> str:
    value = abs(_safe_float(value))
    max_value = abs(_safe_float(max_value))
    if not np.isfinite(value) or not np.isfinite(max_value) or max_value <= 0:
        return "." * width
    filled = int(round(width * min(value / max_value, 1.0)))
    return "#" * filled + "." * (width - filled)


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _numeric_series(df: pd.DataFrame, *columns: str) -> pd.Series:
    """Return the first available numeric column from a DataFrame."""

    for column in columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Return a weighted quantile for one-dimensional data."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot compute a quantile of an empty array.")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("Weighted quantile inputs must be finite.")

    quantile = float(np.clip(quantile, 0.0, 1.0))
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    total = cumulative[-1]
    if total <= 0:
        raise ValueError("Weighted quantile requires positive total weight.")
    cutoff = quantile * total
    return float(sorted_values[np.searchsorted(cumulative, cutoff, side="left")])
