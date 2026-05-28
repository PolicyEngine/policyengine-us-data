import json
from pathlib import Path

import h5py
from policyengine_core.data import Dataset
import pandas as pd
from policyengine_us_data.utils import (
    ABSOLUTE_ERROR_SCALE_TARGETS,
    HOUSEHOLD_COUNT_TARGET,
    build_loss_matrix,
    get_target_error_normalisation,
    get_target_loss_weights,
    HardConcrete,
    print_reweighting_diagnostics,
    set_seeds,
)
import gc
import numpy as np
from tqdm import trange
from typing import Type
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.extended_cps import (
    ExtendedCPS_2024_Half,
    CPS_2024,
)
from policyengine_us_data.storage.calibration_targets.aca_ptc_targets import (
    load_aca_ptc_state_targets,
)
from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode
from policyengine_us_data.utils.randomness import seeded_rng
from policyengine_us_data.utils.takeup import (
    ACA_POST_CALIBRATION_PERSON_TARGETS,
    adjust_aca_takeup_to_state_targets,
    extend_aca_takeup_to_match_target,
)
import logging

try:
    import torch
except ImportError:
    torch = None


HOUSEHOLD_WEIGHT_TOTAL_REL_TOLERANCE = 0.02
PERSON_POVERTY_RATE_MIN = 0.05
PERSON_POVERTY_RATE_MAX = 0.25
# PUF clones enter the extended CPS with zero household weight. They are support
# records for calibration, but the earlier bug starved them to ~0 (unusable in
# log-space optimization). Reserve a small but non-trivial share of prior mass
# for them, and validate that final weights keep them above a floor. There is no
# upper cap: the household-count loss target (loss.py) governs how much weight
# clones ultimately carry.
PUF_CLONE_PRIOR_TOTAL_SHARE = 0.05
MIN_PUF_CLONE_HOUSEHOLD_WEIGHT_SHARE_PCT = 5.0
MAX_PUF_CLONE_TAXES_EXCEED_MARKET_INCOME_SHARE_PCT = 25.0


def initialize_weight_priors(
    original_weights: np.ndarray,
    seed: int = 1456,
    epsilon: float = 1e-6,
    zero_weight_total_share: float = PUF_CLONE_PRIOR_TOTAL_SHARE,
) -> np.ndarray:
    """Build deterministic positive priors for sparse reweighting.

    PUF clone households enter the extended CPS with zero household weight.
    Reserve a small but non-trivial share of prior mass for them so they remain
    usable in log-space optimization (the earlier bug starved them to ~0). Their
    final weight is governed by the household-count loss target, not this prior.
    """

    weights = np.asarray(original_weights, dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("original_weights must be non-negative")
    if weights.size == 0:
        return weights.copy()
    if not 0 < zero_weight_total_share < 1:
        raise ValueError("zero_weight_total_share must be between 0 and 1")

    priors = np.empty_like(weights, dtype=np.float64)
    positive_mask = weights > 0
    zero_mask = ~positive_mask
    if not zero_mask.any():
        return weights.copy()

    positive_total = float(weights[positive_mask].sum())
    if positive_total <= 0:
        return np.full_like(weights, 1.0, dtype=np.float64)

    priors[positive_mask] = weights[positive_mask] * (1 - zero_weight_total_share)
    priors[zero_mask] = positive_total * zero_weight_total_share / zero_mask.sum()

    return priors


def validate_household_weight_total(
    weights: np.ndarray,
    *,
    source_total: float,
    year: int,
    rel_tolerance: float = HOUSEHOLD_WEIGHT_TOTAL_REL_TOLERANCE,
) -> float:
    """Validate calibrated household weights against the source total."""

    weights = np.asarray(weights)
    if np.any(np.isnan(weights)):
        raise ValueError(f"Year {year}: household_weight contains NaN values")
    if np.any(weights < 0):
        raise ValueError(f"Year {year}: household_weight contains negative values")

    weighted_hh_count = float(np.sum(weights))
    if not (1e8 <= weighted_hh_count <= 2e8):
        raise ValueError(
            f"Year {year}: weighted household count "
            f"{weighted_hh_count:,.0f} outside expected range "
            f"[100M, 200M]"
        )

    source_total = float(source_total)
    if not np.isfinite(source_total) or source_total <= 0:
        raise ValueError(
            f"Year {year}: source household count total must be positive; "
            f"got {source_total:,.0f}"
        )

    rel_error = abs(weighted_hh_count - source_total) / source_total
    if rel_error > rel_tolerance:
        raise ValueError(
            f"Year {year}: weighted household count "
            f"{weighted_hh_count:,.0f} differs from source household count "
            f"{source_total:,.0f} by {rel_error:.2%}, exceeding "
            f"{rel_tolerance:.2%} tolerance"
        )

    return weighted_hh_count


def validate_clone_household_weight_share(
    weights: np.ndarray,
    household_is_puf_clone: np.ndarray,
    *,
    year: int,
    min_share: float = MIN_PUF_CLONE_HOUSEHOLD_WEIGHT_SHARE_PCT / 100,
) -> float:
    """Validate that PUF-clone households keep a usable share of final weight.

    Clones must not be starved below ``min_share`` (the earlier bug left them at
    ~0, unusable in log-space optimization). There is no upper cap: the
    household-count loss target governs how much weight clones ultimately carry.
    """

    weights = np.asarray(weights, dtype=np.float64)
    household_is_puf_clone = np.asarray(household_is_puf_clone, dtype=bool)
    if len(weights) != len(household_is_puf_clone):
        raise ValueError(
            f"Year {year}: household_is_puf_clone length "
            f"{len(household_is_puf_clone)} does not match household_weight "
            f"length {len(weights)}"
        )

    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"Year {year}: household_weight total must be positive")

    clone_share = float(weights[household_is_puf_clone].sum()) / total
    if clone_share < min_share:
        raise ValueError(
            f"Year {year}: PUF-clone household weight share "
            f"{clone_share:.2%} is below the {min_share:.2%} floor; clones are "
            f"being starved of weight"
        )

    return clone_share


def _period_array_from_loaded_dataset(
    data: dict,
    variable_name: str,
    period: int,
) -> np.ndarray:
    values_by_period = data[variable_name]
    if period in values_by_period:
        return values_by_period[period]
    period_key = str(period)
    if period_key in values_by_period:
        return values_by_period[period_key]
    raise KeyError(f"{variable_name}[{period}] not found in loaded dataset")


def validate_person_poverty_rate(
    sim,
    *,
    year: int,
    min_rate: float = PERSON_POVERTY_RATE_MIN,
    max_rate: float = PERSON_POVERTY_RATE_MAX,
) -> float:
    """Fail fast when calibrated weights imply an implausible poverty rate."""

    poverty_rate = float(
        sim.calculate("person_in_poverty", period=year, map_to="person").mean()
    )
    if not np.isfinite(poverty_rate):
        raise ValueError(f"Year {year}: person poverty rate is not finite")
    if not (min_rate <= poverty_rate <= max_rate):
        raise ValueError(
            f"Year {year}: person poverty rate {poverty_rate:.2%} outside "
            f"expected range [{min_rate:.2%}, {max_rate:.2%}]"
        )
    return poverty_rate


def validate_clone_diagnostics(
    diagnostics: dict[str, float],
    *,
    min_household_weight_share_pct: float = MIN_PUF_CLONE_HOUSEHOLD_WEIGHT_SHARE_PCT,
    max_taxes_exceed_market_income_share_pct: float = (
        MAX_PUF_CLONE_TAXES_EXCEED_MARKET_INCOME_SHARE_PCT
    ),
) -> None:
    """Reject enhanced CPS artifacts where PUF support clones are starved.

    Enforces a floor on clone household weight share (clones must keep at least
    ``min_household_weight_share_pct`` of total weight, the earlier bug) plus a
    data-quality bound on clones whose imputed taxes exceed market income. There
    is no upper cap on weight share: the household-count loss target governs that.
    """

    clone_household_share = diagnostics["clone_household_weight_share_pct"]
    if clone_household_share < min_household_weight_share_pct:
        raise ValueError(
            "PUF clone household weight share "
            f"{clone_household_share:.1f}% is below the "
            f"{min_household_weight_share_pct:.1f}% floor"
        )

    taxes_exceed_market_income_share = diagnostics[
        "clone_taxes_exceed_market_income_share_pct"
    ]
    if taxes_exceed_market_income_share > max_taxes_exceed_market_income_share_pct:
        raise ValueError(
            "PUF clone taxes-exceed-market-income share "
            f"{taxes_exceed_market_income_share:.1f}% exceeds "
            f"{max_taxes_exceed_market_income_share_pct:.1f}%"
        )


def _to_numpy(value) -> np.ndarray:
    return np.asarray(getattr(value, "values", value))


def _weighted_share(mask, weights) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return 0.0
    mask = np.asarray(mask, dtype=bool)
    return 100 * float(weights[mask].sum()) / total_weight


def compute_clone_diagnostics_summary(
    *,
    household_is_puf_clone,
    household_weight,
    person_is_puf_clone,
    person_weight,
    person_in_poverty,
    spm_unit_is_puf_clone,
    spm_unit_weight,
    spm_unit_capped_work_childcare_expenses,
    spm_unit_pre_subsidy_childcare_expenses,
    spm_unit_taxes,
    spm_unit_market_income,
) -> dict[str, float]:
    household_is_puf_clone = np.asarray(household_is_puf_clone, dtype=bool)
    household_weight = np.asarray(household_weight, dtype=np.float64)
    person_is_puf_clone = np.asarray(person_is_puf_clone, dtype=bool)
    person_weight = np.asarray(person_weight, dtype=np.float64)
    person_in_poverty = np.asarray(person_in_poverty, dtype=bool)
    spm_unit_is_puf_clone = np.asarray(spm_unit_is_puf_clone, dtype=bool)
    spm_unit_weight = np.asarray(spm_unit_weight, dtype=np.float64)
    capped_childcare = np.asarray(
        spm_unit_capped_work_childcare_expenses, dtype=np.float64
    )
    pre_subsidy_childcare = np.asarray(
        spm_unit_pre_subsidy_childcare_expenses, dtype=np.float64
    )
    spm_unit_taxes = np.asarray(spm_unit_taxes, dtype=np.float64)
    spm_unit_market_income = np.asarray(spm_unit_market_income, dtype=np.float64)

    clone_spm_weight = spm_unit_weight[spm_unit_is_puf_clone].sum()

    return {
        "clone_household_weight_share_pct": _weighted_share(
            household_is_puf_clone, household_weight
        ),
        "clone_person_weight_share_pct": _weighted_share(
            person_is_puf_clone, person_weight
        ),
        "clone_poor_person_weight_share_pct": _weighted_share(
            person_is_puf_clone & person_in_poverty,
            person_weight,
        ),
        "clone_childcare_exceeds_pre_subsidy_share_pct": (
            0.0
            if clone_spm_weight <= 0
            else _weighted_share(
                capped_childcare[spm_unit_is_puf_clone]
                > pre_subsidy_childcare[spm_unit_is_puf_clone] + 1,
                spm_unit_weight[spm_unit_is_puf_clone],
            )
        ),
        "clone_childcare_above_5000_share_pct": (
            0.0
            if clone_spm_weight <= 0
            else _weighted_share(
                capped_childcare[spm_unit_is_puf_clone] > 5_000,
                spm_unit_weight[spm_unit_is_puf_clone],
            )
        ),
        "clone_taxes_exceed_market_income_share_pct": (
            0.0
            if clone_spm_weight <= 0
            else _weighted_share(
                spm_unit_taxes[spm_unit_is_puf_clone]
                > spm_unit_market_income[spm_unit_is_puf_clone] + 1,
                spm_unit_weight[spm_unit_is_puf_clone],
            )
        ),
    }


def _load_saved_period_array(
    file_path: str | Path,
    variable_name: str,
    period: int,
) -> np.ndarray:
    with h5py.File(file_path, "r") as h5_file:
        obj = h5_file[variable_name]
        if isinstance(obj, h5py.Dataset):
            return np.asarray(obj[...])
        period_key = str(period)
        if period_key in obj:
            return np.asarray(obj[period_key][...])
        if period in obj:
            return np.asarray(obj[period][...])
        raise KeyError(f"{variable_name} missing period {period}")


def clone_diagnostics_path(file_path: str | Path) -> Path:
    return Path(file_path).with_suffix(".clone_diagnostics.json")


def build_clone_diagnostics_payload(
    period_to_diagnostics: dict[int, dict[str, float]],
) -> dict:
    if not period_to_diagnostics:
        raise ValueError("Expected at least one period of clone diagnostics")

    ordered_periods = sorted(period_to_diagnostics)
    if len(ordered_periods) == 1:
        period = ordered_periods[0]
        diagnostics = dict(period_to_diagnostics[period])
        diagnostics["period"] = int(period)
        return diagnostics

    return {
        "periods": {
            str(period): period_to_diagnostics[period] for period in ordered_periods
        }
    }


def write_clone_diagnostics_report(file_path: str | Path, diagnostics: dict) -> Path:
    output_path = clone_diagnostics_path(file_path)
    output_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    return output_path


def refresh_clone_diagnostics_report(
    file_path: str | Path,
    diagnostics_builder,
) -> Path:
    output_path = clone_diagnostics_path(file_path)
    if output_path.exists():
        output_path.unlink()
    diagnostics = diagnostics_builder()
    return write_clone_diagnostics_report(file_path, diagnostics)


def save_clone_diagnostics_report(
    dataset_cls: Type[Dataset],
    *,
    start_year: int,
    end_year: int,
) -> tuple[Path, dict]:
    periods = list(range(start_year, end_year + 1))

    def build_validated_payload():
        period_to_diagnostics = {
            period: build_clone_diagnostics_for_saved_dataset(
                dataset_cls,
                period,
            )
            for period in periods
        }
        for diagnostics in period_to_diagnostics.values():
            validate_clone_diagnostics(diagnostics)
        return build_clone_diagnostics_payload(period_to_diagnostics)

    output_path = refresh_clone_diagnostics_report(
        dataset_cls.file_path,
        build_validated_payload,
    )
    diagnostics_payload = json.loads(output_path.read_text())
    return output_path, diagnostics_payload


def build_clone_diagnostics_for_saved_dataset(
    dataset_cls: Type[Dataset], period: int
) -> dict[str, float]:
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=dataset_cls)
    dataset_path = Path(dataset_cls.file_path)

    return build_clone_diagnostics_for_simulation(
        sim,
        dataset_path=dataset_path,
        period=period,
    )


def build_clone_diagnostics_for_simulation(
    sim,
    *,
    dataset_path: str | Path,
    period: int,
) -> dict[str, float]:
    """Build clone diagnostics from a simulation and saved clone-flag arrays.

    The enhanced CPS save path preserves zeroed person/spm-unit weight inputs on
    the clone half. For diagnostics, always map the calibrated household weights
    to persons/SPM units explicitly instead of reading those stale entity-level
    weight inputs back from disk.
    """

    return compute_clone_diagnostics_summary(
        household_is_puf_clone=_load_saved_period_array(
            dataset_path, "household_is_puf_clone", period
        ),
        household_weight=_to_numpy(sim.calculate("household_weight", period=period)),
        person_is_puf_clone=_load_saved_period_array(
            dataset_path, "person_is_puf_clone", period
        ),
        person_weight=_to_numpy(
            sim.calculate("household_weight", period=period, map_to="person")
        ),
        person_in_poverty=_to_numpy(sim.calculate("person_in_poverty", period=period)),
        spm_unit_is_puf_clone=_load_saved_period_array(
            dataset_path, "spm_unit_is_puf_clone", period
        ),
        spm_unit_weight=_to_numpy(
            sim.calculate("household_weight", period=period, map_to="spm_unit")
        ),
        spm_unit_capped_work_childcare_expenses=_to_numpy(
            sim.calculate("spm_unit_capped_work_childcare_expenses", period=period)
        ),
        spm_unit_pre_subsidy_childcare_expenses=_to_numpy(
            sim.calculate("spm_unit_pre_subsidy_childcare_expenses", period=period)
        ),
        spm_unit_taxes=_to_numpy(sim.calculate("spm_unit_taxes", period=period)),
        spm_unit_market_income=_to_numpy(
            sim.calculate("spm_unit_market_income", period=period)
        ),
    )


def _get_period_array(period_values: dict, period: int) -> np.ndarray:
    """Get a period array from a TIME_PERIOD_ARRAYS variable dict."""
    value = period_values.get(period)
    if value is None:
        value = period_values.get(str(period))
    if value is None:
        raise KeyError(f"Missing period {period}")
    return np.asarray(value)


def _get_base_aca_takeup(
    data: dict,
    base_year: int,
    tax_unit_count: int,
) -> np.ndarray:
    """Return stored ACA takeup or the default all-True baseline."""
    period_values = data.get("takes_up_aca_if_eligible")
    if period_values is None:
        logging.info(
            "takes_up_aca_if_eligible missing from base dataset; using default "
            "all-True takeup for ACA 2025 override"
        )
        return np.ones(tax_unit_count, dtype=bool)
    return _get_period_array(period_values, base_year).astype(bool, copy=False)


def _set_period_array(
    data: dict,
    variable: str,
    period: int,
    values: np.ndarray,
) -> None:
    """Store a time-period array, creating the variable entry if needed."""
    period_values = data.get(variable)
    if period_values is None:
        period_values = {}
        data[variable] = period_values
    period_values[period] = values


def _load_aca_enrollment_targets(period: int) -> dict[str, float] | None:
    path = (
        STORAGE_FOLDER
        / "calibration_targets"
        / f"aca_spending_and_enrollment_{period}.csv"
    )
    if not path.exists():
        return None
    targets = pd.read_csv(path)
    return {
        str(row.state): float(row.enrollment) for row in targets.itertuples(index=False)
    }


def _load_aca_spending_targets(period: int) -> dict[str, float] | None:
    soi_targets = load_aca_ptc_state_targets(period, storage_folder=STORAGE_FOLDER)
    if soi_targets is not None:
        return {
            str(row.state): float(row.TotalPTCAmount)
            for row in soi_targets.itertuples(index=False)
        }

    path = (
        STORAGE_FOLDER
        / "calibration_targets"
        / f"aca_spending_and_enrollment_{period}.csv"
    )
    if not path.exists():
        return None
    targets = pd.read_csv(path)
    return {
        str(row.state): float(row.spending) * 12
        for row in targets.itertuples(index=False)
    }


def _normalise_state_code(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _tax_unit_state_codes(
    person_state_codes: np.ndarray,
    person_tax_unit_idx: np.ndarray,
    tax_unit_count: int,
) -> np.ndarray:
    state_codes = np.full(tax_unit_count, "", dtype=object)
    for state_code, tax_unit_idx in zip(person_state_codes, person_tax_unit_idx):
        if state_codes[tax_unit_idx] == "":
            state_codes[tax_unit_idx] = _normalise_state_code(state_code)
    return state_codes


@pipeline_node(
    PipelineNode(
        id="aca_2025_override",
        label="ACA 2025 Take-Up Override",
        node_type="process",
        description=(
            "Adds synthetic 2025 ACA take-up assignments until calibrated "
            "person-level APTC enrollment reaches the target."
        ),
        status="transitional",
        stability="moving",
        pathways=["data_build"],
        artifacts_in=["extended_cps_2024"],
        artifacts_out=["aca_2025_takeup"],
        pydoc=True,
    )
)
def create_aca_2025_takeup_override(
    base_takeup: np.ndarray,
    person_enrolled_if_takeup: np.ndarray,
    person_weights: np.ndarray,
    person_tax_unit_ids: np.ndarray,
    tax_unit_ids: np.ndarray,
    target_people: float = ACA_POST_CALIBRATION_PERSON_TARGETS[2025],
    person_state_codes: np.ndarray | None = None,
    target_people_by_state: dict[str, float] | None = None,
    tax_unit_aca_ptc: np.ndarray | None = None,
    tax_unit_weights: np.ndarray | None = None,
    target_spending_by_state: dict[str, float] | None = None,
) -> np.ndarray:
    """Set 2025 ACA take-up to match APTC enrollment targets."""
    tax_unit_id_to_idx = {
        int(tax_unit_id): idx for idx, tax_unit_id in enumerate(tax_unit_ids)
    }
    person_tax_unit_idx = np.array(
        [tax_unit_id_to_idx[int(tax_unit_id)] for tax_unit_id in person_tax_unit_ids],
        dtype=np.int64,
    )
    enrolled_person_weights = np.zeros(len(tax_unit_ids), dtype=np.float64)
    np.add.at(
        enrolled_person_weights,
        person_tax_unit_idx,
        person_enrolled_if_takeup.astype(np.float64) * person_weights,
    )
    draws = seeded_rng("takes_up_aca_if_eligible").random(len(tax_unit_ids))

    if target_people_by_state is not None:
        if person_state_codes is None:
            raise ValueError(
                "person_state_codes are required for state-level ACA targets"
            )
        assigned_spending_weights = None
        if target_spending_by_state is not None:
            if tax_unit_aca_ptc is None or tax_unit_weights is None:
                raise ValueError(
                    "tax_unit_aca_ptc and tax_unit_weights are required for "
                    "state-level ACA spending targets"
                )
            assigned_spending_weights = np.asarray(
                tax_unit_aca_ptc, dtype=np.float64
            ) * np.asarray(tax_unit_weights, dtype=np.float64)
        return adjust_aca_takeup_to_state_targets(
            base_takeup=np.asarray(base_takeup, dtype=bool),
            entity_draws=draws,
            enrolled_person_weights=enrolled_person_weights,
            entity_state_codes=_tax_unit_state_codes(
                person_state_codes=person_state_codes,
                person_tax_unit_idx=person_tax_unit_idx,
                tax_unit_count=len(tax_unit_ids),
            ),
            target_people_by_state=target_people_by_state,
            assigned_spending_weights=assigned_spending_weights,
            target_spending_by_state=target_spending_by_state,
        )

    return extend_aca_takeup_to_match_target(
        base_takeup=np.asarray(base_takeup, dtype=bool),
        entity_draws=draws,
        enrolled_person_weights=enrolled_person_weights,
        target_people=target_people,
    )


@pipeline_node(
    PipelineNode(
        id="reweight",
        label="Enhanced CPS Reweighting",
        node_type="process",
        description=(
            "Fits enhanced CPS weights against calibration targets with the "
            "hard-concrete loss machinery."
        ),
        status="transitional",
        stability="moving",
        pathways=["data_build"],
        artifacts_in=["loss_matrix", "calibration_targets"],
        artifacts_out=["enhanced_cps_weights"],
        pydoc=True,
    )
)
def reweight(
    original_weights,
    loss_matrix,
    targets_array,
    log_path="calibration_log.csv",
    epochs=500,
    l0_lambda=0.0,
    init_mean=0.999,  # initial proportion with non-zero weights
    temperature=0.25,
    seed=1456,
):
    target_names = np.array(loss_matrix.columns)
    is_national = loss_matrix.columns.str.startswith("nation/")
    numerator_shift_np, error_denominator_np = get_target_error_normalisation(
        target_names,
        targets_array,
    )
    loss_matrix = torch.tensor(loss_matrix.values, dtype=torch.float32)
    nation_normalisation_factor = is_national * (1 / is_national.sum())
    state_normalisation_factor = ~is_national * (1 / (~is_national).sum())
    normalisation_factor = np.where(
        is_national, nation_normalisation_factor, state_normalisation_factor
    )
    target_loss_weights = get_target_loss_weights(target_names)
    normalisation_factor = torch.tensor(normalisation_factor, dtype=torch.float32)
    target_loss_weights = torch.tensor(target_loss_weights, dtype=torch.float32)
    targets_array = torch.tensor(targets_array, dtype=torch.float32)
    numerator_shift = torch.tensor(numerator_shift_np, dtype=torch.float32)
    error_denominator = torch.tensor(error_denominator_np, dtype=torch.float32)

    inv_mean_normalisation = 1 / np.mean(normalisation_factor.numpy())

    def loss(weights):
        if torch.isnan(weights).any():
            raise ValueError("Weights contain NaNs")
        if torch.isnan(loss_matrix).any():
            raise ValueError("Loss matrix contains NaNs")
        estimate = weights @ loss_matrix
        if torch.isnan(estimate).any():
            raise ValueError("Estimate contains NaNs")
        rel_error = (
            (estimate - targets_array + numerator_shift) / error_denominator
        ) ** 2
        rel_error_normalized = inv_mean_normalisation * rel_error * normalisation_factor
        rel_error_normalized = rel_error_normalized * target_loss_weights
        if torch.isnan(rel_error_normalized).any():
            raise ValueError("Relative error contains NaNs")
        return rel_error_normalized.mean()

    logging.info(
        f"Hard-concrete optimization using seed {seed}, temp {temperature} "
        + f"init_mean {init_mean}, l0_lambda {l0_lambda}"
    )
    set_seeds(seed)

    weights = torch.tensor(
        np.log(original_weights), requires_grad=True, dtype=torch.float32
    )
    gates = HardConcrete(
        len(original_weights), init_mean=init_mean, temperature=temperature
    )
    # NOTE: Results are pretty sensitve to learning rates
    # optimizer breaks down somewhere near .005, does better at above .1
    optimizer = torch.optim.Adam([weights] + list(gates.parameters()), lr=0.2)
    start_loss = None

    iterator = trange(epochs * 2)  # lower learning rate, harder optimization
    performance = pd.DataFrame()
    for i in iterator:
        optimizer.zero_grad()
        masked = torch.exp(weights) * gates()
        l_main = loss(masked)
        total_loss = l_main + l0_lambda * gates.get_penalty()
        if (log_path is not None) and (i % 10 == 0):
            gates.eval()
            estimates = (torch.exp(weights) * gates()) @ loss_matrix
            gates.train()
            estimates = estimates.detach().numpy()
            df = pd.DataFrame(
                {
                    "target_name": target_names,
                    "estimate": estimates,
                    "target": targets_array.detach().numpy(),
                }
            )
            df["epoch"] = i
            df["error"] = df.estimate - df.target
            df["error_denominator"] = error_denominator.detach().numpy()
            df["rel_error"] = (
                df.error + numerator_shift.detach().numpy()
            ) / df.error_denominator
            df["abs_error"] = df.error.abs()
            df["rel_abs_error"] = df.rel_error.abs()
            df["loss"] = df.rel_abs_error**2
            performance = pd.concat([performance, df], ignore_index=True)

        if (log_path is not None) and (i % 1000 == 0):
            performance.to_csv(log_path, index=False)
        if start_loss is None:
            start_loss = total_loss.item()
        loss_rel_change = (total_loss.item() - start_loss) / start_loss
        total_loss.backward()
        iterator.set_postfix(
            {"loss": total_loss.item(), "loss_rel_change": loss_rel_change}
        )
        optimizer.step()
        if log_path is not None:
            performance.to_csv(log_path, index=False)

    gates.eval()
    final_weights_sparse = (torch.exp(weights) * gates()).detach().numpy()

    print_reweighting_diagnostics(
        final_weights_sparse,
        loss_matrix,
        targets_array,
        "L0 Sparse Solution" if l0_lambda else "Unpenalized HardConcrete Solution",
        target_names=target_names,
    )

    return final_weights_sparse


class EnhancedCPS(Dataset):
    data_format = Dataset.TIME_PERIOD_ARRAYS
    input_dataset: Type[Dataset]
    start_year: int
    end_year: int

    def generate(self):
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=self.input_dataset)
        data = sim.dataset.load_dataset()
        base_year = int(sim.default_calculation_period)
        data["household_weight"] = {}
        original_weights = sim.calculate("household_weight")
        original_weights = initialize_weight_priors(original_weights.values)
        source_household_count = float(np.sum(original_weights))

        bad_targets = [
            "nation/irs/adjusted gross income/total/AGI in 10k-15k/taxable/Head of Household",
            "nation/irs/adjusted gross income/total/AGI in 15k-20k/taxable/Head of Household",
            "nation/irs/adjusted gross income/total/AGI in 10k-15k/taxable/Married Filing Jointly/Surviving Spouse",
            "nation/irs/adjusted gross income/total/AGI in 15k-20k/taxable/Married Filing Jointly/Surviving Spouse",
            "nation/irs/count/count/AGI in 10k-15k/taxable/Head of Household",
            "nation/irs/count/count/AGI in 15k-20k/taxable/Head of Household",
            "nation/irs/count/count/AGI in 10k-15k/taxable/Married Filing Jointly/Surviving Spouse",
            "nation/irs/count/count/AGI in 15k-20k/taxable/Married Filing Jointly/Surviving Spouse",
            "state/RI/adjusted_gross_income/amount/-inf_1",
            "nation/irs/exempt interest/count/AGI in -inf-inf/taxable/All",
        ]

        # Run the optimization procedure to get (close to) minimum loss weights
        for year in range(self.start_year, self.end_year + 1):
            loss_matrix, targets_array = build_loss_matrix(self.input_dataset, year)
            scaled_zero_target_mask = loss_matrix.columns.isin(
                ABSOLUTE_ERROR_SCALE_TARGETS.keys()
            )
            zero_mask = np.isclose(targets_array, 0.0, atol=0.1) & (
                ~scaled_zero_target_mask
            )
            bad_mask = loss_matrix.columns.isin(bad_targets)
            keep_mask_bool = ~(zero_mask | bad_mask)
            keep_idx = np.where(keep_mask_bool)[0]
            loss_matrix_clean = loss_matrix.iloc[:, keep_idx]
            targets_array_clean = targets_array[keep_idx]
            del loss_matrix, targets_array
            gc.collect()
            assert loss_matrix_clean.shape[1] == targets_array_clean.size
            if HOUSEHOLD_COUNT_TARGET not in loss_matrix_clean.columns:
                raise ValueError(
                    f"{HOUSEHOLD_COUNT_TARGET} missing from EnhancedCPS "
                    "calibration targets"
                )

            loss_matrix_clean = loss_matrix_clean.astype(np.float32)

            optimised_weights = reweight(
                original_weights,
                loss_matrix_clean,
                targets_array_clean,
                log_path="calibration_log.csv",
                epochs=500,
                seed=1456,
            )
            data["household_weight"][year] = optimised_weights
            sim.set_input(
                "household_weight",
                year,
                optimised_weights.astype(np.float32),
            )

            weighted_hh_count = validate_household_weight_total(
                optimised_weights,
                source_total=source_household_count,
                year=year,
            )
            clone_household_share = validate_clone_household_weight_share(
                optimised_weights,
                _period_array_from_loaded_dataset(
                    data,
                    "household_is_puf_clone",
                    year,
                ),
                year=year,
            )
            poverty_rate = validate_person_poverty_rate(sim, year=year)
            logging.info(
                f"Year {year}: weights validated — "
                f"{weighted_hh_count:,.0f} weighted households "
                f"vs {source_household_count:,.0f} source households, "
                f"{clone_household_share:.1%} PUF-clone household share, "
                f"{poverty_rate:.1%} person poverty rate, "
                f"{int(np.sum(optimised_weights > 0))} non-zero"
            )

        if 2025 in ACA_POST_CALIBRATION_PERSON_TARGETS:
            sim.set_input(
                "household_weight",
                base_year,
                _get_period_array(data["household_weight"], base_year).astype(
                    np.float32
                ),
            )
            sim.set_input(
                "takes_up_aca_if_eligible",
                2025,
                np.ones(
                    len(_get_period_array(data["tax_unit_id"], base_year)),
                    dtype=bool,
                ),
            )
            sim.delete_arrays("aca_ptc")

            _set_period_array(
                data=data,
                variable="takes_up_aca_if_eligible",
                period=2025,
                values=create_aca_2025_takeup_override(
                    base_takeup=_get_base_aca_takeup(
                        data=data,
                        base_year=base_year,
                        tax_unit_count=len(
                            _get_period_array(data["tax_unit_id"], base_year)
                        ),
                    ),
                    person_enrolled_if_takeup=np.asarray(
                        sim.calculate(
                            "aca_ptc",
                            map_to="person",
                            period=2025,
                            use_weights=False,
                        )
                    )
                    > 0,
                    person_weights=np.asarray(
                        sim.calculate(
                            "person_weight",
                            period=2025,
                            use_weights=False,
                        )
                    ),
                    person_tax_unit_ids=_get_period_array(
                        data["person_tax_unit_id"],
                        base_year,
                    ),
                    tax_unit_ids=_get_period_array(data["tax_unit_id"], base_year),
                    person_state_codes=np.asarray(
                        sim.calculate(
                            "state_code",
                            map_to="person",
                            period=2025,
                            use_weights=False,
                        )
                    ),
                    target_people_by_state=_load_aca_enrollment_targets(2025),
                    tax_unit_aca_ptc=np.asarray(
                        sim.calculate(
                            "aca_ptc",
                            period=2025,
                            use_weights=False,
                        )
                    ),
                    tax_unit_weights=np.asarray(
                        sim.calculate(
                            "tax_unit_weight",
                            period=2025,
                            use_weights=False,
                        )
                    ),
                    target_spending_by_state=_load_aca_spending_targets(2025),
                ),
            )

        logging.info("Post-generation weight validation passed")

        self.save_dataset(data)
        try:
            output_path, diagnostics_payload = save_clone_diagnostics_report(
                type(self),
                start_year=self.start_year,
                end_year=self.end_year,
            )
            logging.info("Saved clone diagnostics to %s", output_path)
            logging.info(
                "Clone diagnostics summary: %s",
                diagnostics_payload,
            )
        except Exception:
            logging.exception(
                "Unable to compute clone diagnostics for %s",
                self.file_path,
            )
            raise


class ReweightedCPS_2024(Dataset):
    data_format = Dataset.ARRAYS
    file_path = STORAGE_FOLDER / "reweighted_cps_2024.h5"
    name = "reweighted_cps_2024"
    label = "Reweighted CPS 2024"
    input_dataset = CPS_2024
    time_period = 2024

    def generate(self):
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=self.input_dataset)
        data = sim.dataset.load_dataset()
        original_weights = sim.calculate("household_weight")
        original_weights = initialize_weight_priors(original_weights.values)
        source_household_count = float(np.sum(original_weights))
        for year in [2024]:
            loss_matrix, targets_array = build_loss_matrix(self.input_dataset, year)
            optimised_weights = reweight(original_weights, loss_matrix, targets_array)
            validate_household_weight_total(
                optimised_weights,
                source_total=source_household_count,
                year=year,
            )
            data["household_weight"] = optimised_weights

        self.save_dataset(data)


class EnhancedCPS_2024(EnhancedCPS):
    input_dataset = ExtendedCPS_2024_Half
    start_year = 2024
    end_year = 2024
    time_period = 2024
    name = "enhanced_cps_2024"
    label = "Enhanced CPS 2024"
    file_path = STORAGE_FOLDER / "enhanced_cps_2024.h5"
    url = "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"


if __name__ == "__main__":
    EnhancedCPS_2024().generate()
