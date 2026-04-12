import json
from pathlib import Path

import h5py
from policyengine_core.data import Dataset
import pandas as pd
from policyengine_us_data.utils import (
    build_loss_matrix,
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
from policyengine_us_data.utils.randomness import seeded_rng
from policyengine_us_data.utils.takeup import (
    ACA_POST_CALIBRATION_PERSON_TARGETS,
    extend_aca_takeup_to_match_target,
)
import logging

try:
    import torch
except ImportError:
    torch = None


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
    person_reported_in_poverty,
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
    person_reported_in_poverty = np.asarray(person_reported_in_poverty, dtype=bool)
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

    poor_modeled_only = person_in_poverty & ~person_reported_in_poverty
    clone_spm_weight = spm_unit_weight[spm_unit_is_puf_clone].sum()

    return {
        "clone_household_weight_share_pct": _weighted_share(
            household_is_puf_clone, household_weight
        ),
        "clone_person_weight_share_pct": _weighted_share(
            person_is_puf_clone, person_weight
        ),
        "clone_poor_modeled_only_person_weight_share_pct": _weighted_share(
            person_is_puf_clone & poor_modeled_only,
            person_weight,
        ),
        "poor_modeled_only_within_clone_person_weight_share_pct": (
            0.0
            if person_weight[person_is_puf_clone].sum() <= 0
            else _weighted_share(
                poor_modeled_only[person_is_puf_clone],
                person_weight[person_is_puf_clone],
            )
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
    output_path = refresh_clone_diagnostics_report(
        dataset_cls.file_path,
        lambda: build_clone_diagnostics_payload(
            {
                period: build_clone_diagnostics_for_saved_dataset(
                    dataset_cls,
                    period,
                )
                for period in periods
            }
        ),
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

    person_reported_in_poverty = _to_numpy(
        sim.calculate("spm_unit_net_income_reported", period=period, map_to="person")
    ) < _to_numpy(
        sim.calculate("spm_unit_spm_threshold", period=period, map_to="person")
    )

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
        person_reported_in_poverty=person_reported_in_poverty,
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


def create_aca_2025_takeup_override(
    base_takeup: np.ndarray,
    person_enrolled_if_takeup: np.ndarray,
    person_weights: np.ndarray,
    person_tax_unit_ids: np.ndarray,
    tax_unit_ids: np.ndarray,
    target_people: float = ACA_POST_CALIBRATION_PERSON_TARGETS[2025],
) -> np.ndarray:
    """Add 2025 ACA takers until weighted APTC enrollment hits target."""
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

    return extend_aca_takeup_to_match_target(
        base_takeup=np.asarray(base_takeup, dtype=bool),
        entity_draws=draws,
        enrolled_person_weights=enrolled_person_weights,
        target_people=target_people,
    )


def reweight(
    original_weights,
    loss_matrix,
    targets_array,
    log_path="calibration_log.csv",
    epochs=500,
    l0_lambda=2.6445e-07,
    init_mean=0.999,  # initial proportion with non-zero weights
    temperature=0.25,
    seed=1456,
):
    target_names = np.array(loss_matrix.columns)
    is_national = loss_matrix.columns.str.startswith("nation/")
    loss_matrix = torch.tensor(loss_matrix.values, dtype=torch.float32)
    nation_normalisation_factor = is_national * (1 / is_national.sum())
    state_normalisation_factor = ~is_national * (1 / (~is_national).sum())
    normalisation_factor = np.where(
        is_national, nation_normalisation_factor, state_normalisation_factor
    )
    normalisation_factor = torch.tensor(normalisation_factor, dtype=torch.float32)
    targets_array = torch.tensor(targets_array, dtype=torch.float32)

    inv_mean_normalisation = 1 / np.mean(normalisation_factor.numpy())

    def loss(weights):
        if torch.isnan(weights).any():
            raise ValueError("Weights contain NaNs")
        if torch.isnan(loss_matrix).any():
            raise ValueError("Loss matrix contains NaNs")
        estimate = weights @ loss_matrix
        if torch.isnan(estimate).any():
            raise ValueError("Estimate contains NaNs")
        rel_error = (((estimate - targets_array) + 1) / (targets_array + 1)) ** 2
        rel_error_normalized = inv_mean_normalisation * rel_error * normalisation_factor
        if torch.isnan(rel_error_normalized).any():
            raise ValueError("Relative error contains NaNs")
        return rel_error_normalized.mean()

    logging.info(
        f"Sparse optimization using seed {seed}, temp {temperature} "
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
            df["rel_error"] = df.error / df.target
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
        "L0 Sparse Solution",
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
        original_weights = original_weights.values + np.random.normal(
            1, 0.1, len(original_weights)
        )

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
            zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
            bad_mask = loss_matrix.columns.isin(bad_targets)
            keep_mask_bool = ~(zero_mask | bad_mask)
            keep_idx = np.where(keep_mask_bool)[0]
            loss_matrix_clean = loss_matrix.iloc[:, keep_idx]
            targets_array_clean = targets_array[keep_idx]
            del loss_matrix, targets_array
            gc.collect()
            assert loss_matrix_clean.shape[1] == targets_array_clean.size

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

            # Validate dense weights
            w = optimised_weights
            if np.any(np.isnan(w)):
                raise ValueError(f"Year {year}: household_weight contains NaN values")
            if np.any(w < 0):
                raise ValueError(
                    f"Year {year}: household_weight contains negative values"
                )
            weighted_hh_count = float(np.sum(w))
            if not (1e8 <= weighted_hh_count <= 2e8):
                raise ValueError(
                    f"Year {year}: weighted household count "
                    f"{weighted_hh_count:,.0f} outside expected range "
                    f"[100M, 200M]"
                )
            logging.info(
                f"Year {year}: weights validated — "
                f"{weighted_hh_count:,.0f} weighted households, "
                f"{int(np.sum(w > 0))} non-zero"
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

            data["takes_up_aca_if_eligible"][2025] = create_aca_2025_takeup_override(
                base_takeup=_get_period_array(
                    data["takes_up_aca_if_eligible"],
                    base_year,
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
            logging.warning(
                "Unable to compute clone diagnostics for %s",
                self.file_path,
                exc_info=True,
            )


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
        original_weights = original_weights.values + np.random.normal(
            1, 0.1, len(original_weights)
        )
        for year in [2024]:
            loss_matrix, targets_array = build_loss_matrix(self.input_dataset, year)
            optimised_weights = reweight(original_weights, loss_matrix, targets_array)
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
