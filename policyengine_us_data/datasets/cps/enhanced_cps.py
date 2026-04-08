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


def initialize_weight_priors(
    original_weights: np.ndarray,
    seed: int = 1456,
    epsilon: float = 1e-6,
    positive_jitter_scale: float = 0.01,
) -> np.ndarray:
    """Build deterministic positive priors for sparse reweighting.

    Original CPS households should keep priors close to their survey
    weights. Clone-half households start with zero weight on purpose, so
    they should receive only a tiny positive epsilon to keep the log
    optimization well-defined without giving them a meaningful head start.
    """

    weights = np.asarray(original_weights, dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("original_weights must be non-negative")

    rng = np.random.default_rng(seed)
    priors = np.empty_like(weights, dtype=np.float64)

    positive_mask = weights > 0
    if positive_mask.any():
        jitter = np.maximum(
            rng.normal(loc=1.0, scale=positive_jitter_scale, size=positive_mask.sum()),
            0.5,
        )
        priors[positive_mask] = np.maximum(weights[positive_mask] * jitter, epsilon)

    zero_mask = ~positive_mask
    if zero_mask.any():
        priors[zero_mask] = epsilon * rng.uniform(1.0, 2.0, size=zero_mask.sum())

    return priors


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
        original_weights = initialize_weight_priors(
            sim.calculate("household_weight").values,
            seed=1456,
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
        original_weights = initialize_weight_priors(
            sim.calculate("household_weight").values,
            seed=1456,
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
