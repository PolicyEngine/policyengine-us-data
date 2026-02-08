from policyengine_core.data import Dataset
import pandas as pd
from policyengine_us_data.utils import (
    pe_to_soi,
    get_soi,
    build_loss_matrix,
    fmt,
    HardConcrete,
    print_reweighting_diagnostics,
    set_seeds,
)
import numpy as np
from tqdm import trange
from typing import Type
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.extended_cps import (
    ExtendedCPS_2024,
    CPS_2024,
)
import logging

try:
    import torch
except ImportError:
    torch = None


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
        is_national,
        nation_normalisation_factor,
        state_normalisation_factor,
    )
    normalisation_factor = torch.tensor(
        normalisation_factor, dtype=torch.float32
    )
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
        rel_error = (
            ((estimate - targets_array) + 1) / (targets_array + 1)
        ) ** 2
        rel_error_normalized = (
            inv_mean_normalisation * rel_error * normalisation_factor
        )
        if torch.isnan(rel_error_normalized).any():
            raise ValueError("Relative error contains NaNs")
        return rel_error_normalized.mean()

    logging.info(
        f"Sparse optimization using seed {seed}, "
        f"temp {temperature} "
        f"init_mean {init_mean}, l0_lambda {l0_lambda}"
    )
    set_seeds(seed)

    weights = torch.tensor(
        np.log(original_weights),
        requires_grad=True,
        dtype=torch.float32,
    )
    gates = HardConcrete(
        len(original_weights),
        init_mean=init_mean,
        temperature=temperature,
    )
    # NOTE: Results are pretty sensitive to learning rates
    # optimizer breaks down near .005, does better above .1
    optimizer = torch.optim.Adam([weights] + list(gates.parameters()), lr=0.2)
    start_loss = None

    # lower learning rate, harder optimization
    iterator = trange(epochs * 2)
    performance = pd.DataFrame()
    for i in iterator:
        optimizer.zero_grad()
        masked = torch.exp(weights) * gates()
        l_main = loss(masked)
        l = l_main + l0_lambda * gates.get_penalty()
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
            start_loss = l.item()
        loss_rel_change = (l.item() - start_loss) / start_loss
        l.backward()
        iterator.set_postfix(
            {
                "loss": l.item(),
                "loss_rel_change": loss_rel_change,
            }
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

    #: When True, use NationalMatrixBuilder + l0-python for
    #: calibration.  When False, use the legacy reweight() path.
    use_db: bool = False

    def reweight_l0(self, db_path=None, lambda_l0=1e-6, epochs=1000):
        """Reweight using L0 calibration from DB targets.

        Args:
            db_path: Path to policy_data.db. If None, falls back
                to legacy build_loss_matrix targets.
            lambda_l0: L0 regularization strength.
            epochs: Number of training epochs.
        """
        from policyengine_us import Microsimulation
        from policyengine_us_data.calibration.fit_national_weights import (
            build_calibration_inputs,
            fit_national_weights,
            initialize_weights,
            save_weights_to_h5,
        )

        sim = Microsimulation(dataset=self.input_dataset)
        data = sim.dataset.load_dataset()
        original_weights = sim.calculate("household_weight").values

        for year in range(self.start_year, self.end_year + 1):
            matrix, targets, names = build_calibration_inputs(
                dataset_class=self.input_dataset,
                time_period=year,
                db_path=db_path,
                sim=sim,
            )
            init_weights = initialize_weights(original_weights)
            calibrated = fit_national_weights(
                matrix=matrix,
                targets=targets,
                initial_weights=init_weights,
                epochs=epochs,
                lambda_l0=lambda_l0,
            )
            data["household_weight"][year] = calibrated

        self.save_dataset(data)

    def generate(self):
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=self.input_dataset)
        data = sim.dataset.load_dataset()
        data["household_weight"] = {}
        original_weights = sim.calculate("household_weight")
        original_weights = original_weights.values + np.random.normal(
            1, 0.1, len(original_weights)
        )

        for year in range(self.start_year, self.end_year + 1):
            if self.use_db:
                optimised_weights = self._generate_db(
                    sim, original_weights, year
                )
            else:
                optimised_weights = self._generate_legacy(
                    original_weights, year
                )
            data["household_weight"][year] = optimised_weights

        self.save_dataset(data)

    # ----------------------------------------------------------
    # DB-driven calibration (NationalMatrixBuilder + l0-python)
    # ----------------------------------------------------------

    def _generate_db(self, sim, original_weights, year):
        """Run DB-driven national calibration for one year.

        Reads active targets from policy_data.db via
        NationalMatrixBuilder, then fits weights using
        l0-python's SparseCalibrationWeights.

        Args:
            sim: Microsimulation instance.
            original_weights: Jittered household weights.
            year: Tax year to calibrate.

        Returns:
            Optimised weight array (n_households,).
        """
        from policyengine_us_data.calibration.fit_national_weights import (
            build_calibration_inputs,
            fit_national_weights,
            initialize_weights,
        )

        db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
        matrix, targets, names = build_calibration_inputs(
            dataset_class=self.input_dataset,
            time_period=year,
            db_path=str(db_path),
            sim=sim,
        )

        init_weights = initialize_weights(original_weights)
        optimised_weights = fit_national_weights(
            matrix=matrix,
            targets=targets,
            initial_weights=init_weights,
            epochs=500,
        )

        logging.info(
            f"DB calibration for {year}: "
            f"{len(targets)} targets, "
            f"{(optimised_weights > 0).sum():,} non-zero weights"
        )
        return optimised_weights

    # ----------------------------------------------------------
    # Legacy calibration (build_loss_matrix + HardConcrete)
    # ----------------------------------------------------------

    def _generate_legacy(self, original_weights, year):
        """Run legacy national calibration for one year.

        Uses build_loss_matrix() and the HardConcrete-based
        reweight() function with the hardcoded bad_targets list.

        Args:
            original_weights: Jittered household weights.
            year: Tax year to calibrate.

        Returns:
            Optimised weight array (n_households,).
        """
        bad_targets = [
            "nation/irs/adjusted gross income/total/"
            "AGI in 10k-15k/taxable/Head of Household",
            "nation/irs/adjusted gross income/total/"
            "AGI in 15k-20k/taxable/Head of Household",
            "nation/irs/adjusted gross income/total/"
            "AGI in 10k-15k/taxable/"
            "Married Filing Jointly/Surviving Spouse",
            "nation/irs/adjusted gross income/total/"
            "AGI in 15k-20k/taxable/"
            "Married Filing Jointly/Surviving Spouse",
            "nation/irs/count/count/"
            "AGI in 10k-15k/taxable/Head of Household",
            "nation/irs/count/count/"
            "AGI in 15k-20k/taxable/Head of Household",
            "nation/irs/count/count/"
            "AGI in 10k-15k/taxable/"
            "Married Filing Jointly/Surviving Spouse",
            "nation/irs/count/count/"
            "AGI in 15k-20k/taxable/"
            "Married Filing Jointly/Surviving Spouse",
            "state/RI/adjusted_gross_income/amount/-inf_1",
            "nation/irs/exempt interest/count/" "AGI in -inf-inf/taxable/All",
        ]

        loss_matrix, targets_array = build_loss_matrix(
            self.input_dataset, year
        )
        zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
        bad_mask = loss_matrix.columns.isin(bad_targets)
        keep_mask_bool = ~(zero_mask | bad_mask)
        keep_idx = np.where(keep_mask_bool)[0]
        loss_matrix_clean = loss_matrix.iloc[:, keep_idx]
        targets_array_clean = targets_array[keep_idx]
        assert loss_matrix_clean.shape[1] == targets_array_clean.size

        optimised_weights = reweight(
            original_weights,
            loss_matrix_clean,
            targets_array_clean,
            log_path="calibration_log.csv",
            epochs=200,
            seed=1456,
        )
        return optimised_weights


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
            loss_matrix, targets_array = build_loss_matrix(
                self.input_dataset, year
            )
            optimised_weights = reweight(
                original_weights, loss_matrix, targets_array
            )
            data["household_weight"] = optimised_weights

        self.save_dataset(data)


class EnhancedCPS_2024(EnhancedCPS):
    input_dataset = ExtendedCPS_2024
    start_year = 2024
    end_year = 2024
    name = "enhanced_cps_2024"
    label = "Enhanced CPS 2024"
    file_path = STORAGE_FOLDER / "enhanced_cps_2024.h5"
    url = "hf://policyengine/policyengine-us-data/" "enhanced_cps_2024.h5"


if __name__ == "__main__":
    EnhancedCPS_2024().generate()
