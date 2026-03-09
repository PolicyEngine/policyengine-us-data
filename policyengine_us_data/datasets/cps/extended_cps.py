import logging
from typing import Type

import numpy as np
from policyengine_core.data import Dataset

from policyengine_us_data.datasets.cps.cps import *  # noqa: F403
from policyengine_us_data.datasets.puf import *  # noqa: F403
from policyengine_us_data.storage import STORAGE_FOLDER

logger = logging.getLogger(__name__)


class ExtendedCPS(Dataset):
    cps: Type[CPS]
    puf: Type[PUF]
    data_format = Dataset.TIME_PERIOD_ARRAYS

    def generate(self):
        from policyengine_us import Microsimulation

        from policyengine_us_data.calibration.clone_and_assign import (
            load_global_block_distribution,
        )
        from policyengine_us_data.calibration.puf_impute import (
            puf_clone_dataset,
        )

        logger.info("Loading CPS dataset: %s", self.cps)
        cps_sim = Microsimulation(dataset=self.cps)
        data = cps_sim.dataset.load_dataset()
        del cps_sim

        data_dict = {}
        for var in data:
            data_dict[var] = {self.time_period: data[var][...]}

        n_hh = len(data_dict["household_id"][self.time_period])
        _, _, block_states, block_probs = load_global_block_distribution()
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(block_states), size=n_hh, p=block_probs)
        state_fips = block_states[indices]

        logger.info("PUF clone with dataset: %s", self.puf)
        new_data = puf_clone_dataset(
            data=data_dict,
            state_fips=state_fips,
            time_period=self.time_period,
            puf_dataset=self.puf,
            dataset_path=str(self.cps.file_path),
        )

        new_data = self._rename_imputed_to_inputs(new_data)
        new_data = self._drop_formula_variables(new_data)
        self.save_dataset(new_data)

    @classmethod
    def _rename_imputed_to_inputs(cls, data):
        """Rename QRF-imputed formula vars to their leaf inputs.

        The QRF imputes formula-level aggregates (e.g.
        taxable_pension_income) but the engine needs leaf inputs
        (e.g. taxable_private_pension_income) so formulas work.
        """
        for formula_var, input_var in cls._IMPUTED_TO_INPUT.items():
            if formula_var in data and input_var not in data:
                logger.info(
                    "Renaming %s -> %s (leaf input)",
                    formula_var,
                    input_var,
                )
                data[input_var] = data.pop(formula_var)
        return data

    # Variables with formulas that must still be stored (e.g. IDs
    # needed by the dataset loader before formulas can run).
    _KEEP_FORMULA_VARS = {"person_id"}

    # QRF imputes formula-level variables (e.g. taxable_pension_income)
    # but we must store them under leaf input names so
    # _drop_formula_variables doesn't discard them. The engine then
    # recomputes the formula var from its adds.
    _IMPUTED_TO_INPUT = {
        "taxable_pension_income": "taxable_private_pension_income",
        "tax_exempt_pension_income": "tax_exempt_private_pension_income",
        "interest_deduction": "deductible_mortgage_interest",
        "self_employed_pension_contribution_ald": (
            "self_employed_pension_contribution_ald_person"
        ),
        "self_employed_health_insurance_ald": (
            "self_employed_health_insurance_ald_person"
        ),
    }

    @classmethod
    def _drop_formula_variables(cls, data):
        """Remove variables that are computed by policyengine-us.

        Variables with formulas, ``adds``, or ``subtracts`` are
        recomputed by the simulation engine, so storing them wastes
        space and can mislead validation.

        Aggregate variables whose ``adds`` include a behavioral-
        response input (e.g. ``employment_income_before_lsr``) are
        renamed to that input before dropping so the raw data is
        preserved under the correct input-variable name.
        """
        from policyengine_us import CountryTaxBenefitSystem

        tbs = CountryTaxBenefitSystem()

        _RESPONSE_SUFFIXES = ("_before_lsr", "_before_response")
        for name, var in tbs.variables.items():
            if name not in data:
                continue
            for add_var in getattr(var, "adds", None) or []:
                if any(add_var.endswith(s) for s in _RESPONSE_SUFFIXES):
                    if add_var not in data:
                        logger.info(
                            "Renaming %s -> %s before drop",
                            name,
                            add_var,
                        )
                        data[add_var] = data.pop(name)
                    break

        formula_vars = {
            name
            for name, var in tbs.variables.items()
            if (hasattr(var, "formulas") and len(var.formulas) > 0)
            or getattr(var, "adds", None)
            or getattr(var, "subtracts", None)
        } - cls._KEEP_FORMULA_VARS
        dropped = sorted(set(data.keys()) & formula_vars)
        if dropped:
            logger.info(
                "Dropping %d formula variables: %s",
                len(dropped),
                dropped,
            )
            for var in dropped:
                del data[var]
        return data


class ExtendedCPS_2024(ExtendedCPS):
    cps = CPS_2024_Full
    puf = PUF_2024
    name = "extended_cps_2024"
    label = "Extended CPS (2024)"
    file_path = STORAGE_FOLDER / "extended_cps_2024.h5"
    time_period = 2024


class ExtendedCPS_2024_Half(ExtendedCPS):
    cps = CPS_2024
    puf = PUF_2024
    name = "extended_cps_2024_half"
    label = "Extended CPS 2024 (half sample)"
    file_path = STORAGE_FOLDER / "extended_cps_2024_half.h5"
    time_period = 2024


if __name__ == "__main__":
    ExtendedCPS_2024().generate()
    ExtendedCPS_2024_Half().generate()
