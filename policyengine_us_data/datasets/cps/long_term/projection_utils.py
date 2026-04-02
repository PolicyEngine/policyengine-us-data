import os
import gc
import sys
import numpy as np
import h5py

from policyengine_us import Microsimulation
from policyengine_core.data.dataset import Dataset


def validate_projected_social_security_cap(
    parameter_accessor,
    year: int,
    *,
    reference_year: int = 2035,
) -> float:
    """
    Ensure the Social Security taxable earnings cap keeps growing beyond the
    last explicitly projected year.

    The long-run calibration and diagnostics use taxable payroll targets
    through 2100. If the payroll cap flattens after the reference year, the
    late-year taxable payroll problem becomes mechanically distorted.
    """
    current_cap = float(
        parameter_accessor(year).gov.irs.payroll.social_security.cap
    )
    reference_cap = float(
        parameter_accessor(reference_year).gov.irs.payroll.social_security.cap
    )
    if year > reference_year and current_cap <= reference_cap * (1 + 1e-12):
        raise RuntimeError(
            "Social Security payroll cap is flat after "
            f"{reference_year}: {current_cap:,.2f} in {year}. "
            "This usually means policyengine-us is missing the long-run NAWI/"
            "payroll-cap extension."
        )
    return current_cap


def build_household_age_matrix(sim, n_ages=86):
    """
    Build household age composition matrix from simulation.

    Args:
        sim: PolicyEngine Microsimulation instance
        n_ages: Number of age groups (default 86 for ages 0-85+)

    Returns:
        X: Household age matrix (n_households x n_ages)
        household_ids_unique: Array of unique household IDs
        hh_id_to_idx: Dict mapping household ID to row index
    """
    age_person = sim.calculate("age")
    person_household_id = sim.calculate("person_household_id")

    household_ids_unique = np.unique(person_household_id.values)
    n_households = len(household_ids_unique)

    X = np.zeros((n_households, n_ages))
    hh_id_to_idx = {hh_id: idx for idx, hh_id in enumerate(household_ids_unique)}

    for person_idx in range(len(age_person)):
        age = int(age_person.values[person_idx])
        hh_id = person_household_id.values[person_idx]
        hh_idx = hh_id_to_idx[hh_id]
        age_idx = min(age, 85)
        X[hh_idx, age_idx] += 1

    return X, household_ids_unique, hh_id_to_idx


def build_age_bins(n_ages=86, bucket_size=None):
    """
    Build age-bucket ranges over the single-year age target vector.

    The final bucket always preserves the open-ended 85+ slot.
    """
    if bucket_size is None or bucket_size <= 1:
        return [(age_idx, age_idx + 1) for age_idx in range(n_ages)]

    bins = []
    upper_single_age = max(n_ages - 1, 0)
    for start in range(0, upper_single_age, bucket_size):
        end = min(start + bucket_size, upper_single_age)
        bins.append((start, end))
    bins.append((upper_single_age, n_ages))
    return bins


def aggregate_household_age_matrix(X, age_bins):
    """
    Aggregate a single-year household age matrix into coarser age buckets.
    """
    if len(age_bins) == X.shape[1] and all(end - start == 1 for start, end in age_bins):
        return X
    return np.column_stack([X[:, start:end].sum(axis=1) for start, end in age_bins])


def aggregate_age_targets(targets, age_bins):
    """
    Aggregate age targets over the first axis.

    Accepts either a single target vector `(n_ages,)` or a matrix
    `(n_ages, n_years)`.
    """
    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        return np.array(
            [targets[start:end].sum() for start, end in age_bins],
            dtype=float,
        )

    return np.vstack(
        [
            targets[start:end, :].sum(axis=0)
            for start, end in age_bins
        ]
    )


def get_pseudo_input_variables(sim):
    """
    Identify variables that appear as inputs but aggregate calculated values.

    These variables have 'adds' attribute but no formula, yet their components
    ARE calculated. Storing them leads to stale values corrupting calculations.
    """
    tbs = sim.tax_benefit_system
    pseudo_inputs = set()

    for var_name in sim.input_variables:
        var = tbs.variables.get(var_name)
        if not var:
            continue
        adds = getattr(var, "adds", None)
        if not adds or not isinstance(adds, list):
            continue
        for component in adds:
            comp_var = tbs.variables.get(component)
            if comp_var and len(getattr(comp_var, "formulas", {})) > 0:
                pseudo_inputs.add(var_name)
                break

    return pseudo_inputs


def create_household_year_h5(
    year,
    household_weights,
    base_dataset,
    output_dir,
    *,
    reform=None,
):
    """
    Create a year-specific .h5 file with calibrated household weights.

    This simplified version only saves weights and essential IDs, letting
    PolicyEngine uprate and calculate all other variables on-the-fly.

    Args:
        year: The year for this dataset
        household_weights: Calibrated household weights for this year
        base_dataset: Path to base dataset or in-memory Dataset instance
        output_dir: Directory to save the .h5 file
        reform: Optional reform to apply when materializing year-specific values

    Returns:
        Path to the created .h5 file
    """
    output_path = os.path.join(output_dir, f"{year}.h5")

    sim = Microsimulation(dataset=base_dataset, reform=reform)
    base_period = int(sim.default_calculation_period)

    df = sim.to_input_dataframe()

    # Remove pseudo-input variables (aggregates of calculated values)
    pseudo_inputs = get_pseudo_input_variables(sim)
    cols_to_drop = [
        f"{var}__{base_period}"
        for var in pseudo_inputs
        if f"{var}__{base_period}" in df.columns
    ]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    household_ids = sim.calculate("household_id", map_to="household").values
    person_household_id = df[f"person_household_id__{base_period}"]

    hh_to_weight = dict(zip(household_ids, household_weights))
    person_weights = person_household_id.map(hh_to_weight)

    df[f"household_weight__{year}"] = person_weights
    df.drop(
        columns=[
            f"household_weight__{base_period}",
            f"person_weight__{base_period}",
        ],
        inplace=True,
        errors="ignore",
    )

    for col in df.columns:
        if f"__{base_period}" in col:
            var_name = col.replace(f"__{base_period}", "")
            col_name_new = f"{var_name}__{year}"

            if var_name in ["household_weight", "person_weight"]:
                continue

            try:
                uprated_values = sim.calculate(var_name, period=year).values

                if len(uprated_values) == len(df):
                    df[col_name_new] = uprated_values
                    df.drop(columns=[col], inplace=True)
                else:
                    print(
                        f"Warning: uprating {var_name} for {year} returned "
                        f"{len(uprated_values)} rows instead of {len(df)}; "
                        "renaming the base-year column without recalculation.",
                        file=sys.stderr,
                    )
                    df.rename(columns={col: col_name_new}, inplace=True)

            except Exception as error:
                print(
                    f"Warning: failed to uprate {var_name} for {year}: {error}; "
                    "renaming the base-year column without recalculation.",
                    file=sys.stderr,
                )
                df.rename(columns={col: col_name_new}, inplace=True)

    dataset = Dataset.from_dataframe(df, year)

    new_sim = Microsimulation()
    new_sim.dataset = dataset
    new_sim.build_from_dataset()

    data = {}
    for variable in new_sim.tax_benefit_system.variables:
        holder = new_sim.get_holder(variable)
        known_periods = holder.get_known_periods()

        if len(known_periods) > 0:
            data[variable] = {}
            for period in known_periods:
                values = holder.get_array(period)
                values = np.array(values)

                if values.dtype == np.object_:
                    try:
                        values = values.astype("S")
                    except (TypeError, ValueError):
                        continue

                data[variable][period] = values

    with h5py.File(output_path, "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)

    del sim, new_sim, dataset
    gc.collect()

    return output_path


def calculate_year_statistics(
    sim, year, X, y_target, baseline_weights, calibrate_fn, use_ss=False
):
    """
    Calculate statistics for a single projection year.

    Args:
        sim: PolicyEngine Microsimulation instance
        year: Year to calculate for
        X: Household age matrix
        y_target: Target age distribution
        baseline_weights: Initial household weights
        calibrate_fn: Function to calibrate weights
        use_ss: Whether to include Social Security in calibration

    Returns:
        Dictionary with year statistics and calibrated weights
    """
    income_tax_hh = sim.calculate("income_tax", period=year, map_to="household")
    income_tax_baseline_total = income_tax_hh.sum()
    income_tax_values = income_tax_hh.values

    household_microseries = sim.calculate("household_id", map_to="household")
    baseline_weights_actual = household_microseries.weights.values
    household_ids_hh = household_microseries.values

    ss_values = None
    ss_target = None
    if use_ss:
        ss_hh = sim.calculate("social_security", period=year, map_to="household")
        ss_baseline_total = ss_hh.sum()
        ss_values = ss_hh.values

    w_new, iterations = calibrate_fn(
        X=X,
        y_target=y_target,
        baseline_weights=baseline_weights_actual,
        ss_values=ss_values,
        ss_target=ss_target,
    )

    total_income_tax = np.sum(income_tax_values * w_new)
    total_population = np.sum(y_target)

    return {
        "year": year,
        "weights": w_new,
        "baseline_weights": baseline_weights_actual,
        "total_income_tax": total_income_tax,
        "total_income_tax_baseline": income_tax_baseline_total,
        "total_population": total_population,
        "iterations": iterations,
    }
