from functools import lru_cache

from microimpute.models.qrf import QRF
import numpy as np
import pandas as pd

from policyengine_us_data.storage import STORAGE_FOLDER

ORG_FILENAME = "census_cps_org_2024_wages.csv.gz"
ORG_YEAR = 2024
ORG_MONTHS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)
CPS_BASIC_MONTHLY_ORG_COLUMNS = [
    "HRMIS",
    "gestfips",
    "prtage",
    "pesex",
    "ptdtrace",
    "pehspnon",
    "pworwgt",
    "pternwa",
    "pternhly",
    "peernhry",
    "pehruslt",
    "prerelg",
    "pemlr",
    "peio1cow",
]

ORG_PREDICTORS = [
    "employment_income",
    "weekly_hours_worked",
    "age",
    "is_female",
    "is_hispanic",
    "race_wbho",
    "state_fips",
]

ORG_IMPUTED_VARIABLES = [
    "hourly_wage",
    "is_paid_hourly",
    "is_union_member_or_covered",
]
ORG_QRF_IMPUTED_VARIABLES = [
    "hourly_wage",
    "is_paid_hourly",
]

ORG_BOOL_VARIABLES = [
    "is_paid_hourly",
    "is_union_member_or_covered",
]

# BLS Table 1 and Table 5 represented-by-union rates for 2024.
# Sources:
#   https://www.bls.gov/news.release/union2.t01.htm
#   https://www.bls.gov/news.release/union2.t05.htm
BLS_UNION_REPRESENTATION_RATE_2024 = np.float32(0.111)
BLS_SEX_UNION_REPRESENTATION_RATE_2024 = {
    False: np.float32(0.113),  # men
    True: np.float32(0.108),  # women
}
BLS_AGE_UNION_REPRESENTATION_RATE_2024 = {
    (16, 24): np.float32(0.054),
    (25, 34): np.float32(0.099),
    (35, 44): np.float32(0.122),
    (45, 54): np.float32(0.138),
    (55, 64): np.float32(0.130),
    (65, 150): np.float32(0.102),
}
BLS_RACE_WBHO_UNION_REPRESENTATION_RATE_2024 = {
    1: np.float32(0.108),  # White, non-Hispanic
    2: np.float32(0.132),  # Black, non-Hispanic
    3: np.float32(0.097),  # Hispanic
    4: BLS_UNION_REPRESENTATION_RATE_2024,  # Other / not separately published
}
BLS_FULL_TIME_UNION_REPRESENTATION_RATE_2024 = np.float32(0.120)
BLS_PART_TIME_UNION_REPRESENTATION_RATE_2024 = np.float32(0.066)
BLS_STATE_UNION_REPRESENTATION_RATE_2024 = {
    1: np.float32(0.078),
    2: np.float32(0.195),
    4: np.float32(0.045),
    5: np.float32(0.044),
    6: np.float32(0.163),
    8: np.float32(0.080),
    9: np.float32(0.178),
    10: np.float32(0.089),
    11: np.float32(0.117),
    12: np.float32(0.063),
    13: np.float32(0.044),
    15: np.float32(0.275),
    16: np.float32(0.059),
    17: np.float32(0.142),
    18: np.float32(0.104),
    19: np.float32(0.083),
    20: np.float32(0.080),
    21: np.float32(0.112),
    22: np.float32(0.050),
    23: np.float32(0.153),
    24: np.float32(0.134),
    25: np.float32(0.156),
    26: np.float32(0.147),
    27: np.float32(0.148),
    28: np.float32(0.079),
    29: np.float32(0.093),
    30: np.float32(0.131),
    31: np.float32(0.081),
    32: np.float32(0.134),
    33: np.float32(0.106),
    34: np.float32(0.174),
    35: np.float32(0.088),
    36: np.float32(0.219),
    37: np.float32(0.031),
    38: np.float32(0.063),
    39: np.float32(0.133),
    40: np.float32(0.062),
    41: np.float32(0.175),
    42: np.float32(0.124),
    44: np.float32(0.153),
    45: np.float32(0.041),
    46: np.float32(0.037),
    47: np.float32(0.056),
    48: np.float32(0.054),
    49: np.float32(0.078),
    50: np.float32(0.158),
    51: np.float32(0.057),
    53: np.float32(0.183),
    54: np.float32(0.100),
    55: np.float32(0.069),
    56: np.float32(0.067),
}


def _derive_wbho_from_cps_race(
    cps_race: np.ndarray, is_hispanic: np.ndarray
) -> np.ndarray:
    """Map CPS race + Hispanic flag into a four-way WBHO-style scheme."""
    cps_race = np.asarray(cps_race)
    is_hispanic = np.asarray(is_hispanic).astype(bool)
    return np.select(
        [
            is_hispanic,
            (cps_race == 1) & ~is_hispanic,
            (cps_race == 2) & ~is_hispanic,
        ],
        [
            3,
            1,
            2,
        ],
        default=4,
    ).astype(np.int8)


def _cps_basic_org_month_url(year: int, month: str) -> str:
    year_suffix = str(year)[-2:]
    return (
        f"https://www2.census.gov/programs-surveys/cps/datasets/"
        f"{year}/basic/{month}{year_suffix}pub.csv"
    )


def _transform_cps_basic_org_month(month_df: pd.DataFrame) -> pd.DataFrame:
    """Convert one monthly CPS basic file into ORG donor rows.

    Uses the official public-use outgoing rotation group records and
    reconstructs hourly wage from the public weekly/hourly earnings
    recodes when a direct hourly rate is not reported.
    """
    org = month_df.copy()
    org = org.loc[
        org["HRMIS"].isin([4, 8])
        & (org["pworwgt"] > 0)
        & (org["prerelg"] == 1)
        & org["pemlr"].isin([1, 2])
        & org["peio1cow"].isin([1, 2, 3, 4, 5])
        & org["peernhry"].isin([1, 2])
        & (org["gestfips"] > 0)
        & (org["prtage"] >= 16)
        & (org["pehruslt"] > 0)
        & (org["pternwa"] > 0)
    ].copy()

    weekly_earnings = org["pternwa"].astype(np.float32) / 100
    direct_hourly_wage = org["pternhly"].astype(np.float32) / 100
    weekly_hours_worked = org["pehruslt"].astype(np.float32)
    imputed_hourly_wage = weekly_earnings / weekly_hours_worked

    is_hispanic = (org["pehspnon"] == 1).astype(np.float32)
    org["employment_income"] = weekly_earnings * 52
    org["weekly_hours_worked"] = weekly_hours_worked
    org["age"] = org["prtage"].astype(np.float32)
    org["is_female"] = (org["pesex"] == 2).astype(np.float32)
    org["is_hispanic"] = is_hispanic
    org["race_wbho"] = _derive_wbho_from_cps_race(
        org["ptdtrace"].values,
        is_hispanic.values,
    ).astype(np.float32)
    org["state_fips"] = org["gestfips"].astype(np.float32)
    org["hourly_wage"] = np.where(
        (org["peernhry"] == 1) & (direct_hourly_wage > 0),
        direct_hourly_wage,
        imputed_hourly_wage,
    ).astype(np.float32)
    org["is_paid_hourly"] = (org["peernhry"] == 1).astype(np.float32)
    org["sample_weight"] = org["pworwgt"].astype(np.float32)

    org = org.loc[
        (org["employment_income"] > 0)
        & (org["weekly_hours_worked"] > 0)
        & (org["hourly_wage"] > 0)
    ].copy()

    return org[ORG_PREDICTORS + ORG_QRF_IMPUTED_VARIABLES + ["sample_weight"]]


def build_org_receiver_frame(
    *,
    age: np.ndarray,
    is_female: np.ndarray,
    is_hispanic: np.ndarray,
    cps_race: np.ndarray,
    state_fips: np.ndarray,
    employment_income: np.ndarray,
    weekly_hours_worked: np.ndarray,
) -> pd.DataFrame:
    """Build the receiver-side feature frame used by ORG QRF models."""
    receiver = pd.DataFrame(
        {
            "employment_income": np.asarray(employment_income, dtype=np.float32),
            "weekly_hours_worked": np.asarray(
                weekly_hours_worked, dtype=np.float32
            ),
            "age": np.asarray(age, dtype=np.float32),
            "is_female": np.asarray(is_female, dtype=np.float32),
            "is_hispanic": np.asarray(is_hispanic, dtype=np.float32),
            "state_fips": np.asarray(state_fips, dtype=np.float32),
        }
    )
    receiver["race_wbho"] = _derive_wbho_from_cps_race(
        cps_race=cps_race,
        is_hispanic=is_hispanic,
    ).astype(np.float32)
    return receiver


def _lookup_state_union_representation_rates(
    state_fips: np.ndarray,
) -> np.ndarray:
    rates = np.full(
        len(state_fips),
        BLS_UNION_REPRESENTATION_RATE_2024,
        dtype=np.float32,
    )
    state_fips = np.asarray(state_fips, dtype=np.int32)
    for fips, rate in BLS_STATE_UNION_REPRESENTATION_RATE_2024.items():
        rates[state_fips == fips] = rate
    return rates


def _build_union_priority_weights(receiver: pd.DataFrame) -> np.ndarray:
    base_rate = float(BLS_UNION_REPRESENTATION_RATE_2024)

    age = np.asarray(receiver["age"], dtype=np.float32)
    age_rates = np.full(len(receiver), base_rate, dtype=np.float32)
    for (lower, upper), rate in BLS_AGE_UNION_REPRESENTATION_RATE_2024.items():
        age_rates[(age >= lower) & (age <= upper)] = rate

    is_female = np.asarray(receiver["is_female"] >= 0.5, dtype=bool)
    sex_rates = np.where(
        is_female,
        BLS_SEX_UNION_REPRESENTATION_RATE_2024[True],
        BLS_SEX_UNION_REPRESENTATION_RATE_2024[False],
    ).astype(np.float32)

    race_wbho = np.asarray(receiver["race_wbho"], dtype=np.int32)
    race_rates = np.full(len(receiver), base_rate, dtype=np.float32)
    for race_code, rate in BLS_RACE_WBHO_UNION_REPRESENTATION_RATE_2024.items():
        race_rates[race_wbho == race_code] = rate

    weekly_hours = np.asarray(receiver["weekly_hours_worked"], dtype=np.float32)
    hours_rates = np.where(
        weekly_hours >= 35,
        BLS_FULL_TIME_UNION_REPRESENTATION_RATE_2024,
        BLS_PART_TIME_UNION_REPRESENTATION_RATE_2024,
    ).astype(np.float32)

    weights = (
        (age_rates / base_rate)
        * (sex_rates / base_rate)
        * (race_rates / base_rate)
        * (hours_rates / base_rate)
    )
    return np.clip(weights.astype(np.float64), 1e-6, None)


def _stable_uniform_from_receiver(receiver: pd.DataFrame) -> np.ndarray:
    hash_frame = pd.DataFrame(
        {
            col: np.round(
                np.asarray(receiver[col], dtype=np.float64),
                4,
            )
            for col in ORG_PREDICTORS
        }
    )
    hashes = pd.util.hash_pandas_object(hash_frame, index=False).to_numpy(
        dtype=np.uint64
    )
    return (hashes.astype(np.float64) + 0.5) / (np.iinfo(np.uint64).max + 1.0)


def _predict_union_coverage_from_bls_tables(
    receiver: pd.DataFrame,
    *,
    self_employment_income: np.ndarray | None = None,
) -> np.ndarray:
    """Assign union coverage using official BLS annual rates.

    The public-use monthly CPS union variables appear unusable for
    donor imputation, so union coverage is assigned from BLS-published
    2024 state rates with demographic reweighting from the national
    characteristic tables.
    """
    result = np.zeros(len(receiver), dtype=np.float32)
    if len(receiver) == 0:
        return result

    employment_income = np.asarray(
        receiver["employment_income"],
        dtype=np.float32,
    )
    weekly_hours_worked = np.asarray(
        receiver["weekly_hours_worked"],
        dtype=np.float32,
    )
    age = np.asarray(receiver["age"], dtype=np.float32)
    eligible = (
        (employment_income > 0)
        & (weekly_hours_worked > 0)
        & (age >= 16)
    )
    if self_employment_income is not None:
        self_employment_income = np.asarray(
            self_employment_income,
            dtype=np.float32,
        )
        eligible &= ~(
            (self_employment_income > 0) & (employment_income <= 0)
        )

    if not eligible.any():
        return result

    state_fips = np.nan_to_num(
        np.asarray(receiver["state_fips"], dtype=np.float32),
        nan=-1,
    ).astype(np.int32)
    target_rates = _lookup_state_union_representation_rates(state_fips)
    priority_weights = _build_union_priority_weights(receiver)
    uniforms = np.clip(_stable_uniform_from_receiver(receiver), 1e-12, 1 - 1e-12)
    selection_keys = -np.log(uniforms) / priority_weights

    for state in np.unique(state_fips[eligible]):
        state_mask = eligible & (state_fips == state)
        state_idx = np.flatnonzero(state_mask)
        if len(state_idx) == 0:
            continue

        state_rate = float(target_rates[state_idx[0]])
        n_union = int(np.rint(state_rate * len(state_idx)))
        n_union = max(0, min(n_union, len(state_idx)))
        if n_union == 0:
            continue
        if n_union == len(state_idx):
            result[state_idx] = 1
            continue

        chosen = np.argpartition(selection_keys[state_idx], n_union - 1)[:n_union]
        result[state_idx[chosen]] = 1

    return result


@lru_cache(maxsize=1)
def load_org_training_data() -> pd.DataFrame:
    """Load ORG donor rows built from official CPS basic monthly files."""
    cache_path = STORAGE_FOLDER / ORG_FILENAME
    if cache_path.exists():
        return pd.read_csv(cache_path)

    months = []
    for month in ORG_MONTHS:
        month_df = pd.read_csv(
            _cps_basic_org_month_url(ORG_YEAR, month),
            usecols=CPS_BASIC_MONTHLY_ORG_COLUMNS,
            low_memory=False,
        )
        months.append(_transform_cps_basic_org_month(month_df))

    org = pd.concat(months, ignore_index=True)
    org.to_csv(cache_path, index=False, compression="gzip")
    return org


@lru_cache(maxsize=1)
def get_org_model():
    """Fit and cache the CPS-basic ORG model used for wage inputs."""
    train = load_org_training_data()
    qrf = QRF()
    return qrf.fit(
        X_train=train,
        predictors=ORG_PREDICTORS,
        imputed_variables=ORG_QRF_IMPUTED_VARIABLES,
        weight_col="sample_weight",
        tune_hyperparameters=False,
    )


def apply_org_domain_constraints(
    predictions: pd.DataFrame,
    receiver: pd.DataFrame,
    self_employment_income: np.ndarray | None = None,
) -> pd.DataFrame:
    """Clamp ORG predictions to basic labor-market domain constraints."""
    result = predictions.copy()

    inactive = (receiver["employment_income"].values <= 0) | (
        receiver["weekly_hours_worked"].values <= 0
    )
    if self_employment_income is not None:
        self_employment_income = np.asarray(self_employment_income)
        inactive |= (self_employment_income > 0) & (
            receiver["employment_income"].values <= 0
        )

    if "hourly_wage" in result:
        result["hourly_wage"] = result["hourly_wage"].clip(lower=0).astype(
            np.float32
        )
        result.loc[inactive, "hourly_wage"] = 0

    for col in ORG_BOOL_VARIABLES:
        if col in result:
            result[col] = result[col].fillna(0) >= 0.5
            result.loc[inactive, col] = False

    return result


def predict_org_features(
    receiver: pd.DataFrame,
    *,
    self_employment_income: np.ndarray | None = None,
) -> pd.DataFrame:
    """Predict ORG-derived labor-market features for receiver records."""
    missing = [col for col in ORG_PREDICTORS if col not in receiver.columns]
    if missing:
        raise ValueError(f"ORG receiver frame missing required columns: {missing}")

    predictions = get_org_model().predict(X_test=receiver[ORG_PREDICTORS])
    predictions["is_union_member_or_covered"] = (
        _predict_union_coverage_from_bls_tables(
            receiver,
            self_employment_income=self_employment_income,
        )
    )
    return apply_org_domain_constraints(
        predictions=predictions,
        receiver=receiver,
        self_employment_income=self_employment_income,
    )
