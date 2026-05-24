"""Collapsed SOCA-style basis and holding-period imputation.

This module creates one representative long-term capital-gains holding
period and cost basis per tax unit, then stores the result on people.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd


LONG_TERM_CAPITAL_GAINS_BASIS = "long_term_capital_gains_basis"
LONG_TERM_CAPITAL_GAINS_YEARS_HELD = "long_term_capital_gains_years_held"
CAPITAL_GAINS_BASIS_VARIABLES = (
    LONG_TERM_CAPITAL_GAINS_BASIS,
    LONG_TERM_CAPITAL_GAINS_YEARS_HELD,
)


@dataclass(frozen=True)
class CapitalGainsBasisResource:
    bucket_names: tuple[str, ...]
    bucket_lower_years: tuple[float, ...]
    bucket_upper_years: tuple[float, ...]
    bucket_midpoint_years: tuple[float, ...]
    gain_dollar_shares: tuple[float, ...]
    loss_dollar_shares: tuple[float, ...]
    gain_basis_sales_ratios: tuple[float, ...]
    loss_basis_sales_ratios: tuple[float, ...]
    weibull_shape: float = 0.7711
    weibull_scale: float = 9.1458
    gain_bsr_floor: float = 0.001
    gain_bsr_ceiling: float = 0.999
    loss_bsr_floor: float = 1.001
    loss_bsr_ceiling: float = 100.0


DEFAULT_SOCA_RESOURCE = CapitalGainsBasisResource(
    bucket_names=(
        "Under 18 months",
        "18 months under 2 years",
        "2 years under 3 years",
        "3 years under 4 years",
        "4 years under 5 years",
        "5 years under 10 years",
        "10 years under 15 years",
        "15 years under 20 years",
        "20 years or more",
    ),
    bucket_lower_years=(1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0),
    bucket_upper_years=(1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, np.inf),
    bucket_midpoint_years=(1.25, 1.75, 2.5, 3.5, 4.5, 7.5, 12.5, 17.5, 27.5),
    # IRS SOI Sales of Capital Assets, 2013-2015, as compacted by holding
    # period. Gain and loss shares are dollar-weighted within sign.
    gain_dollar_shares=(
        0.09265227810410295,
        0.07010752237381986,
        0.10520812743077781,
        0.08298825059272091,
        0.0743411463263887,
        0.20575715295741653,
        0.11737305049024667,
        0.0755711716188086,
        0.17600130010571793,
    ),
    loss_dollar_shares=(
        0.1482795960468827,
        0.10517359565806916,
        0.13462839298735996,
        0.09369439202769471,
        0.07359794203326578,
        0.29944369490176603,
        0.09122874216922709,
        0.03107240651916826,
        0.0228812376565664,
    ),
    gain_basis_sales_ratios=(
        0.8478328330650043,
        0.8160574582327029,
        0.8021013607528408,
        0.8060933128473603,
        0.7693845730205952,
        0.770253613744043,
        0.6358599451460517,
        0.5146618879371708,
        0.41336120762839623,
    ),
    loss_basis_sales_ratios=(
        1.1483448499553495,
        1.1815679561321597,
        1.2064462486658172,
        1.261228512659838,
        1.329629488793004,
        1.395990775722507,
        1.4840458650441617,
        1.636674383986131,
        1.63034354751029,
    ),
)


@dataclass(frozen=True)
class CapitalGainsBasisImputation:
    basis: np.ndarray
    years_held: np.ndarray
    holding_period_bucket: np.ndarray


def impute_tax_unit_long_term_capital_gains_basis(
    gains: np.ndarray,
    *,
    tax_unit_ids: np.ndarray,
    sample_weight: np.ndarray | None = None,
    tax_year: int = 2017,
    resource: CapitalGainsBasisResource = DEFAULT_SOCA_RESOURCE,
    imputation_version: str = "soca_collapsed_v1",
) -> CapitalGainsBasisImputation:
    """Impute collapsed basis and holding period for tax-unit gains.

    Args:
        gains: Net long-term capital gains by tax unit.
        tax_unit_ids: Stable tax-unit identifiers.
        sample_weight: Optional tax-unit weights for dollar-share quotas.
        tax_year: Sale year used only in deterministic keys for now.
        resource: Holding-period and basis-to-sales resource.
        imputation_version: Stable key salt.

    Returns:
        Basis, years held, and zero-based holding-period bucket arrays.
    """

    gains = np.asarray(gains, dtype=float)
    tax_unit_ids = np.asarray(tax_unit_ids)
    if gains.shape[0] != tax_unit_ids.shape[0]:
        raise ValueError("gains and tax_unit_ids must have the same length")

    if sample_weight is None:
        weights = np.ones_like(gains, dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape[0] != gains.shape[0]:
            raise ValueError("sample_weight must match gains length")
        if not np.any(weights > 0):
            weights = np.ones_like(gains, dtype=float)

    buckets = _assign_holding_period_buckets(
        gains,
        tax_unit_ids=tax_unit_ids,
        sample_weight=weights,
        tax_year=tax_year,
        resource=resource,
        imputation_version=imputation_version,
    )
    years_held = _draw_years_held(
        buckets,
        tax_unit_ids=tax_unit_ids,
        gains=gains,
        tax_year=tax_year,
        resource=resource,
        imputation_version=imputation_version,
    )
    basis = _basis_from_gains_and_years(gains, years_held, resource)

    zero_gain = gains == 0
    buckets = np.where(zero_gain, -1, buckets)
    years_held = np.where(zero_gain, 0.0, years_held)
    basis = np.where(zero_gain, 0.0, basis)
    return CapitalGainsBasisImputation(
        basis=basis,
        years_held=years_held,
        holding_period_bucket=buckets,
    )


def impute_person_level_long_term_capital_gains_basis(
    person_gains: np.ndarray,
    *,
    person_tax_unit_ids: np.ndarray,
    person_ids: np.ndarray | None = None,
    person_sample_weight: np.ndarray | None = None,
    tax_year: int = 2017,
    resource: CapitalGainsBasisResource = DEFAULT_SOCA_RESOURCE,
    imputation_version: str = "soca_collapsed_v1",
) -> CapitalGainsBasisImputation:
    """Impute tax-unit-collapsed basis and allocate it to people.

    The representative holding period is shared by every person with
    nonzero long-term gains in the tax unit. Basis is allocated by each
    person's absolute long-term gain so aggregation reproduces the
    collapsed tax-unit basis exactly.
    """

    person_gains = np.asarray(person_gains, dtype=float)
    person_tax_unit_ids = np.asarray(person_tax_unit_ids)
    if person_gains.shape[0] != person_tax_unit_ids.shape[0]:
        raise ValueError("person_gains and person_tax_unit_ids must match")

    if person_ids is None:
        person_ids = np.arange(person_gains.shape[0])
    else:
        person_ids = np.asarray(person_ids)
        if person_ids.shape[0] != person_gains.shape[0]:
            raise ValueError("person_ids must match person_gains length")

    frame = pd.DataFrame(
        {
            "person_gain": person_gains,
            "tax_unit_id": person_tax_unit_ids,
        }
    )
    if person_sample_weight is not None:
        sample_weight = np.asarray(person_sample_weight, dtype=float)
        if sample_weight.shape[0] != person_gains.shape[0]:
            raise ValueError("person_sample_weight must match person_gains length")
        frame["sample_weight"] = sample_weight
    else:
        frame["sample_weight"] = 1.0

    grouped = frame.groupby("tax_unit_id", sort=False).agg(
        gain=("person_gain", "sum"),
        sample_weight=("sample_weight", "max"),
    )
    tax_unit_imputation = impute_tax_unit_long_term_capital_gains_basis(
        grouped["gain"].to_numpy(),
        tax_unit_ids=grouped.index.to_numpy(),
        sample_weight=grouped["sample_weight"].to_numpy(),
        tax_year=tax_year,
        resource=resource,
        imputation_version=imputation_version,
    )

    basis_by_tax_unit = pd.Series(tax_unit_imputation.basis, index=grouped.index)
    years_by_tax_unit = pd.Series(tax_unit_imputation.years_held, index=grouped.index)
    bucket_by_tax_unit = pd.Series(
        tax_unit_imputation.holding_period_bucket,
        index=grouped.index,
    )
    abs_gain_sum = (
        frame.assign(abs_gain=np.abs(person_gains))
        .groupby("tax_unit_id", sort=False)["abs_gain"]
        .transform("sum")
    )
    tax_unit_basis = frame["tax_unit_id"].map(basis_by_tax_unit).to_numpy()
    tax_unit_years = frame["tax_unit_id"].map(years_by_tax_unit).to_numpy()
    tax_unit_buckets = frame["tax_unit_id"].map(bucket_by_tax_unit).to_numpy()

    abs_person_gain = np.abs(person_gains)
    basis = np.divide(
        tax_unit_basis * abs_person_gain,
        abs_gain_sum.to_numpy(),
        out=np.zeros_like(person_gains, dtype=float),
        where=abs_gain_sum.to_numpy() > 0,
    )
    years_held = np.where(abs_person_gain > 0, tax_unit_years, 0.0)
    buckets = np.where(abs_person_gain > 0, tax_unit_buckets, -1)
    return CapitalGainsBasisImputation(
        basis=basis,
        years_held=years_held,
        holding_period_bucket=buckets,
    )


def add_long_term_capital_gains_basis_to_puf_frame(
    puf: pd.DataFrame,
    *,
    tax_year: int = 2017,
    resource: CapitalGainsBasisResource = DEFAULT_SOCA_RESOURCE,
) -> pd.DataFrame:
    """Add collapsed basis and holding period columns to a PUF frame."""

    if "long_term_capital_gains" not in puf:
        return puf
    record_ids = puf["RECID"].to_numpy() if "RECID" in puf else puf.index.to_numpy()
    weights = puf["S006"].to_numpy() if "S006" in puf else None
    imputation = impute_tax_unit_long_term_capital_gains_basis(
        puf["long_term_capital_gains"].to_numpy(),
        tax_unit_ids=record_ids,
        sample_weight=weights,
        tax_year=tax_year,
        resource=resource,
    )
    puf[LONG_TERM_CAPITAL_GAINS_BASIS] = imputation.basis
    puf[LONG_TERM_CAPITAL_GAINS_YEARS_HELD] = imputation.years_held
    return puf


def _assign_holding_period_buckets(
    gains: np.ndarray,
    *,
    tax_unit_ids: np.ndarray,
    sample_weight: np.ndarray,
    tax_year: int,
    resource: CapitalGainsBasisResource,
    imputation_version: str,
) -> np.ndarray:
    buckets = np.full(gains.shape[0], -1, dtype=int)
    for sign, probabilities, label in (
        (1, resource.gain_dollar_shares, "gain"),
        (-1, resource.loss_dollar_shares, "loss"),
    ):
        mask = gains * sign > 0
        if not np.any(mask):
            continue

        probabilities_array = _normalise_probabilities(probabilities)
        masked_indices = np.flatnonzero(mask)
        dollar_weights = np.abs(gains[mask]) * sample_weight[mask]
        keys = _stable_uniforms(
            tax_unit_ids[mask],
            salt=f"{imputation_version}|{tax_year}|bucket|{label}",
        )
        if dollar_weights.sum() <= 0:
            assigned = np.searchsorted(
                np.cumsum(probabilities_array),
                keys,
                side="right",
            )
            buckets[masked_indices] = np.minimum(
                assigned,
                len(probabilities_array) - 1,
            )
            continue

        order = np.argsort(keys, kind="mergesort")
        sorted_indices = masked_indices[order]
        sorted_weights = dollar_weights[order]
        weighted_midpoints = (
            np.cumsum(sorted_weights) - 0.5 * sorted_weights
        ) / sorted_weights.sum()
        assigned = np.searchsorted(
            np.cumsum(probabilities_array),
            weighted_midpoints,
            side="right",
        )
        buckets[sorted_indices] = np.minimum(assigned, len(probabilities_array) - 1)
    return buckets


def _draw_years_held(
    buckets: np.ndarray,
    *,
    tax_unit_ids: np.ndarray,
    gains: np.ndarray,
    tax_year: int,
    resource: CapitalGainsBasisResource,
    imputation_version: str,
) -> np.ndarray:
    years = np.zeros_like(gains, dtype=float)
    for bucket in range(len(resource.bucket_names)):
        mask = buckets == bucket
        if not np.any(mask):
            continue
        signs = np.where(gains[mask] > 0, "gain", "loss")
        salts = [
            f"{imputation_version}|{tax_year}|years|{sign}|{bucket}" for sign in signs
        ]
        uniforms = np.array(
            [
                _stable_uniform(record_id, salt=salt)
                for record_id, salt in zip(tax_unit_ids[mask], salts)
            ],
            dtype=float,
        )
        years[mask] = _draw_years_in_bucket(bucket, uniforms, resource)
    return years


def _draw_years_in_bucket(
    bucket: int,
    uniforms: np.ndarray,
    resource: CapitalGainsBasisResource,
) -> np.ndarray:
    lo = resource.bucket_lower_years[bucket]
    hi = resource.bucket_upper_years[bucket]
    if lo >= 20:
        return 20 + (-np.log1p(-uniforms) / _top_bucket_exponential_rate(resource))

    x_lo = lo - 1
    x_hi = hi - 1
    f_lo = _weibull_cdf(x_lo, resource.weibull_shape, resource.weibull_scale)
    f_hi = _weibull_cdf(x_hi, resource.weibull_shape, resource.weibull_scale)
    u = f_lo + uniforms * (f_hi - f_lo)
    return 1 + _weibull_quantile(u, resource.weibull_shape, resource.weibull_scale)


def _basis_from_gains_and_years(
    gains: np.ndarray,
    years_held: np.ndarray,
    resource: CapitalGainsBasisResource,
) -> np.ndarray:
    basis = np.zeros_like(gains, dtype=float)
    positive = gains > 0
    negative = gains < 0

    gain_bsr = _gain_basis_sales_ratio(years_held[positive], resource)
    basis[positive] = np.abs(gains[positive]) * gain_bsr / (1 - gain_bsr)

    loss_bsr = _loss_basis_sales_ratio(years_held[negative], resource)
    basis[negative] = np.abs(gains[negative]) * loss_bsr / (loss_bsr - 1)
    return basis


def _gain_basis_sales_ratio(
    years_held: np.ndarray,
    resource: CapitalGainsBasisResource,
) -> np.ndarray:
    h_top = _top_bucket_mean(resource)
    h_knots = np.asarray(resource.bucket_midpoint_years, dtype=float)
    h_knots[-1] = h_top
    ratio_knots = np.asarray(resource.gain_basis_sales_ratios, dtype=float)
    interpolated = np.interp(
        np.minimum(years_held, h_top),
        h_knots,
        ratio_knots,
        left=ratio_knots[0],
        right=ratio_knots[-1],
    )
    g_extrap = (1 / ratio_knots[-1]) ** (1 / h_top) - 1
    extrapolated = 1 / (1 + g_extrap) ** years_held
    ratio = np.where(years_held <= h_top, interpolated, extrapolated)
    return np.clip(ratio, resource.gain_bsr_floor, resource.gain_bsr_ceiling)


def _loss_basis_sales_ratio(
    years_held: np.ndarray,
    resource: CapitalGainsBasisResource,
) -> np.ndarray:
    h_knots = np.asarray(resource.bucket_midpoint_years, dtype=float)
    h_knots[-1] = _top_bucket_mean(resource)
    ratio_knots = np.asarray(resource.loss_basis_sales_ratios, dtype=float)
    ratio = np.interp(
        years_held,
        h_knots,
        ratio_knots,
        left=ratio_knots[0],
        right=ratio_knots[-1],
    )
    return np.clip(ratio, resource.loss_bsr_floor, resource.loss_bsr_ceiling)


def _top_bucket_mean(resource: CapitalGainsBasisResource) -> float:
    return 20 + 1 / _top_bucket_exponential_rate(resource)


def _top_bucket_exponential_rate(resource: CapitalGainsBasisResource) -> float:
    density_at_boundary = _weibull_pdf(
        19,
        resource.weibull_shape,
        resource.weibull_scale,
    )
    bucket_8_mass = _weibull_cdf(
        19,
        resource.weibull_shape,
        resource.weibull_scale,
    ) - _weibull_cdf(14, resource.weibull_shape, resource.weibull_scale)
    gain_shares = np.asarray(resource.gain_dollar_shares, dtype=float)
    return density_at_boundary / bucket_8_mass * gain_shares[7] / gain_shares[8]


def _weibull_cdf(x: float, shape: float, scale: float) -> float:
    return 1 - np.exp(-((x / scale) ** shape))


def _weibull_pdf(x: float, shape: float, scale: float) -> float:
    return (
        (shape / scale) * ((x / scale) ** (shape - 1)) * np.exp(-((x / scale) ** shape))
    )


def _weibull_quantile(u: np.ndarray, shape: float, scale: float) -> np.ndarray:
    u = np.clip(u, np.finfo(float).tiny, np.nextafter(1.0, 0.0))
    return scale * (-np.log1p(-u)) ** (1 / shape)


def _normalise_probabilities(probabilities: tuple[float, ...]) -> np.ndarray:
    probabilities_array = np.asarray(probabilities, dtype=float)
    total = probabilities_array.sum()
    if total <= 0:
        raise ValueError("holding-period probabilities must sum to a positive value")
    return probabilities_array / total


def _stable_uniforms(values: np.ndarray, *, salt: str) -> np.ndarray:
    return np.array(
        [_stable_uniform(value, salt=salt) for value in values], dtype=float
    )


def _stable_uniform(value, *, salt: str) -> float:
    digest = hashlib.blake2b(
        f"{salt}|{value}".encode("utf-8"),
        digest_size=8,
    ).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return (integer + 0.5) / 2**64
