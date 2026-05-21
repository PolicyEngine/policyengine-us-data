"""Utilities for training imputations from observed donor-survey targets."""

from __future__ import annotations

import logging
from collections.abc import Container, Mapping, Sequence

import numpy as np
import pandas as pd

from policyengine_us_data.utils.randomness import seeded_rng

logger = logging.getLogger(__name__)


def sipp_allocation_flag_for(source_column: str) -> str:
    """Return the SIPP allocation flag name for a source variable."""
    if not source_column:
        raise ValueError("source_column must be non-empty")
    return f"A{source_column[1:]}"


def require_columns_present(
    available_columns: Container[str],
    required_columns: Sequence[str],
    *,
    source_name: str,
) -> None:
    """Raise if required donor-source provenance columns are unavailable."""
    missing_columns = sorted(
        {column for column in required_columns if column not in available_columns}
    )
    if missing_columns:
        raise KeyError(
            f"{source_name} is missing required source-quality columns: "
            f"{', '.join(missing_columns)}. Regenerate the donor artifact with "
            "allocation flag columns before fitting source imputations."
        )


def observed_source_mask(
    df: pd.DataFrame,
    *,
    source_columns: Sequence[str] = (),
    allocation_flag_columns: Sequence[str] = (),
    require_nonmissing_source: bool = True,
) -> pd.Series:
    """Mask rows whose donor source values are observed for one target.

    Source-survey allocation flags conventionally use ``0`` for not allocated
    and non-zero values for allocated/imputed. Missing flag columns are ignored
    so callers can use this helper across sources with different flag coverage.
    """
    mask = pd.Series(True, index=df.index)

    if require_nonmissing_source:
        for column in source_columns:
            if column in df:
                mask &= df[column].notna()

    for column in allocation_flag_columns:
        if column not in df:
            continue
        flag = pd.to_numeric(df[column], errors="coerce").fillna(0)
        mask &= flag.eq(0)

    return mask


def filter_observed_source_rows(
    df: pd.DataFrame,
    *,
    target_name: str,
    source_columns: Sequence[str] = (),
    allocation_flag_columns: Sequence[str] = (),
    require_nonmissing_source: bool = True,
) -> pd.DataFrame:
    """Return rows whose source values are observed for ``target_name``."""
    mask = observed_source_mask(
        df,
        source_columns=source_columns,
        allocation_flag_columns=allocation_flag_columns,
        require_nonmissing_source=require_nonmissing_source,
    )
    dropped = int((~mask).sum())
    if dropped:
        logger.info(
            "Dropped %d/%d donor rows with imputed source values for %s",
            dropped,
            len(df),
            target_name,
        )
    return df.loc[mask].copy()


def target_observed_source_masks(
    df: pd.DataFrame,
    *,
    targets: Sequence[str],
    target_source_columns: Mapping[str, Sequence[str]] | None = None,
    target_allocation_flag_columns: Mapping[str, Sequence[str]] | None = None,
    require_nonmissing_source: bool = True,
) -> dict[str, pd.Series]:
    """Return target-specific masks for rows with observed source values.

    The masks can be passed directly to microimpute models that support
    ``target_filters``. Source values are checked per target so a row with an
    allocated value for one target can still train another target whose source
    value is observed.
    """
    target_source_columns = target_source_columns or {}
    target_allocation_flag_columns = target_allocation_flag_columns or {}

    masks = {}
    for target in targets:
        masks[target] = observed_source_mask(
            df,
            source_columns=target_source_columns.get(target, (target,)),
            allocation_flag_columns=target_allocation_flag_columns.get(target, ()),
            require_nonmissing_source=require_nonmissing_source,
        )
        if not masks[target].any():
            raise ValueError(f"No observed donor rows available for {target}")
        dropped = int((~masks[target]).sum())
        if dropped:
            logger.info(
                "Target %s has %d observed donor rows; excluded %d rows",
                target,
                int(masks[target].sum()),
                dropped,
            )

    return masks


def cap_training_sample(
    df: pd.DataFrame,
    *,
    max_train_samples: int,
    seed_name: str,
    target_filters: Mapping[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Deterministically cap a target-filtered QRF training frame.

    microimpute's QRF ``max_train_samples`` currently does not subsample when
    ``target_filters`` are present, because target-specific masks must remain
    aligned with each target's training rows. This helper applies the same
    positional, without-replacement cap locally and returns reindexed masks.
    """
    if max_train_samples < 1:
        raise ValueError("max_train_samples must be a positive integer")

    filters = {}
    for target, mask in (target_filters or {}).items():
        aligned = mask.reindex(df.index)
        if aligned.isna().any():
            raise ValueError(f"target_filters[{target!r}] contains missing values")
        filters[target] = aligned.astype(bool)

    if not filters:
        if len(df) <= max_train_samples:
            return df, filters
        sample_positions = seeded_rng(seed_name).choice(
            len(df),
            size=max_train_samples,
            replace=False,
        )
    else:
        if max_train_samples < len(filters):
            raise ValueError(
                "max_train_samples must be at least the number of target filters"
            )
        union_mask = pd.Series(False, index=df.index)
        for mask in filters.values():
            union_mask |= mask
        if not union_mask.any():
            raise ValueError("No observed donor rows available across target_filters")

        union_positions = np.flatnonzero(union_mask.to_numpy())
        if len(union_positions) <= max_train_samples:
            sample_positions = union_positions
        else:
            selected: list[int] = []
            selected_set: set[int] = set()
            per_target_cap = max(1, max_train_samples // len(filters))
            for target, mask in filters.items():
                target_positions = np.flatnonzero(mask.to_numpy())
                target_n = min(per_target_cap, len(target_positions))
                target_sample = seeded_rng(seed_name, salt=target).choice(
                    target_positions,
                    size=target_n,
                    replace=False,
                )
                for position in target_sample:
                    if int(position) not in selected_set:
                        selected.append(int(position))
                        selected_set.add(int(position))

            remaining_n = max_train_samples - len(selected)
            if remaining_n > 0:
                remaining_positions = np.array(
                    [
                        position
                        for position in union_positions
                        if int(position) not in selected_set
                    ],
                    dtype=int,
                )
                if len(remaining_positions):
                    fill_sample = seeded_rng(seed_name, salt="fill").choice(
                        remaining_positions,
                        size=min(remaining_n, len(remaining_positions)),
                        replace=False,
                    )
                    selected.extend(int(position) for position in fill_sample)

            sample_positions = np.asarray(selected, dtype=int)

    sampled_df = df.iloc[sample_positions].copy().reset_index(drop=True)
    sampled_filters = {
        target: pd.Series(
            np.asarray(mask.iloc[sample_positions], dtype=bool),
            index=sampled_df.index,
        )
        for target, mask in filters.items()
    }
    return sampled_df, sampled_filters
