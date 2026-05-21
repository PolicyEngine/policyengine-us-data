"""Utilities for training imputations from observed donor-survey targets."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def sipp_allocation_flag_for(source_column: str) -> str:
    """Return the SIPP allocation flag name for a source variable."""
    if not source_column:
        raise ValueError("source_column must be non-empty")
    return f"A{source_column[1:]}"


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
