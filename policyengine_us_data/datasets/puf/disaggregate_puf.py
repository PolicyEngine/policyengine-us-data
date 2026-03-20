"""Disaggregate IRS aggregate PUF rows into synthetic extreme-tail donors.

The IRS PUF replaces a small set of returns with four aggregate rows
(`RECID` 999996-999999). Those rows are not ordinary AGI buckets: they are
returns excluded because one or more amount fields were extremely large, then
grouped by AGI. This module reconstructs them by:

1. Selecting non-aggregate donors from the same AGI bucket
2. Upweighting donors that are extreme in at least one screened amount field
3. Copying structural and flag variables directly from selected donors
4. Calibrating only amount fields to match the published aggregate totals
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from . import aggregate_record_utils as utils

AGGREGATE_RECIDS = utils.AGGREGATE_RECIDS
SYNTHETIC_RECID_START = utils.SYNTHETIC_RECID_START
_choose_n_synthetic = utils._choose_n_synthetic
_assign_weights = utils._assign_weights
_get_bucket_mask = utils._get_bucket_mask
_get_amount_columns = utils._get_amount_columns
_get_bucket_targets = utils._get_bucket_targets
_get_donor_bucket = utils._get_donor_bucket
_coerce_amount_columns = utils._coerce_amount_columns
compute_aggregate_eligibility_scores = utils.compute_aggregate_eligibility_scores
_project_weighted_sum_to_bounds = utils._project_weighted_sum_to_bounds
_allocate_weighted_values = utils._allocate_weighted_values
_allocate_agi_values = utils._allocate_agi_values
_selection_probabilities = utils._selection_probabilities
_sample_bucket_donors = utils._sample_bucket_donors
_apply_structural_templates = utils._apply_structural_templates
_calibrate_amount_columns = utils._calibrate_amount_columns
_disaggregate_bucket = utils._disaggregate_bucket


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def disaggregate_aggregate_records(
    puf: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Replace the four IRS aggregate rows with calibrated synthetic donors."""

    rng = np.random.default_rng(seed)

    agg_mask = puf.RECID.isin(AGGREGATE_RECIDS)
    if agg_mask.sum() == 0:
        return puf

    agg_rows = puf[agg_mask].copy().set_index("RECID")
    regular = puf[~agg_mask].copy()
    amount_columns = _get_amount_columns(puf.columns)
    donor_scores = compute_aggregate_eligibility_scores(regular)

    all_synthetic = []
    next_recid = SYNTHETIC_RECID_START

    for recid in AGGREGATE_RECIDS:
        synthetic = utils._disaggregate_bucket(
            recid=recid,
            row=agg_rows.loc[recid],
            regular=regular,
            amount_columns=amount_columns,
            donor_scores=donor_scores,
            next_recid=next_recid,
            rng=rng,
        )
        next_recid += len(synthetic)
        all_synthetic.append(synthetic[puf.columns])

    synthetic_df = pd.concat(all_synthetic, ignore_index=True)
    result = pd.concat([regular, synthetic_df], ignore_index=True)

    print(
        f"Disaggregated {int(agg_mask.sum())} aggregate records into "
        f"{len(synthetic_df)} synthetic records"
    )

    return result
