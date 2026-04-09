"""Clone CPS records and assign random geography."""

import logging
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
import pandas as pd

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode

logger = logging.getLogger(__name__)


@dataclass
class GeographyAssignment:
    """Random geography assignment for cloned CPS records.

    All arrays have length n_records * n_clones.
    Index i corresponds to clone i // n_records,
    record i % n_records.
    """

    block_geoid: np.ndarray  # str array, 15-char block GEOIDs
    cd_geoid: np.ndarray  # str array of CD GEOIDs
    county_fips: np.ndarray  # str array of 5-char county FIPS
    state_fips: np.ndarray  # int array of 2-digit state FIPS
    n_records: int
    n_clones: int


@lru_cache(maxsize=1)
def load_global_block_distribution():
    """Load block_cd_distributions.csv.gz and build
    global distribution.

    Returns:
        Tuple of (block_geoids, cd_geoids, state_fips,
        probabilities) where each is a numpy array indexed
        by block row. Probabilities are normalized to sum
        to 1 globally.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_path = STORAGE_FOLDER / "block_cd_distributions.csv.gz"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run make_block_cd_distributions.py to generate."
        )

    df = pd.read_csv(csv_path, dtype={"block_geoid": str})

    # Normalize at-large districts: Census uses 00 (and 98 for DC) → 01
    district_num = df["cd_geoid"] % 100
    state_fips_col = df["cd_geoid"] // 100
    at_large = (district_num == 0) | ((state_fips_col == 11) & (district_num == 98))
    df.loc[at_large, "cd_geoid"] = state_fips_col[at_large] * 100 + 1

    block_geoids = df["block_geoid"].values
    cd_geoids = np.array(df["cd_geoid"].astype(str).tolist())
    state_fips = np.array([int(b[:2]) for b in block_geoids])

    probs = df["probability"].values.astype(np.float64)
    probs = probs / probs.sum()

    return block_geoids, cd_geoids, state_fips, probs


def _build_agi_block_probs(cds, pop_probs, cd_agi_targets):
    """Reweight block probabilities to match district AGI target shares.

    District totals should be proportional to ``cd_agi_targets``, while
    block shares within each district should preserve the original
    population-weighted distribution.
    """
    agi_weights = np.array([cd_agi_targets.get(cd, 0.0) for cd in cds])
    agi_weights = np.maximum(agi_weights, 0.0)
    if agi_weights.sum() == 0:
        return pop_probs

    district_pop_mass = (
        pd.Series(pop_probs, copy=False).groupby(cds).transform("sum").to_numpy()
    )
    agi_probs = np.divide(
        pop_probs * agi_weights,
        district_pop_mass,
        out=np.zeros_like(pop_probs, dtype=np.float64),
        where=district_pop_mass > 0,
    )
    if agi_probs.sum() == 0:
        return pop_probs
    return agi_probs / agi_probs.sum()


@pipeline_node(
    PipelineNode(
        id="geo_assign_s4",
        label="Geography Assignment",
        node_type="process",
        description="Population-weighted block assignment with AGI-conditioned reweighting for top-income households",
        details="Preserves within-district block shares while reweighting extreme-household draws to district AGI targets",
        source_file="policyengine_us_data/calibration/clone_and_assign.py",
    )
)
def assign_random_geography(
    n_records: int,
    n_clones: int = 10,
    seed: int = 42,
    household_agi: np.ndarray = None,
    cd_agi_targets: dict = None,
    agi_threshold_pctile: float = 90.0,
) -> GeographyAssignment:
    """Assign random census block geography to cloned
    CPS records.

    Each of n_records * n_clones total records gets a
    random census block sampled from the global
    population-weighted distribution. State and CD are
    derived from the block GEOID.

    Args:
        n_records: Number of households in the base CPS
            dataset.
        n_clones: Number of clones (default 10).
        seed: Random seed for reproducibility.

    Returns:
        GeographyAssignment with arrays of length
        n_records * n_clones.
    """
    blocks, cds, states, probs = load_global_block_distribution()

    n_total = n_records * n_clones
    rng = np.random.default_rng(seed)

    agi_probs = None
    extreme_mask = None
    if household_agi is not None and cd_agi_targets is not None:
        threshold = np.percentile(household_agi, agi_threshold_pctile)
        extreme_mask = household_agi >= threshold
        agi_probs = _build_agi_block_probs(cds, probs, cd_agi_targets)
        logger.info(
            "AGI-conditional assignment: %d extreme HHs (AGI >= $%.0f) "
            "use AGI-weighted block probs",
            extreme_mask.sum(),
            threshold,
        )

    def _sample(size, mask_slice=None):
        """Sample block indices, using AGI-weighted probs for extreme HHs."""
        if (
            extreme_mask is not None
            and agi_probs is not None
            and mask_slice is not None
        ):
            out = np.empty(size, dtype=np.int64)
            ext = mask_slice
            n_ext = ext.sum()
            n_norm = size - n_ext
            if n_ext > 0:
                out[ext] = rng.choice(len(blocks), size=n_ext, p=agi_probs)
            if n_norm > 0:
                out[~ext] = rng.choice(len(blocks), size=n_norm, p=probs)
            return out
        return rng.choice(len(blocks), size=size, p=probs)

    indices = np.empty(n_total, dtype=np.int64)

    # Clone 0: unrestricted draw
    indices[:n_records] = _sample(n_records, extreme_mask)

    assigned_cds = np.empty((n_clones, n_records), dtype=object)
    assigned_cds[0] = cds[indices[:n_records]]

    for clone_idx in range(1, n_clones):
        start = clone_idx * n_records
        clone_indices = _sample(n_records, extreme_mask)
        clone_cds = cds[clone_indices]

        collisions = np.zeros(n_records, dtype=bool)
        for prev in range(clone_idx):
            collisions |= clone_cds == assigned_cds[prev]

        for _ in range(50):
            n_bad = collisions.sum()
            if n_bad == 0:
                break
            bad_mask = collisions
            if extreme_mask is not None and agi_probs is not None:
                bad_ext = bad_mask & extreme_mask
                bad_norm = bad_mask & ~extreme_mask
                if bad_ext.sum() > 0:
                    clone_indices[bad_ext] = rng.choice(
                        len(blocks), size=bad_ext.sum(), p=agi_probs
                    )
                if bad_norm.sum() > 0:
                    clone_indices[bad_norm] = rng.choice(
                        len(blocks), size=bad_norm.sum(), p=probs
                    )
            else:
                clone_indices[collisions] = rng.choice(len(blocks), size=n_bad, p=probs)
            clone_cds = cds[clone_indices]
            collisions = np.zeros(n_records, dtype=bool)
            for prev in range(clone_idx):
                collisions |= clone_cds == assigned_cds[prev]

        indices[start : start + n_records] = clone_indices
        assigned_cds[clone_idx] = clone_cds

    assigned_blocks = blocks[indices]
    return GeographyAssignment(
        block_geoid=assigned_blocks,
        cd_geoid=cds[indices],
        county_fips=np.array([b[:5] for b in assigned_blocks]),
        state_fips=states[indices],
        n_records=n_records,
        n_clones=n_clones,
    )


def double_geography_for_puf(
    geography: GeographyAssignment,
) -> GeographyAssignment:
    """Double geography arrays for PUF clone step.

    After PUF cloning doubles the base records, the geography
    assignment must also double: each record and its PUF copy
    share the same geographic assignment.

    The output has n_records = 2 * geography.n_records, with
    the first half being the CPS records and the second half
    being the PUF copies.

    Args:
        geography: Original geography assignment.

    Returns:
        New GeographyAssignment with doubled n_records.
    """
    n_old = geography.n_records
    n_new = n_old * 2
    n_clones = geography.n_clones

    new_blocks = []
    new_cds = []
    new_counties = []
    new_states = []

    for c in range(n_clones):
        start = c * n_old
        end = start + n_old
        clone_blocks = geography.block_geoid[start:end]
        clone_cds = geography.cd_geoid[start:end]
        clone_counties = geography.county_fips[start:end]
        clone_states = geography.state_fips[start:end]
        new_blocks.append(np.concatenate([clone_blocks, clone_blocks]))
        new_cds.append(np.concatenate([clone_cds, clone_cds]))
        new_counties.append(np.concatenate([clone_counties, clone_counties]))
        new_states.append(np.concatenate([clone_states, clone_states]))

    return GeographyAssignment(
        block_geoid=np.concatenate(new_blocks),
        cd_geoid=np.concatenate(new_cds),
        county_fips=np.concatenate(new_counties),
        state_fips=np.concatenate(new_states),
        n_records=n_new,
        n_clones=n_clones,
    )
