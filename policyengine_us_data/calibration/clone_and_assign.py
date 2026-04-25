"""Clone CPS records and assign random geography."""

import logging
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
import pandas as pd

from policyengine_us_data.storage import STORAGE_FOLDER

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


def assign_random_geography(
    n_records: int,
    n_clones: int = 10,
    seed: int = 42,
    household_agi: np.ndarray = None,
    cd_agi_targets: dict = None,
    agi_threshold_pctile: float = 90.0,
    fixed_state_fips: np.ndarray = None,
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
        fixed_state_fips: Optional state FIPS per base record. Positive
            values constrain every clone of that record to blocks in the
            requested state; zero or missing values remain unrestricted.

    Returns:
        GeographyAssignment with arrays of length
        n_records * n_clones.
    """
    blocks, cds, states, probs = load_global_block_distribution()
    fixed_states = _validate_fixed_state_fips(
        fixed_state_fips,
        n_records=n_records,
        available_states=states,
    )

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

    state_draw_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

    def _state_draw_inputs(state: int, probability_source: str):
        key = (int(state), probability_source)
        cached = state_draw_cache.get(key)
        if cached is not None:
            return cached

        state_indices = np.flatnonzero(states == state)
        base_probs = agi_probs if probability_source == "agi" else probs
        state_probs = base_probs[state_indices].astype(np.float64)
        if not np.isfinite(state_probs).all() or state_probs.sum() <= 0:
            state_probs = probs[state_indices].astype(np.float64)
        if not np.isfinite(state_probs).all() or state_probs.sum() <= 0:
            state_probs = np.ones(len(state_indices), dtype=np.float64)
        state_probs = state_probs / state_probs.sum()
        state_draw_cache[key] = (state_indices, state_probs)
        return state_indices, state_probs

    def _sample_state(state: int, size: int, probability_source: str):
        state_indices, state_probs = _state_draw_inputs(state, probability_source)
        return rng.choice(state_indices, size=size, p=state_probs)

    def _sample_unrestricted(size, mask_slice=None):
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

    def _sample(size, mask_slice=None, fixed_slice=None):
        out = np.empty(size, dtype=np.int64)
        remaining = np.ones(size, dtype=bool)

        if fixed_slice is not None:
            fixed_slice = np.asarray(fixed_slice, dtype=np.int32)
            for state in np.unique(fixed_slice[fixed_slice > 0]):
                state_mask = fixed_slice == state
                if mask_slice is not None and agi_probs is not None:
                    extreme_state_mask = state_mask & mask_slice
                    normal_state_mask = state_mask & ~mask_slice
                    if extreme_state_mask.any():
                        out[extreme_state_mask] = _sample_state(
                            int(state),
                            int(extreme_state_mask.sum()),
                            "agi",
                        )
                    if normal_state_mask.any():
                        out[normal_state_mask] = _sample_state(
                            int(state),
                            int(normal_state_mask.sum()),
                            "pop",
                        )
                else:
                    out[state_mask] = _sample_state(
                        int(state),
                        int(state_mask.sum()),
                        "pop",
                    )
                remaining[state_mask] = False

        if remaining.any():
            remaining_mask = mask_slice[remaining] if mask_slice is not None else None
            out[remaining] = _sample_unrestricted(int(remaining.sum()), remaining_mask)
        return out

    indices = np.empty(n_total, dtype=np.int64)

    # Clone 0: unrestricted draw
    indices[:n_records] = _sample(n_records, extreme_mask, fixed_states)

    assigned_cds = np.empty((n_clones, n_records), dtype=object)
    assigned_cds[0] = cds[indices[:n_records]]

    for clone_idx in range(1, n_clones):
        start = clone_idx * n_records
        clone_indices = _sample(n_records, extreme_mask, fixed_states)
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
                replacement = _sample(n_records, extreme_mask, fixed_states)
                clone_indices[bad_mask] = replacement[bad_mask]
            else:
                replacement = _sample(n_records, fixed_slice=fixed_states)
                clone_indices[collisions] = replacement[collisions]
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


def _validate_fixed_state_fips(
    fixed_state_fips: np.ndarray | None,
    n_records: int,
    available_states: np.ndarray,
) -> np.ndarray | None:
    """Validate optional record-level state constraints."""

    if fixed_state_fips is None:
        return None

    fixed = np.asarray(fixed_state_fips)
    if len(fixed) != n_records:
        raise ValueError(
            "fixed_state_fips must have one value per base record: "
            f"got {len(fixed)} for {n_records} records."
        )

    fixed = np.nan_to_num(fixed.astype(float), nan=0.0).astype(np.int32)
    positive = np.unique(fixed[fixed > 0])
    if len(positive) == 0:
        return None

    available = set(np.asarray(available_states, dtype=np.int32).tolist())
    missing = [int(state) for state in positive if int(state) not in available]
    if missing:
        raise ValueError(
            "fixed_state_fips contains states absent from the block "
            f"distribution: {missing}"
        )

    logger.info(
        "Preserving fixed state geography for %d of %d records",
        int((fixed > 0).sum()),
        n_records,
    )
    return fixed


def save_geography(geography: GeographyAssignment, path) -> None:
    """Save a GeographyAssignment to a compressed .npz file.

    Args:
        geography: The geography assignment to save.
        path: Output file path (should end in .npz).
    """
    from pathlib import Path

    path = Path(path)
    np.savez_compressed(
        path,
        block_geoid=geography.block_geoid,
        cd_geoid=geography.cd_geoid,
        county_fips=geography.county_fips,
        state_fips=geography.state_fips,
        n_records=np.array([geography.n_records]),
        n_clones=np.array([geography.n_clones]),
    )


def load_geography(path) -> GeographyAssignment:
    """Load a GeographyAssignment from a .npz file.

    Args:
        path: Path to the .npz file saved by save_geography.

    Returns:
        GeographyAssignment with all fields restored.
    """
    from pathlib import Path

    path = Path(path)
    data = np.load(path, allow_pickle=True)
    return GeographyAssignment(
        block_geoid=data["block_geoid"],
        cd_geoid=data["cd_geoid"],
        county_fips=data["county_fips"],
        state_fips=data["state_fips"],
        n_records=int(data["n_records"][0]),
        n_clones=int(data["n_clones"][0]),
    )


@lru_cache(maxsize=1)
def load_sorted_block_cd_lookup():
    """Load a sorted block -> CD lookup for legacy block artifacts."""
    blocks, cds, _, _ = load_global_block_distribution()
    order = np.argsort(blocks)
    return blocks[order], cds[order]


def reconstruct_geography_from_blocks(
    block_geoids: np.ndarray,
    n_records: int,
    n_clones: int,
) -> GeographyAssignment:
    """Reconstruct a GeographyAssignment from saved block GEOIDs."""
    block_geoids = np.asarray(block_geoids, dtype=str)
    expected_len = n_records * n_clones
    if len(block_geoids) != expected_len:
        raise ValueError(
            f"Expected {expected_len} block GEOIDs for "
            f"{n_records} records x {n_clones} clones, got {len(block_geoids)}"
        )

    sorted_blocks, sorted_cds = load_sorted_block_cd_lookup()
    indices = np.searchsorted(sorted_blocks, block_geoids)
    valid = indices < len(sorted_blocks)
    matched = np.zeros(len(block_geoids), dtype=bool)
    matched[valid] = sorted_blocks[indices[valid]] == block_geoids[valid]

    if not np.all(matched):
        missing = np.unique(block_geoids[~matched])[:5]
        raise KeyError(
            "Could not recover congressional districts for some blocks. "
            f"Examples: {missing.tolist()}"
        )

    county_fips = np.fromiter(
        (block[:5] for block in block_geoids),
        dtype="U5",
        count=len(block_geoids),
    )
    state_fips = np.fromiter(
        (int(block[:2]) for block in block_geoids),
        dtype=np.int32,
        count=len(block_geoids),
    )
    return GeographyAssignment(
        block_geoid=block_geoids,
        cd_geoid=sorted_cds[indices],
        county_fips=county_fips,
        state_fips=state_fips,
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
