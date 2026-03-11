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

    block_geoids = df["block_geoid"].values
    cd_geoids = np.array(df["cd_geoid"].astype(str).tolist())
    state_fips = np.array([int(b[:2]) for b in block_geoids])

    probs = df["probability"].values.astype(np.float64)
    probs = probs / probs.sum()

    return block_geoids, cd_geoids, state_fips, probs


def assign_random_geography(
    n_records: int,
    n_clones: int = 10,
    seed: int = 42,
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

    indices = np.empty(n_total, dtype=np.int64)

    # Clone 0: unrestricted draw
    indices[:n_records] = rng.choice(len(blocks), size=n_records, p=probs)

    assigned_cds = np.empty((n_clones, n_records), dtype=object)
    assigned_cds[0] = cds[indices[:n_records]]

    for clone_idx in range(1, n_clones):
        start = clone_idx * n_records
        clone_indices = rng.choice(len(blocks), size=n_records, p=probs)
        clone_cds = cds[clone_indices]

        collisions = np.zeros(n_records, dtype=bool)
        for prev in range(clone_idx):
            collisions |= clone_cds == assigned_cds[prev]

        for _ in range(50):
            n_bad = collisions.sum()
            if n_bad == 0:
                break
            clone_indices[collisions] = rng.choice(
                len(blocks), size=n_bad, p=probs
            )
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
    data = np.load(path, allow_pickle=False)
    return GeographyAssignment(
        block_geoid=data["block_geoid"],
        cd_geoid=data["cd_geoid"],
        county_fips=data["county_fips"],
        state_fips=data["state_fips"],
        n_records=int(data["n_records"][0]),
        n_clones=int(data["n_clones"][0]),
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
