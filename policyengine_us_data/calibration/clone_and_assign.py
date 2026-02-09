"""Clone extended CPS records and assign random geography."""

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
            f"{csv_path} not found. "
            "Run make_block_cd_distributions.py to generate."
        )

    df = pd.read_csv(csv_path, dtype={"block_geoid": str})

    block_geoids = df["block_geoid"].values
    cd_geoids = df["cd_geoid"].astype(str).values
    # State FIPS is first 2 digits of block GEOID
    state_fips = np.array([int(b[:2]) for b in block_geoids])

    probs = df["probability"].values.astype(np.float64)
    probs = probs / probs.sum()  # Normalize globally

    return block_geoids, cd_geoids, state_fips, probs


def assign_random_geography(
    n_records: int,
    n_clones: int = 130,
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
        n_clones: Number of clones (default 130).
        seed: Random seed for reproducibility.

    Returns:
        GeographyAssignment with arrays of length
        n_records * n_clones.
    """
    blocks, cds, states, probs = load_global_block_distribution()

    n_total = n_records * n_clones
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(blocks), size=n_total, p=probs)

    return GeographyAssignment(
        block_geoid=blocks[indices],
        cd_geoid=cds[indices],
        state_fips=states[indices],
        n_records=n_records,
        n_clones=n_clones,
    )
