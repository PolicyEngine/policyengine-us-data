# build_h5 — Unified H5 Builder

`build_h5` is the single function that produces all local-area H5 datasets (national, state, district, city). It lives in `policyengine_us_data/calibration/publish_local_area.py`.

## Signature

```python
def build_h5(
    weights: np.ndarray,
    blocks: np.ndarray,
    dataset_path: Path,
    output_path: Path,
    cds_to_calibrate: List[str],
    cd_subset: List[str] = None,
    county_filter: set = None,
    rerandomize_takeup: bool = False,
    takeup_filter: List[str] = None,
) -> Path:
```

## Parameter Semantics

| Parameter | Type | Purpose |
|---|---|---|
| `weights` | `np.ndarray` | Stacked weight vector, shape `(n_geo * n_hh,)` |
| `blocks` | `np.ndarray` | Block GEOID per weight entry (same shape). If `None`, generated from CD assignments. |
| `dataset_path` | `Path` | Path to base dataset H5 file |
| `output_path` | `Path` | Where to write the output H5 file |
| `cds_to_calibrate` | `List[str]` | Ordered list of CD GEOIDs defining weight matrix row ordering |
| `cd_subset` | `List[str]` | If provided, only include rows for these CDs |
| `county_filter` | `set` | If provided, scale weights by P(target counties \| CD) for city datasets |
| `rerandomize_takeup` | `bool` | Re-draw takeup using block-level seeds |
| `takeup_filter` | `List[str]` | List of takeup variables to re-randomize |

## How `cd_subset` Controls Output Level

The `cd_subset` parameter determines what geographic level the output represents:

- **National** (`cd_subset=None`): All CDs included — produces a full national dataset.
- **State** (`cd_subset=[CDs in state]`): Filter to CDs whose FIPS prefix matches the state — produces a state dataset.
- **District** (`cd_subset=[single_cd]`): Single CD — produces a district dataset.
- **City** (`cd_subset=[NYC CDs]` + `county_filter=NYC_COUNTIES`): Multiple CDs with county filtering — produces a city dataset. The `county_filter` scales weights by the probability that a household in each CD falls within the target counties.

## Internal Pipeline

1. **Load base simulation** — One `Microsimulation` loaded from `dataset_path`. Entity arrays and membership mappings extracted.

2. **Reshape weights** — The flat weight vector is reshaped to `(n_geo, n_hh)`.

3. **CD subset filtering** — Rows for CDs not in `cd_subset` are zeroed out.

4. **County filtering** — If `county_filter` is set, each row is scaled by `P(target_counties | CD)` via `get_county_filter_probability()`.

5. **Identify active clones** — `np.where(W > 0)` finds all nonzero entries. Each represents a distinct household clone.

6. **Clone entity arrays** — Entity arrays (household, person, tax_unit, spm_unit, family, marital_unit) are cloned using fancy indexing on the base simulation arrays.

7. **Reindex entity IDs** — All entity IDs are reassigned to be globally unique. Cross-reference arrays (e.g., `person_household_id`) are updated accordingly.

8. **Derive geography** — Block GEOIDs are mapped to state FIPS, county, tract, CBSA, etc. via `derive_geography_from_blocks()`. Unique blocks are deduplicated for efficiency.

9. **Recalculate SPM thresholds** — SPM thresholds are recomputed using `calculate_spm_thresholds_vectorized()` with the clone's CD-level geographic adjustment factor.

10. **Rerandomize takeup** (optional) — If enabled, takeup booleans are redrawn per census block using `apply_block_takeup_to_arrays()`.

11. **Write H5** — All variable arrays are written to the output file.

## Usage Examples

### National
```python
build_h5(
    weights=w,
    blocks=blocks,
    dataset_path=Path("base.h5"),
    output_path=Path("national/US.h5"),
    cds_to_calibrate=cds,
)
```

### State
```python
state_fips = 6  # California
cd_subset = [cd for cd in cds if int(cd) // 100 == state_fips]
build_h5(
    weights=w,
    blocks=blocks,
    dataset_path=Path("base.h5"),
    output_path=Path("states/CA.h5"),
    cds_to_calibrate=cds,
    cd_subset=cd_subset,
)
```

### District
```python
build_h5(
    weights=w,
    blocks=blocks,
    dataset_path=Path("base.h5"),
    output_path=Path("districts/CA-12.h5"),
    cds_to_calibrate=cds,
    cd_subset=["0612"],
)
```

### City (NYC)
```python
from policyengine_us_data.calibration.publish_local_area import (
    NYC_COUNTIES, NYC_CDS,
)

cd_subset = [cd for cd in cds if cd in NYC_CDS]
build_h5(
    weights=w,
    blocks=blocks,
    dataset_path=Path("base.h5"),
    output_path=Path("cities/NYC.h5"),
    cds_to_calibrate=cds,
    cd_subset=cd_subset,
    county_filter=NYC_COUNTIES,
)
```
