# Geo-Stacking Calibration: Project Status

### Congressional District Calibration - RESOLVED ✓

**Matrix Dimensions Verified**: 34,089 × 4,612,880
- 30 national targets
- 7,848 age targets (18 bins × 436 CDs)  
- 436 CD SNAP household counts
- 487 total SNAP targets (436 CD + 51 state costs)
- 25,288 IRS SOI targets (58 × 436 CDs)
- **Total: 34,089 targets** ✓

**Critical Fix Applied (2024-12)**: Fixed IRS target deduplication by including constraint operations in concept IDs. AGI bins with boundaries like `< 10000` and `>= 10000` are now properly distinguished.

**Key Design Decision for CD Calibration**: State SNAP cost targets (51 total) apply to households within each state but remain state-level constraints. Households in CDs within a state have non-zero values in the design matrix for their state's SNAP cost target.

**Note**: This target accounting is specific to congressional district calibration. State-level calibration will have a different target structure and count.

#### What Should Happen (Hierarchical Target Selection)
For each target concept (e.g., "age 25-30 population in Texas"):
1. **If CD-level target exists** → use it for that CD only
2. **If no CD target but state target exists** → use state target for all CDs in that state  
3. **If neither CD nor state target exists** → use national target

For administrative data (e.g., SNAP):
- **Always prefer administrative over survey data**, even if admin is less granular
- State-level SNAP admin data should override CD-level survey estimates

## Analysis

#### State Activation Patterns

#### Population Target Achievement

## L0 Package (~/devl/L0)
- `l0/calibration.py` - Core calibration class
- `tests/test_calibration.py` - Test coverage

## Documentation
- `GEO_STACKING_TECHNICAL.md` - Technical documentation and architecture
- `PROJECT_STATUS.md` - This file (active project management)
