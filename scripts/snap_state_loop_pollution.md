# SNAP ~4% Gap: State Loop Pollution in Matrix Builder

## Summary

The matrix builder's `_build_state_values` reuses one `Microsimulation`
object and cycles through all 51 states. Between iterations it calls
`delete_arrays` on calculated variables, but this does not fully purge
intermediate cached state. Residual values from earlier states leak into
SNAP calculations for later states, inflating eligible amounts by ~3-4%
at the aggregate level.

The stacked dataset builder is unaffected because it creates a fresh
simulation per congressional district.

## How we got here

### Step 1: verify_county_fix.py surfaced the gap

`verify_county_fix.py` (N_CLONES=3, uniform weights) compares
`X @ w` from the matrix builder against weighted sums from stacked
h5 files for the same CDs.

Key result:

```
snap (NC state):
  X @ w:       $462,310
  Stacked sum: $444,658
  Ratio:       1.040  [GAP]
```

Per-CD checks all passed (ratio ~1.0). The gap only appeared at
the state level, when aggregating across all NC congressional
districts.

### Step 2: Ruling out draw-level causes

Over several debugging sessions we systematically ruled out:

| Hypothesis | Result |
|---|---|
| Block collision in stacked format | Zero collisions with N_CLONES=3 |
| Benefit interaction (TANF→SNAP) | Both builders force non-filtered takeup=True |
| Entity-to-household mapping differs | 100% match on all 3 entity types |
| SPM geographic adjustment | SNAP uses FPL, not SPM thresholds |
| Entity ID reindexing | Happens after takeup draws |

### Step 3: debug_snap_draws.py confirmed identical draws

`debug_snap_draws.py` picks 10 NC households with SNAP-eligible SPM
units and traces every detail of the takeup draw from both builders:
block GEOID, salt, RNG seed, raw draws, rate, takeup booleans,
eligible amounts, and final values.

Result: **all draws are byte-identical.** Blocks, salts, seeds,
random numbers, and takeup booleans match perfectly for every
sampled household.

But the script also revealed a hidden clue. For 2 of the 10 sampled
households, the actual X matrix value at the state-level SNAP row
differed from the manually computed eligible × takeup:

```
HH 48097:  manual eligible=$3,253  X[snap_NC]=$3,350  (+3.0%)
HH 153976: manual eligible=$1,448  X[snap_NC]=$1,512  (+4.4%)
```

The manual computation used a fresh sim. The X matrix used
`state_values[37]["entity"]["snap"]` from the builder's
precomputation loop. The eligible amounts themselves were
different.

### Step 4: debug_state_precomp.py isolated the cause

`debug_state_precomp.py` tests whether cycling states on one sim
object produces different SNAP values than a fresh sim:

| Test | Description | SNAP total (NC) | Diff | SPM units affected |
|---|---|---|---|---|
| A | Fresh sim, state=37 | $6,802,671 | — | — |
| B | After 51-state loop | $7,013,358 | +$210,686 (+3.1%) | 340 / 12,515 |
| C | After NY→NC only | $6,825,187 | +$22,516 (+0.3%) | 74 / 12,515 |
| D | NC twice, no other state | $6,802,671 | $0 | 0 / 12,515 |

**Test D** proves NC-on-NC is perfectly reproducible — no issue with
the sim framework itself.

**Test C** proves even a single state switch (NY→NC) pollutes 74 SPM
units, adding $22k.

**Test B** proves the full 51-state loop compounds pollution to 340
SPM units and +$210k (+3.1%), matching the observed ~4% gap.

Among the most polluted SPM units, some jump from $0 to $5,000+ —
households that should have zero SNAP eligibility under NC rules but
inherit stale eligibility from a previous state's calculation.

## Root cause

`_build_state_values` (unified_matrix_builder.py, lines 101-264)
runs this loop:

```python
for state in unique_states:
    sim.set_input("state_fips", ..., state)
    for var in get_calculated_variables(sim):
        sim.delete_arrays(var)
    # ... calculate snap, aca_ptc, etc.
```

`get_calculated_variables` returns variables that have cached
computed arrays. `delete_arrays` removes those arrays. But at least
one intermediate variable in SNAP's dependency tree is not being
caught — likely because it is classified as an input variable, or
because it was set via `set_input` during a previous state's
computation and is therefore not in the "calculated" set.

When the loop reaches NC (position 33 of 51), the SNAP formula for
certain households picks up a stale intermediate value from one of
the 33 previously processed states.

## Why per-CD checks passed

The stacked builder creates a fresh `Microsimulation(dataset=...)`
per CD, so it never encounters this pollution. The matrix builder's
per-CD X values are also polluted, but when `verify_county_fix.py`
compared them against a stacked sim for the same CD, both the
numerator and denominator reflected the same geographic slice of
the polluted data. The state-level aggregation across all NC CDs
amplified the absolute magnitude of the error, making it visible
as a ~4% ratio gap.

## Affected code

- `unified_matrix_builder.py`: `_build_state_values` (lines 101-264)
- Also potentially `_build_county_values` (lines 266+), which uses
  the same sim-reuse pattern for county-dependent variables

## Fix options

1. **Fresh sim per state** in `_build_state_values`: create a new
   `Microsimulation(dataset=...)` for each of the 51 states instead
   of reusing one. Correct but slower (~51× sim load overhead).

2. **Identify the leaking variable**: trace SNAP's full dependency
   tree and find which intermediate variable `get_calculated_variables`
   misses. Ensure it is explicitly deleted (or never set as input)
   between state iterations.

3. **Hybrid approach**: reuse the sim but call a deeper cache-clearing
   method that resets all non-input arrays, not just those returned by
   `get_calculated_variables`.

## Reproducing

```bash
# Confirm the gap exists (~40 min, includes county precomputation)
python scripts/verify_county_fix.py

# Confirm draws are identical, spot the eligible-amount discrepancy (~40 min)
python scripts/debug_snap_draws.py

# Confirm state loop pollution is the cause (~15 min)
python scripts/debug_state_precomp.py
```
