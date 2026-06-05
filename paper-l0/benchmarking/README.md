# Benchmarking Scaffold

This directory contains the implementation scaffold for benchmarking the
`L0` calibration pipeline against:

- `GREG` via the `svy` Python package (classical linear `cal.linear`)
- `IPF` (raking) via the `svy` Python package

Both run in-process; the benchmark has no R dependency.

## Experimental Setup

The benchmark is organized around one shared exported bundle and multiple
method adapters.

- `L0` and `GREG` are compared on the shared calibration representation:
  a sparse target-by-unit matrix, the selected target table, and 
  initial .npy weights.
- `IPF` is benchmarked from the same target selection, but it requires a
  conversion step because raking consumes a microdata table plus categorical
  margin constraints rather than a generic sparse linear system.
- The intended benchmark tiers are:
  - a practical reduced-size comparison tier, used for like-for-like `L0`
    versus `GREG` runs that are small enough to execute routinely during
    development
  - an IPF-focused reduced-size tier on count-style targets, used because
    classical `IPF` is most naturally evaluated on count or indicator margins
    rather than the full arbitrary target set
  - a scaling ladder over increasing target counts, used to show how runtime,
    memory use, convergence, and outright failure change as the benchmark moves
    from small target subsets toward the full calibration problem
  - a production-feasibility tier, used to test which methods can still run at
    something close to the full production clone count and target volume

Methodologically, the benchmark treats the methods as related but not
identical:

- `L0` and `GREG` can consume arbitrary linear calibration targets.
- `IPF` is most natural for count-style or indicator-style targets, and the
  `svy` raking engine operates on a **single entity scope per run** (one flat
  frame of units, one row per unit). Each IPF run is therefore restricted to a
  single count family via `method_options.ipf.count_variable` (default
  `household_count`); the configured family's targets are converted to
  categorical margins, and all other count families (`person_count`,
  `tax_unit_count`, `spm_unit_count`, `family_count`, `marital_unit_count`) are
  dropped at the count check with explicit diagnostics. Those targets remain in
  the shared sparse system that L0 and GREG fit, so the cross-method comparison
  on the IPF-feasible subset is still apples-to-apples via
  `--score-on ipf_retained_authored`.

  Within its scope, the engine reproduces classical raking, including leaving
  units outside a margin's targeted cells (e.g. units in untargeted
  geographies) at their base weight — `svy` raking achieves this by padding
  uncovered observed categories with their current weighted totals. Solving a
  single weight vector jointly across multiple entity levels is the two-level
  `conP`/`conH` regime that `svy` raking does not implement; such mixed-scope
  bundles are rejected with a clear error rather than silently mis-counted.

The core workflow is:

1. select a benchmark target subset with a manifest
2. export a shared benchmark bundle from a saved calibration package
3. auto-convert the bundle to IPF inputs when needed
4. run `L0`, `GREG`, or `IPF`
5. score each method against the target set that matches its benchmark contract

## Layout

- `benchmark_cli.py`
  Main CLI for exporting benchmark bundles and running methods.
- `benchmark_manifest.py`
  Manifest schema and target-filter logic.
- `benchmark_export.py`
  Export utilities for shared benchmark artifacts.
- `ipf_conversion.py`
  Automatic conversion from the saved calibration package to IPF-ready
  unit and target metadata.
- `benchmark_metrics.py`
  Common diagnostics and summary generation.
- `svy_engine.py`
  In-process `svy` GREG and IPF (raking) engines.
- `requirements-python.txt`
  Python dependencies for the benchmarking scaffold (includes the `calibration`
  extra, which provides `svy`).
- `manifests/*.example.json`
  Example benchmark manifests.

## Environment Setup

```bash
pip install -r paper-l0/benchmarking/requirements-python.txt
```

Or, from the repo root:

```bash
make benchmarking-install-python
```

## Chosen Interchange Formats

- sparse matrix: Matrix Market `.mtx`
- target metadata: `.csv`
- unit metadata: `.csv`
- initial weights: `.npy`
- benchmark manifest: `.json`
- method result summary: `.json`
- fitted weights: `.npy`

## Notes

### Shared calibration package

The exporter reads the saved calibration package directly from pickle rather
than importing the full calibration CLI. This keeps the benchmark I/O path
lightweight.

### IPF inputs

The exporter auto-generates IPF inputs when the manifest includes `ipf` and no
external overrides are supplied. It reconstructs an IPF microdata table from:

- the saved calibration package
- the package metadata's `dataset_path`
- the package metadata's `db_path`
- the selected count-like targets and their stratum constraints

The generated `unit_metadata.csv` is built for `person_count` and
`household_count` targets — the person/household entity levels classical raking
supports (a single run uses one scope). It expands cloned households to a person-level
table when person targets are present, carries a repeated household
`unit_index` so per-person weights collapse cleanly back to per-household, and
adds one string-valued derived category column per declared bucket schema
(e.g. `age_bracket`, `agi_bracket_district`, `snap_positive`). Targets at
other entity levels (e.g. `tax_unit_count`, `spm_unit_count`) are dropped at
the count check with `non_count_style` diagnostics; they remain in the shared
sparse target matrix that L0 and GREG fit.

The generated `ipf_target_metadata.csv` contains one `categorical_margin` row
per retained IPF cell after validation. That means:

- authored cells that belong to a closed categorical system are kept
- binary subset families may gain exactly-derived complement cells when an
  authored parent total exists on the exact reduced key
- open subset families are dropped rather than emitted as 1-cell margins

The exporter also writes:

- `ipf_scoring_target_metadata.csv`
- `ipf_scoring_X_targets_by_units.mtx`

These score IPF on its retained authored targets only. Derived complements are
recorded for transparency in `ipf_conversion_diagnostics.json`, but they are
not part of the main benchmark metric set.

When comparing `L0` or `GREG` against that same subset, pass:

```bash
python paper-l0/benchmarking/benchmark_cli.py run \
  --method l0 \
  --run-dir <bundle> \
  --score-on ipf_retained_authored
```

### Matched IPF / L0 / GREG comparison

When IPF retains a strict subset of the requested targets (because not every
authored target survives the closure rules), the natural comparison is to fit
L0 and GREG on the same training set IPF was given. The CLI supports this with
`--train-on ipf_retained_authored`, which loads the IPF scoring subset as the
training matrix and target CSV instead of the shared requested bundle. Two-fit
recipe:

```bash
# Full-info L0 (default; trains and scores on the shared requested set)
python paper-l0/benchmarking/benchmark_cli.py run \
  --method l0 --run-dir <bundle>

# Matched L0 — same training inputs IPF saw, scored on the same subset
python paper-l0/benchmarking/benchmark_cli.py run \
  --method l0 --run-dir <bundle> \
  --train-on ipf_retained_authored \
  --score-on ipf_retained_authored
```

The matched run writes its summary to `outputs/{method}_matched_summary.json`
so it does not overwrite the full-info run's `outputs/{method}_summary.json`.
GREG follows the same pattern. IPF ignores `--train-on` because its training
inputs are always its own categorical-margin tables.

### Determinism

L0 is the only method with a stochastic optimizer. Set `method_options.l0.seed`
in the manifest (the example manifests use `42`). The CLI plumbs this seed
through to `fit_l0_weights`, which seeds torch, CUDA (when available), and
numpy at fit-time. Re-running with the same seed produces bit-identical
weights. GREG and IPF are deterministic by construction.

### GREG and weight non-negativity

The GREG engine uses classical linear calibration (`svy`'s `calibrate` with
`bounded=False`, numerically equivalent to R `survey::grake` with `cal.linear`
and `bounds = (-Inf, Inf)`). This is classical GREG and routinely emits negative
fitted weights — the trade-off for an exact, closed-form linear-system fit.
L0 and IPF produce non-negative weights by construction. The benchmark records
this as `negative_weight_share` in `compute_common_metrics`, and the paper
should report it prominently when comparing GREG to L0 / IPF on weight quality.
A bounded raking variant is out of scope for the current benchmark.

### Strictness contract

The export step succeeds with a loud diagnostic summary as long as the
retained authored IPF target set is coherent. Requested targets that the
converter dropped (non-count style, unresolvable, missing parent total,
ambiguous parent, negative complement, unsupported partial margin, mixed
universe, or incompatible totals) are listed in
`inputs/ipf_conversion_diagnostics.json`. If no closed system survives, or if
the package is missing the `target_id` column needed to assemble the scoring
subset, export fails fast with `IPFConversionError` and the diagnostics file
is still written so the failure mode is auditable.

External CSVs are still supported through `external_inputs.*` and override the
automatic conversion path when provided. The external-IPF contract is strict:

- `external_inputs.ipf_unit_metadata_csv`
- `external_inputs.ipf_target_metadata_csv`
- `external_inputs.ipf_scoring_target_metadata_csv`
- `external_inputs.ipf_scoring_matrix_mtx`

must be provided together. An optional
`external_inputs.ipf_conversion_diagnostics_json` can also be supplied and will
be copied through for reporting. External CSVs must also follow the
`categorical_margin` schema below; the runner rejects `numeric_total` rows.

### IPF conversion step by step

The IPF conversion is implemented in
[ipf_conversion.py](./ipf_conversion.py) and runs during
`benchmark_cli.py export`.

1. Load the saved calibration package and apply the manifest target filters.
2. Read `dataset_path`, `db_path`, and `n_clones` from the package metadata.
3. Query `stratum_constraints` for the selected targets from the target DB.
4. Identify the source variables needed to evaluate those constraints, such as
   `age`, `snap`, or `medicaid_enrolled`.
5. Reconstruct the cloned household universe from `initial_weights`,
   `block_geoid`, and `cd_geoid`. This yields one benchmark unit per matrix
   column.
6. If any selected IPF target is `person_count`, expand that cloned household
   universe to a person-level table using the source dataset's person-to-
   household links. Multiple person rows may therefore share the same
   household-clone `unit_index`.
7. Calculate the needed source variables from the dataset and attach them to
   the IPF unit table.
8. Materialize the string-valued derived category columns the margins cover
   (e.g. `age_bracket`, `snap_positive`) on that unit table.
9. Group the resolved targets into margin families, validate them against the
   observed unit-table support, and keep only families that are already closed
   or can be closed exactly from authored parent totals.
10. Emit one `categorical_margin` row per retained authored or exactly-derived
    cell, sharing a `margin_id` within each family.
11. Write diagnostics (`dropped_targets`, retained-authored counts, derived
    complements, and any coherence issues) to
    `inputs/ipf_conversion_diagnostics.json`.
12. Run `svy` raking once on the generated unit table and full validated
    IPF target metadata (single scope; uncovered categories padded to their
    base totals so they are left untouched).
13. Collapse the fitted IPF row weights back to one weight per shared
    benchmark `unit_index`, so the fitted result can be scored against the
    retained-authored sparse target subset used for the IPF benchmark.

This means the benchmark keeps a shared requested target space for the export,
but an IPF-specific retained-authored scoring space for the actual IPF
comparison.

### Why the IPF conversion exists

`L0` and `GREG` can work directly with a sparse linear system of the form
`X w = t`.

Classical `IPF` does not start from that object. It expects:

- a unit-record table
- categorical or indicator variables on that table
- target totals over those variables

So the benchmark exporter translates selected count-style calibration targets
into that IPF-friendly representation instead of trying to feed the sparse
matrix directly into the raking engine.

### IPF target metadata schema

The IPF engine (`fit_ipf_svy`) accepts one encoding: `categorical_margin`. One
row per authored margin cell:

- `scope`: `person` or `household` (a single run uses one scope)
- `target_type`: `categorical_margin`
- `margin_id`: identifier for a margin block. Rows sharing a `margin_id` are
  grouped into one raking control (one categorical margin; multi-variable
  margins become a composite category column).
- `variables`: pipe-separated variable names, e.g.
  `congressional_district_geoid|age_bracket`
- `cell`: pipe-separated assignments, e.g.
  `congressional_district_geoid=0601|age_bracket=0-4`
- `target_value`: numeric target

Open subset systems are not exported. If a subset family cannot be closed from
an authored parent total, it is dropped before raking.

## Example Commands

Export a benchmark bundle:

```bash
python paper-l0/benchmarking/benchmark_cli.py export \
  --manifest paper-l0/benchmarking/manifests/greg_demo_small.example.json \
  --output-dir paper-l0/benchmarking/runs/greg_demo_small
```

Run a GREG benchmark from an exported bundle:

```bash
python paper-l0/benchmarking/benchmark_cli.py run \
  --method greg \
  --run-dir paper-l0/benchmarking/runs/greg_demo_small
```

Run `L0` on an exported bundle:

```bash
python paper-l0/benchmarking/benchmark_cli.py run \
  --method l0 \
  --run-dir paper-l0/benchmarking/runs/greg_demo_small
```

Equivalent root Make targets:

```bash
make benchmarking-export MANIFEST=paper-l0/benchmarking/manifests/greg_demo_small.example.json RUN_DIR=paper-l0/benchmarking/runs/greg_demo_small
make benchmarking-run-greg RUN_DIR=paper-l0/benchmarking/runs/greg_demo_small
make benchmarking-run-l0 RUN_DIR=paper-l0/benchmarking/runs/greg_demo_small
```

## Paper-reported tiers

The manifests under `manifests/tier*.json` are the paper-reported benchmark
configurations from `paper-l0/BENCHMARK_PLAN.md`. They all read from the same
saved calibration package and differ only in `target_filters` — the unit
universe, clone count, source dataset, and initial calibration package are
fixed.

| File | Tier | Methods | Scope |
| --- | --- | --- | --- |
| `tier1_mixed.json` | 1 | L0, GREG | Full filtered slice (count + dollar) over a 5-state, 10-district subset plus national targets |
| `tier1_ipf.json` | 1 | L0, IPF | Same slice; IPF retains the authored closed subset |
| `tier2_scaling_250.json` … `tier2_scaling_10000.json` | 2 | L0, GREG, IPF | Scaling ladder by `max_targets`, expanding geography coverage to grow the target set |
| `tier2_scaling_largest.json` | 2 | L0, GREG, IPF | Largest coherent pre-production subset (no `max_targets` cap) |
| `tier3_production.json` | 3 | L0, GREG, IPF | Least-filtered view; failures are reportable results |

Non-convergence is treated as a reportable result: when `svy` raking fails to
converge (or a bundle is mixed-scope) the engine raises, and the suite records a
visible failed row rather than a silent fitted-weight column. A bounded GREG
variant is intentionally out of scope for the current benchmark.

### One-shot orchestration

`run_benchmark_suite.py` exports each manifest, runs every method declared in
it, schedules matched IPF / L0 / GREG comparisons (`--train-on
ipf_retained_authored --score-on ipf_retained_authored`) when IPF is in play,
and aggregates per-tier summary tables.

```bash
# All three tiers end-to-end (requires built calibration_package.pkl).
python paper-l0/benchmarking/run_benchmark_suite.py \
  --runs-dir paper-l0/benchmarking/runs

# A single tier.
python paper-l0/benchmarking/run_benchmark_suite.py \
  --tier tier_1 \
  --runs-dir paper-l0/benchmarking/runs

# A single rung (re-run after a CI failure).
python paper-l0/benchmarking/run_benchmark_suite.py \
  --manifest paper-l0/benchmarking/manifests/tier2_scaling_2500.json \
  --runs-dir paper-l0/benchmarking/runs
```

Outputs in `--runs-dir`:

- `tier_1_summary.csv`, `tier_2_summary.csv`, `tier_3_summary.csv` — one row
  per method per manifest, with status (`completed` / `failed`), runtime,
  target / unit counts, and the standard error metrics from
  `compute_common_metrics`. Matched IPF / L0 / GREG rows are tagged with
  `training_target_set = ipf_retained_authored`.
- `suite_summary.csv` — concatenated view across all tiers.
- `<manifest>/inputs/`, `<manifest>/outputs/` — the per-manifest bundle that
  `benchmark_cli.py export` and `run` produce, including
  `ipf_conversion_diagnostics.json` whenever IPF was in scope.

Failures (export-time `IPFConversionError`, runner non-zero exit, missing
output files) appear as `status = failed` rows with the captured reason in
`notes`. The orchestrator never aborts the suite — Tier 3 explicitly relies
on this so a GREG out-of-memory or IPF non-convergence is a reportable result
rather than a missing row.
