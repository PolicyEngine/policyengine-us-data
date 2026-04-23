# Benchmarking Scaffold

This directory contains the implementation scaffold for benchmarking the
`L0` calibration pipeline against:

- `GREG` via R's `survey` package
- `IPF` via R's `surveysd` package

## Experimental Setup

The benchmark is organized around one shared exported bundle and multiple
method adapters.

- `L0` and `GREG` are compared on the shared calibration representation:
  a sparse target-by-unit matrix, the selected target table, and 
  initial .npy weights.
- `IPF` is benchmarked from the same target selection, but it requires a
  conversion step because `surveysd::ipf` consumes a microdata table plus
  IPF constraints rather than a generic sparse linear system.
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
- `IPF` is most natural for count-style or indicator-style targets, so the
  current automatic conversion path supports `person_count` and
  `household_count`.

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
- `runners/greg_runner.R`
  R backend for `survey`-based GREG.
- `runners/ipf_runner.R`
  R backend for `surveysd`-based IPF.
- `runners/read_npy.R`
  Minimal `.npy` reader used by the R scripts.
- `requirements-python.txt`
  Python dependencies for the benchmarking scaffold.
- `install_r_packages.R`
  Installs the required R packages for the benchmark runners.
- `manifests/*.example.json`
  Example benchmark manifests.

## Environment Setup

Python:

```bash
pip install -r paper-l0/benchmarking/requirements-python.txt
```

R:

```bash
Rscript paper-l0/benchmarking/install_r_packages.R
```

Or, from the repo root:

```bash
make benchmarking-install-python
make benchmarking-install-r
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
`household_count` targets. It expands cloned households to a person-level table
when person targets are present, carries a repeated household `unit_index` so
per-person weights collapse cleanly back to per-household, and adds one
string-valued derived category column per declared bucket schema (e.g.
`age_bracket`, `agi_bracket_district`, `snap_positive`).

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
12. Run `surveysd::ipf` once on the generated unit table and full validated
    IPF target metadata.
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
matrix directly into `surveysd::ipf`.

### IPF target metadata schema

`ipf_runner.R` accepts one encoding: `categorical_margin`. One row per
authored margin cell:

- `scope`: `person` or `household`
- `target_type`: `categorical_margin`
- `margin_id`: identifier for a margin block. Rows sharing a `margin_id` are
  grouped into one `surveysd::ipf` constraint (via `xtabs`).
- `variables`: pipe-separated variable names, e.g.
  `congressional_district_geoid|age_bracket`
- `cell`: pipe-separated assignments, e.g.
  `congressional_district_geoid=0601|age_bracket=0-4`
- `target_value`: numeric target

Open subset systems are not exported. If a subset family cannot be closed from
an authored parent total, it is dropped before the R call.

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
