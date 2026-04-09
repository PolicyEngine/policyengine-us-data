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
5. score all fitted weights against the same shared target matrix

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

The exporter now auto-generates IPF inputs when the manifest includes `ipf`
and no external overrides are supplied. It reconstructs an IPF microdata table
from:

- the saved calibration package
- the package metadata's `dataset_path`
- the package metadata's `db_path`
- the selected count-like targets and their stratum constraints

The generated `unit_metadata.csv` is currently built for `person_count` and
`household_count` targets. It expands cloned households to a person-level table
when person targets are present, carries a repeated household `unit_index`, and
adds one derived indicator column per selected target. The generated
`ipf_target_metadata.csv` then references those indicator columns as numerical
IPF totals.

External CSVs are still supported through `external_inputs.*` and override the
automatic conversion path when provided.

### IPF conversion step by step

The IPF conversion is implemented in
[ipf_conversion.py](/Users/movil1/Desktop/PYTHONJOBS/PolicyEngine/policyengine-us-data/paper-l0/benchmarking/ipf_conversion.py)
and runs during `benchmark_cli.py export`.

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
8. For each selected target, evaluate its original stratum logic row by row and
   materialize the result as a derived indicator column such as
   `ipf_indicator_00000`.
9. Write `ipf_target_metadata.csv` so each selected target becomes a
   `numeric_total` IPF constraint over one of those derived indicator columns.
10. Run `surveysd::ipf` on the generated unit table and target metadata.
11. Collapse the fitted IPF row weights back to one weight per shared benchmark
   `unit_index`, so the fitted result can be scored against the same sparse
   calibration matrix used by `L0` and `GREG`.

This means the benchmark uses one common scoring space even though `IPF`
requires a richer input representation than `L0` and `GREG`.

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

`ipf_runner.R` supports two target metadata encodings:

- `numeric_total`
  One row per target with:
  - `scope`: `person` or `household`
  - `target_type`: `numeric_total`
  - `value_column`: unit-data column to calibrate
  - `variables`: grouping variables used to wrap the numeric total in a one-cell
    or multi-cell array
  - `cell`: pipe-separated assignments for the target cell
  - `target_value`: numeric total
- `categorical_margin`
  One row per margin cell with:
  - `scope`: `person` or `household`
  - `target_type`: `categorical_margin`
  - `margin_id`: identifier for a margin table
  - `variables`: pipe-separated variable names, e.g. `district_id|age_bin`
  - `cell`: pipe-separated assignments, e.g.
    `district_id=0601|age_bin=18_24`
  - `target_value`: numeric target

The automatic conversion path currently emits `numeric_total` rows.

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
