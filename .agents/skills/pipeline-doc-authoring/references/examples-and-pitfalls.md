# Examples and Pitfalls

## Strong examples

### Stage 2 Extended CPS process nodes

- `policyengine_us_data/datasets/cps/extended_cps.py`
  - `clone_features`
  - `cps_only`
  - `qrf_pass2`
  - `formula_drop`

Why these are good examples:

- They document real pipeline-visible transformations.
- Their wording tracks the current code path rather than vague historical descriptions.
- Their stage order is enforced in `pipeline_stages.yaml`.

### Shared node reused across stages

- `policyengine_us_data/utils/mortgage_interest.py`
  - `mortgage_convert`

Why this is a good example:

- The same decorated function is reused in both Stage 1 and Stage 2 by referencing the same node ID
  in multiple stage edge sets.

### Orchestration wrapper rendered as a group

- `pipeline_stages.yaml`
  - Stage `3b` group for `create_stratified_cps_dataset()`
  - Stage `5` and `6` groups for `run_calibration()`

Why this is a good example:

- The wrapper is visible without creating fake data-flow nodes.
- The real substeps remain the actual graph nodes.

### Local-area build orchestration

- `policyengine_us_data/calibration/publish_local_area.py`
  - `build_h5`
  - `phase1`
  - `phase2`
  - `phase3`

Why this is a good example:

- It documents a multi-phase orchestrator while still allowing `main` to evolve helper behavior
  underneath it.

## Common pitfalls

### Decorator added, node still missing

Cause:

- The node ID is not referenced by any stage edge in `pipeline_stages.yaml`.

Fix:

- Add the node to the correct stage by wiring its ID into one or more edges.

### Wrapper function duplicated as a normal node

Cause:

- The function is only an orchestrator around already-expanded substeps.

Fix:

- Prefer a `group` unless the wrapper itself is a meaningful pipeline step.

### Description is technically true but still stale

Cause:

- `main` changed the behavior inside the decorated function after the docs were first written.

Fix:

- Re-read the rebased code path and update `description`, `details`, and stage text together.

### Visual renderer edited for a metadata problem

Cause:

- A missing node or wrong order is treated as a ReactFlow/ELK issue.

Fix:

- Fix decorators, YAML, or extractor output first. Touch renderer code only when the rendering model
  itself is wrong.

### Shared node reused incorrectly

Cause:

- The same node ID is reused across stages for two behaviors that have drifted apart.

Fix:

- Split into separate node IDs when the semantics differ, even if the implementation used to be
  shared.
