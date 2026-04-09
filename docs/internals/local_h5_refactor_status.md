# Local H5 Refactor Status

Date: 2026-04-09

This note records what actually landed in the `fix/target-architecture-h5`
refactor for the US local and national H5 publishing path.

It is intentionally narrower than the broader architecture planning docs. The goal here is to
describe the code that now exists, the remaining thin spots, and the work that was explicitly
deferred.

## What Landed

The H5 path now has explicit internal contracts and a request-driven architecture:

- `policyengine_us_data.calibration.local_h5.contracts`
  - request, filter, validation, and worker result contracts
  - `to_dict()` / `from_dict()` support for adapter boundaries
- `policyengine_us_data.calibration.local_h5.partitioning`
  - tested weighted work partitioning
- `policyengine_us_data.calibration.local_h5.package_geography`
  - exact calibration-package geography loading
- `policyengine_us_data.calibration.local_h5.fingerprinting`
  - typed publish fingerprint inputs and records
- `policyengine_us_data.calibration.local_h5.selection`
  - clone-weight layout and area selection
- `policyengine_us_data.calibration.local_h5.source_dataset`
  - worker-scoped source snapshot with lazy variable access
- `policyengine_us_data.calibration.local_h5.reindexing`
  - pure entity reindexing
- `policyengine_us_data.calibration.local_h5.variables`
  - variable cloning and export policy
- `policyengine_us_data.calibration.local_h5.us_augmentations`
  - US-only payload augmentation
- `policyengine_us_data.calibration.local_h5.builder`
  - `LocalAreaDatasetBuilder` as the one-area orchestration root
- `policyengine_us_data.calibration.local_h5.writer`
  - `H5Writer` as the H5 persistence boundary
- `policyengine_us_data.calibration.local_h5.worker_service`
  - `WorkerSession`
  - `LocalH5WorkerService`
  - validation context loading
  - request/result adaptation helpers
- `policyengine_us_data.calibration.local_h5.area_catalog`
  - concrete `USAreaCatalog`

The public entrypoints still exist, but they are now adapters over the internal components:

- `policyengine_us_data.calibration.publish_local_area.build_h5(...)`
- `modal_app.worker_script`
- `modal_app.local_area.coordinate_publish(...)`
- `modal_app.local_area.coordinate_national_publish(...)`

## Current Shape

The current H5 publishing path is:

1. coordinator derives publish inputs and fingerprint
2. coordinator builds concrete US requests from `USAreaCatalog`
3. coordinator partitions weighted requests across workers
4. worker script loads one `WorkerSession`
5. worker service iterates requests in the chunk
6. builder creates one in-memory payload per request
7. writer persists the H5
8. validation runs per output when enabled
9. coordinator aggregates structured worker results

In other words:

- one-area build logic now lives in `LocalAreaDatasetBuilder`
- one-worker-chunk logic now lives in `LocalH5WorkerService`
- coordinator logic is thinner and request-driven

## What Stayed Concrete And US-Specific

This refactor deliberately did **not** try to create a fake shared cross-country core.

Still US-specific by design:

- `CloneWeightMatrix`
- `USAreaCatalog`
- `USAugmentationService`
- the current local-H5 coordinator/orchestration adapters

That is intentional. The code was only generalized where there was already a real stable seam.

## Test Status

The refactor added a cheap unit-first suite around the new seams. At the end of
the coordinator refactor, the targeted local-H5 suite was passing:

```text
81 passed
```

Coverage now exists for:

- contracts
- partitioning
- validation helpers and worker validation contract
- package geography loading
- fingerprinting
- selection
- source snapshot loading
- reindexing
- variable cloning
- US augmentations
- builder and writer seams
- worker service behavior
- US area catalog behavior
- coordinator contract behavior
- calibration package serialized geography round-trip

The deliberate gap is heavy runtime integration. The branch does **not** add a broad slow parity
suite.

This was intentional. The PR was designed so most correctness lives in unit-testable
components, with only thin compatibility or seam coverage on top.

## Deferred Follow-Ups

These items were explicitly left out or only partially handled:

1. Heavy compatibility and invariant testing
   - broader `build_h5` runtime parity
   - deeper `X @ w` / area-aggregate invariants
   - full Modal-like integration coverage

2. Validator unification
   - per-area target validation is now structurally correct
   - national validation is still partly separate
   - only `ValidationPolicy.enabled` is enforced today; the finer-grained
     validation policy fields are present but not fully wired through

3. Fingerprint schema simplification
   - clone count is now canonicalized from weights
   - long-term package-backed fingerprinting should stop treating `n_clones` and `seed` as
     semantic equality inputs

4. Possible later shared-core extraction
   - nothing in this branch proves that the US abstractions are yet the right shared abstractions
     for UK or another country

5. Coordinator cleanup beyond the H5 scope
   - Modal upload/promotion/manifest logic remains adapter-heavy
   - that is outside the intended scope of this refactor

## What This Documentation Does Not Claim

This branch does **not** establish a reusable cross-country core library.

It does establish a cleaner set of seams that another country pipeline could
learn from:

- request/result contracts
- builder and worker-service boundaries
- package-backed geography loading
- lazy source snapshot handling

Whether any of those should later move into a real shared abstraction should be
decided only after a second concrete implementation proves the shape.

## Reading Order

If you need to understand the landed architecture quickly, read in this order:

1. `policyengine_us_data/calibration/local_h5/contracts.py`
2. `policyengine_us_data/calibration/local_h5/builder.py`
3. `policyengine_us_data/calibration/local_h5/worker_service.py`
4. `policyengine_us_data/calibration/local_h5/area_catalog.py`
5. `policyengine_us_data/calibration/publish_local_area.py`
6. `modal_app/worker_script.py`
7. `modal_app/local_area.py`
