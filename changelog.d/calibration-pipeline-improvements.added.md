Unified calibration pipeline with GPU-accelerated L1/L0 solver, target config YAML, and CLI package validator.
Per-state and per-county precomputation replacing per-clone Microsimulation (51 sims instead of 436).
Parallel state, county, and clone loop processing via ProcessPoolExecutor.
Block-level takeup re-randomization with deterministic seeded draws.
Hierarchical uprating with ACA PTC state-level CSV factors and CD reconciliation.
Modal remote runner with Volume support, CUDA OOM fixes, and checkpointing.
H5 builder that filters calibrated clone weights by CD subset, uses pre-assigned random census blocks from `geography.npz` to derive full sub-state geography, and produces self-contained local area datasets.
Staging validation script (validate_staging.py) with sim.calculate() comparison and sanity checks.
