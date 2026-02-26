Unified calibration pipeline with GPU-accelerated L1/L0 solver, target config YAML, and CLI package validator.
Per-state and per-county precomputation replacing per-clone Microsimulation (51 sims instead of 436).
Parallel state, county, and clone loop processing via ProcessPoolExecutor.
Block-level takeup re-randomization with deterministic seeded draws.
Hierarchical uprating with ACA PTC state-level CSV factors and CD reconciliation.
Modal remote runner with Volume support, CUDA OOM fixes, and checkpointing.
Stacked dataset builder with sparse CD subsets and calibration block propagation.
