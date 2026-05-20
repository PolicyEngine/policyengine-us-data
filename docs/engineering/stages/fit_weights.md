# Stage 3: Fit Weights

Stage 3 produces scoped fitted-weight artifacts for regional and national H5
builds. The public identity boundary lives in `policyengine_us_data.fit_weights`:

- `FitScope` names the durable regional and national scopes.
- `FittedWeightsSpec` defines the scoped optimization parameters recorded in
  step manifests for reuse decisions.
- `ScopedFitArtifacts` defines the artifact filenames written by the Modal fit
  step and consumed by downstream H5 builders.

The current artifact names remain behavior-compatible:

- regional: `calibration_weights.npy`, `geography_assignment.npz`,
  `unified_run_config.json`, `unified_diagnostics.csv`, and
  `calibration_log.csv`;
- national: `national_calibration_weights.npy`,
  `national_geography_assignment.npz`, `national_unified_run_config.json`,
  `national_unified_diagnostics.csv`, and `national_calibration_log.csv`.

When changing Stage 3 fitting parameters, artifact names, or scope behavior,
update the central specs first and then adapt Modal callers to consume those
specs. Do not add parallel filename constants in orchestration code.
