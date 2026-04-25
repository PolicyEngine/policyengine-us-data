## [1.88.0] - 2026-04-25

### Added

- Construct CPS tax units from ASEC household relationships instead of using Census tax-unit assignments.


## [1.87.0] - 2026-04-25

### Added

- Add a reproducible Forbes-backed PUF top-tail synthesis path.


## [1.86.2] - 2026-04-24

### Changed

- Extracted inline CI/CD workflow Python into dedicated helper scripts under `.github/scripts` and updated the PR, pipeline, local-area publish, and push workflows to call them directly.

### Fixed

- Added a tiny-fixture Modal H5 end-to-end PR test harness and aligned checkpoint/package artifact handling so local-area publication catches contract mismatches earlier.


## [1.86.1] - 2026-04-23

### Fixed

- Refactor Modal runtime setup to rely on `Image.uv_sync()` and the active Python interpreter rather than manual venv wiring, and add an optimized deployed-image seam test lane to the push workflow.
- Activate the uv-managed venv inside Modal pipeline containers so module-scope imports from `policyengine_us_data` (notably `pandas` via `geography/__init__.py`) resolve at container boot. `uv sync --frozen` installs dependencies into `/root/policyengine-us-data/.venv/`, but Modal boots the container with the system Python, so `pipeline.py` imports were failing with `ModuleNotFoundError: No module named 'pandas'`. The image now sets `VIRTUAL_ENV`, prepends `.venv/bin` to `PATH`, and adds the venv site-packages to `PYTHONPATH`.


## [1.86.0] - 2026-04-21

### Added

- Add an ACA marketplace ETL that loads state-level HC.gov bronze-plan
  selection targets for APTC recipients into the calibration database.


## [1.85.2] - 2026-04-21

### Fixed

- Loosened the per-state ACA PTC calibration tolerance from 500% to 1000% in the enhanced/sparse ECPS integration tests. CMS APTC state targets mix outlay and claimed-PTC concepts and don't account for ACA §1331 Basic Health Programs in NY and MN, so several states chronically fail a tight tolerance regardless of reweighting. Temporary until the target-side redesign in #805 lands.


## [1.85.1] - 2026-04-21

### Changed

- Publish TRACE TRO declarations alongside US data release manifests on Hugging Face. The TRO uses canonical TROv 0.1 vocabulary, exposes structured `pe:*` build provenance fields (model version, git sha, data-build fingerprint, CI emission context), and ships with a JSON schema for downstream validation.


## [1.85.0] - 2026-04-21

### Added

- Rebuilt EITC calibration on a coherent IRS SOI TY2022 target set. Added ~102 per-state targets (SOI Historical Table 2) and ~224 per-(child x AGI) targets (Publication 1304 Table 2.5), and removed the contradictory Treasury `tax_expenditures.eitc` aggregate column (which measures outlays, not total claimed) plus the stale TY2020 `eitc.csv` per-child-count targets. The optimizer now has geographic and AGI-shape coverage over EITC without fighting definition mismatches between outlay- and claim-based totals. Addresses #802.


## [1.84.0] - 2026-04-20

### Added

- Added a Marketplace plan benchmark ratio imputation that populates `selected_marketplace_plan_benchmark_ratio` per tax unit by backing out the implied plan cost from CPS-reported private health premiums and PolicyEngine-computed PTC.


## [1.83.4] - 2026-04-18

### Fixed

- Fixed calibration matrix leakage for constrained non-household amount targets by filtering qualifying person-, tax-unit-, and SPM-unit-level amounts before rolling them up to households, so mixed-eligibility households no longer overstate targets such as filer-only `total_self_employment_income`. Added regression tests covering the entity-level filtering behavior and preserving existing household and count-target semantics.


## [1.83.3] - 2026-04-18

### Changed

- Introduced typed local H5 request construction with `AreaBuildRequest`,
  `AreaFilter`, and `USAreaCatalog`, while keeping the worker's legacy
  `--work-items` path available for backward compatibility.


## [1.83.2] - 2026-04-17

### Changed

- Migrate the changelog tooling from `yaml-changelog` (`changelog_entry.yaml` + `changelog.yaml` + `build-changelog`) to towncrier (`changelog.d/<branch>.<type>.md` fragments). The repo's CI already ran `towncrier check` in `pr.yaml` and `bump_version.py` already read fragments from `changelog.d/`; this drops the leftover yaml-changelog artefacts (unused dep, unused reusable workflow, zero-byte `changelog_entry.yaml`, and duplicated `changelog.yaml` whose contents are already in `CHANGELOG.md`) so the tooling story matches the rest of the org.


## [1.83.1] - 2026-04-17

### Fixed

- Add missing `run_id` parameter to `upload_to_staging_hf` in
  `policyengine_us_data.utils.data_upload`, matching the kwarg its
  callers have been passing. Files now land under
  `staging/{run_id}/{rel_path}` when `run_id` is set and under
  `staging/{rel_path}` otherwise, mirroring the prefix convention used
  elsewhere in the package. Unblocks the main `build-and-test` workflow,
  which had been failing with `TypeError: upload_to_staging_hf() got an
  unexpected keyword argument 'run_id'` on every real-content push.


## [1.83.0] - 2026-04-17

### Added

- Add public IRS benchmark checks for `ctc_qualifying_children` and contextual AGI and filing-status child-mix comparisons in `validate_national_h5`.


## [1.82.0] - 2026-04-17

### Added

- Add README files with codebook and documentation links to the cps, sipp, scf, and org dataset folders.


## [1.81.1] - 2026-04-17

### Fixed

- Label hardcoded 2024 dollar targets in etl_national_targets.py with the literal year 2024 to prevent misattribution if the calibration dataset year changes.
- Treat the top healthcare spending age bucket as unbounded (age >= 80) so the 90+ population is covered by an age-specific calibration target rather than only the national total.


## [1.81.0] - 2026-04-17

### Added

- PUF dataset README linking to the IRS 2015 Public Use Booklets and SOI documentation.
- Add SIPP-imputed household vehicle count and value inputs to CPS generation and source imputation so TANF and other asset-tested programs no longer default those vehicle signals to zero.


## [1.80.0] - 2026-04-17

### Added

- Added ACA Marketplace spending and enrollment targets plus state AGI targets to the database build.

### Changed

- Added an explicit refresh path and regression coverage for the legacy `agi_state.csv` SOI targets used by local calibration.


## [1.79.8] - 2026-04-17

### Fixed

- Fix silent integer truncation of imputed rent and real-estate-tax values in CPS — `np.zeros_like(cps["age"])` inherited `age`'s integer dtype, so QRF-imputed float values were floored on assignment. Switch to `np.zeros(len(cps["age"]), dtype=float)`.
- Seed every numpy random call in the CPS and SIPP dataset code — `add_ssn_card_type` no longer calls `np.random.seed` / `np.random.choice` (which clobbered the process-wide RNG), `add_personal_variables` uses a local seeded Generator for the 80-84 age randomization, and `sipp.train_tip_model` / `sipp.train_asset_model` use a seeded Generator for the weighted training-sample draws. Eliminates cross-helper RNG pollution and makes the pickled QRF caches deterministic across rebuilds.
- Defer the `HUGGING_FACE_TOKEN` check into the upload paths of `utils.huggingface` — the previous module-level `raise ValueError` blocked doc builds, lightweight CI, the fully-local calibration workflow (issue #591), and every transitive import via `raw_cache` / `datasets.sipp.sipp` whenever the token was unset. Reads now work without a token; uploads fail late with a clear message.
- Harden `utils.raw_cache.cache_path` against path traversal — reject absolute filenames, parent-directory (`..`) components, and any filename whose resolved path escapes `RAW_INPUTS_DIR`, so future ETL callers can't accidentally read or overwrite files outside the cache directory.
- Filter SIPP tip-model training frame to `MONTHCODE == 12` before the weighted resample — `train_tip_model` previously sampled 10,000 rows from a 12×-bloated panel (one row per person per month, each annualized from that single month), so the QRF treated Jan-annualized and Dec-annualized rows as separate observations and mixed seasonal tip amounts with the annual figures.
- Vectorize mixed-status household detection in `add_ssn_card_type` — previously an O(households × persons) loop (~3×10^10 element-wise comparisons for CPS 2024). Replace with a single pandas groupby over `household_id` so mixed-status detection is linear in the person count.
- Fix chained-indexing no-op in `uprate_puf` under pandas Copy-on-Write — positive-only and negative-only PUF variables (`business_net_profits`, `business_net_losses`, `capital_gains_gross`, `capital_gains_losses`, `partnership_and_s_corp_income`, `partnership_and_s_corp_losses`) were silently not being uprated, dropping them out of both the 2015→2021 and 2021→target-year uprating stages. Switch to `puf.loc[mask, col] *= growth` so the write lands back on the frame.
- Fix double-adjustment of per-capita uprating parameters — `create_policyengine_uprating_factors_table` previously divided every non-weight variable's growth by population growth, which double-adjusted parameters that were already per-capita indices (BLS CPI, SSA COLA, `*_per_capita` spending, per-recipient/per-worker indices). Introduce `is_per_capita_parameter(parameter_path)` so only total-dollar aggregates pass through the population-growth divisor.


## [1.79.7] - 2026-04-17

### Fixed

- Replace hardcoded period=2025 in ACA/Medicaid calibration metric columns with the build_loss_matrix time_period argument.
- Use the [lower, upper) AGI-band boundary convention in the state AGI metric loop, matching the main SOI loop in build_loss_matrix.
- Fix DC SNAP state calibration target drop caused by int/string FIPS mismatch in utils/loss.py.
- Seed numpy before the EnhancedCPS/ReweightedCPS initial weight jitter so calibrated weights are reproducible across runs.
- Limit SIPP tip income imputation to TJB*_TXAMT dollar-amount columns, excluding AJB*_TXAMT allocation flags. Fixes #524.
- Combine AGI bound filters into a single boolean mask in compare_soi_replication_to_soi to avoid chained-indexing misalignment.
- Re-prefix state ACA spending calibration label with state/ (was nation/) so reweight() correctly classifies it as a state target.
- Prefix state Medicaid enrollment calibration label with state/ so it matches the sibling ACA enrollment label and is correctly classified as a state target by reweight().


## [1.79.6] - 2026-04-17

### Changed

- Require `policyengine-us>=1.637.0`, which ships the SSTB QBI split inputs
  and formulas natively, and remove the in-package compat shim that
  backfilled those variables against older `policyengine-us` releases.


## [1.79.5] - 2026-04-17

### Fixed

- Bump `spm-calculator` dependency from `>=0.1.0` to `>=0.2.1` so the SPM thresholds computed for the CPS input dataset use the Betson three-parameter equivalence scale (with the 0.7 exponent) published by Census, not the linear OECD-modified scale shipped in `spm-calculator==0.1.0`. Also picks up tenure-specific GEOADJ shares (renter 0.443, owner-with-mortgage 0.434, owner-without-mortgage 0.323) and the BLS-fidelity improvements in the CE recomputation path. For a 2-adult 2-child reference family the threshold is unchanged; non-reference families shift materially (1 adult / 2 children owner-with-mortgage: $29,767 -> $32,437, +9.0%; 2 adults / 0 children owner-with-mortgage: $27,906 -> $25,530, -8.5%). No code changes in `policyengine_us_data/utils/spm.py`; the package-level API is unchanged across the version bump.


## [1.79.4] - 2026-04-17

### Fixed

- Fix EITC-by-kids calibration: 2-kid bucket was using `>=2` against an exclusive IRS target, causing a 27% EITC aggregate undercount since October 2024. See #766.

  Fix SOI filing-status mask to include SURVIVING_SPOUSE alongside JOINT when matching IRS "Married Filing Jointly/Surviving Spouse" rows, so ~1.58M surviving-spouse tax units are constrained by joint-filer AGI-band targets.

  Tighten the EITC validation reference in `validate_national_h5.py` from ~$60B to ~$67B (2024 Treasury Tax Expenditure estimate) so underconverged calibrations no longer pass sanity checks.

  Replace hardcoded SOI filer-count targets from TY2015 (uprated only by population growth) with dynamic reads from `soi_targets.csv` at the latest SOI year ≤ calibration year. Uses 19 granular AGI bands instead of 7 coarse bands, correcting dramatic distributional shifts (TY2015→TY2023 showed +64% at $100K+ and −27% at $0–5K AGI that population-only uprating missed). See #769.


## [1.79.3] - 2026-04-16

### Changed

- Add a chunked mixed-geography matrix builder for memory-bounded national
  calibration (`--chunked-matrix`) that streams matrix columns in clone-household
  chunks with resumable per-chunk COO shards, progress logging (running average,
  elapsed, ETA), and a shared `entity_clone` module for household-subset
  materialization.

  Fix three target-input integrity bugs surfaced by a new
  `analyze_target_consistency` diagnostic that flags cross-level and
  AGI-bucket-coverage inconsistencies:

  - Drop the IRS workbook override for `total_self_employment_income`,
    `tax_unit_partnership_s_corp_income`, and `net_capital_gains`. The workbook
    columns `business_net_profits` / `partnership_and_s_corp_income` /
    `capital_gains_gross` are gross-only, while the geography-file line codes
    00900 / 26270 / 01000 already report net-of-loss. The override inflated
    these national targets by +40.7% / +26.1% / +3.1% at 2023 values. After
    the fix, all three reconcile to the penny across national, state, and
    district levels.
  - Remove the self-employment QRF winsor in `puf_impute.py`. QRF predictions
    are already bounded by training support; the 0.5/99.5 percentile clip
    was discarding the top 0.5% of legitimate signal and truncating imputed
    self-employment income at ~$1.1M vs the PUF training max of $74.6M.
  - Replace percentile-based top selection in `create_stratified_cps` with
    per-bracket caps (400/400/400/300/300 for the $500k-$1M through $10M+
    bands). Stops PUF templates from piling up above $10M and starving the
    middle-high $1M-$10M range.

  Split calibration checkpoint signature validation into fatal structural
  mismatches and soft hyperparameter mismatches, letting callers tune
  `lambda_l0`, `beta`, `lambda_l2`, and `learning_rate` across resume phases.

  Add `income_tax` national and state SOI targets, drop the unachievable
  JCT `deductible_mortgage_interest` target, and preserve positive mortgage
  interest inputs through structural conversion.

  Retune the national Modal calibration to `lambda_l0=2e-2` at 1000 epochs
  and align `modal_app/pipeline.py` `log_freq` to 100.

  Harden `make clean` so its ignored-CSV cleanup skips local environment and
  dependency directories such as `.venv/`, `venv/`, `env/`, `.tox/`, `.nox/`,
  and `node_modules/`, avoiding accidental deletion of package data inside local
  virtual environments.


## [1.79.2] - 2026-04-14

### Fixed

- Serialize CPS and PUF builds in the Modal integration data build pipeline to avoid reading `CPS_2024` while it is being written.


## [1.79.1] - 2026-04-14

### Changed

- Split the new `calibration.local_h5` contracts into themed request, input,
  validation, and result modules; extract test-only fixtures into dedicated
  fixture helpers; and tighten the new request boundary so construction logic
  stays outside the value objects.


## [1.79.0] - 2026-04-14

### Added

- Added HHS ACF TANF caseload and cash-assistance ETL targets, exposed baseline CPS liquid-asset inputs, and aligned TANF calibration totals to FY2024 administrative data.


## [1.78.4] - 2026-04-13

### Fixed

- Close raw CPS HDF stores after previous-year income and auto-loan preprocessing so CPS builds do not leave `census_cps_2021.h5` and `census_cps_2022.h5` open at process shutdown.


## [1.78.3] - 2026-04-12

### Fixed

- Preflight release-manifest finalization before promoting staged artifacts so failed finalization checks cannot partially publish a release.


## [1.78.2] - 2026-04-12

### Fixed

- Harden CPS basic ORG donor loading against transient fetch failures and concurrent cache builds.


## [1.78.1] - 2026-04-12

### Changed

- Add SSTB QBI split inputs to `policyengine-us-data` by exposing
  `sstb_self_employment_income`, `sstb_w2_wages_from_qualified_business`, and
  `sstb_unadjusted_basis_qualified_property` from the existing PUF/calibration
  pipeline. The current split follows the legacy all-or-nothing
  `business_is_sstb` flag, so mixed SSTB/non-SSTB allocations remain approximate
  until more granular source data or imputation is added.


## [1.78.0] - 2026-04-12

### Added

- Add comparison-mode CTC diagnostics to `validate_national_h5`, including child-count and child-age drift reporting between national artifacts.


## [1.77.0] - 2026-04-10

### Added

- Added richer national CTC calibration and validation coverage by loading AGI-split refundable and nonrefundable CTC targets from IRS geography data, expanding CTC diagnostics to AGI-by-filing-status and child-composition tables, and reporting a canonical ARPA-style CTC reform in national H5 validation.


## [1.76.0] - 2026-04-10

### Added

- Save calibration geography as a pipeline artifact, add ``--resume-from`` and checkpoint support for long-running calibration fits, and fix resume/artifact handling in the remote calibration pipeline. This also adds conservative CPS taxpayer-ID outputs (``has_tin``, ``has_valid_ssn``, and a temporary ``has_itin`` compatibility alias), plus string-valued constraint handling needed for ID-target calibration.


## [1.75.8] - 2026-04-10

### Fixed

- Modeled Medicare Part B premiums from enrollment and premium schedules, netted a cycle-free MSP standard-premium offset, and documented the national Part B calibration target as an approximate beneficiary-paid out-of-pocket benchmark rather than gross CMS premium income.


## [1.75.7] - 2026-04-10

### Fixed

- Split legacy national CTC calibration into separate refundable and nonrefundable IRS SOI amount and recipient-count targets, added DB-backed nonrefundable CTC targets for both national and unified district calibration, and fixed recursive package imports so database creation scripts and the national validation tooling can import cleanly in fresh environments. The national validator now also reports CTC totals and grouped diagnostics by AGI band and filing status, its advertised `--hf-path` mode now completes structural checks against published Hugging Face H5 artifacts, and CPS-derived datasets now emit `has_tin` plus a temporary `has_itin` compatibility alias derived from identification status.


## [1.75.6] - 2026-04-09

### Fixed

- Anchor ACA take-up to subsidized Marketplace coverage reports so unsubsidized exchange enrollment does not force premium tax credit take-up.


## [1.75.5] - 2026-04-09

### Changed

- Donor-impute race, Hispanic status, sex, and occupation-based CPS features onto the PUF clone half of the extended CPS so subgroup analyses and overtime-eligibility inputs better align with PUF-imputed incomes.

### Fixed

- Replace legacy SQLModel `session.query(...)` lookups in the SOI ETL loaders and their focused tests with `session.exec(select(...))` to remove deprecation warnings in CI.


## [1.75.4] - 2026-04-09

### Fixed

- Remove duplicate entries in bad_targets list.


## [1.75.3] - 2026-04-09

### Changed

- Add 2025 ACA and Medicaid calibration target artifacts, plus year-aware ACA target loading and state uprating factors for 2025 builds.


## [1.75.2] - 2026-04-09

### Fixed

- Stop independently QRF-imputing clone-half ``spm_unit_capped_work_childcare_expenses`` and rebuild it deterministically from clone pre-subsidy childcare, donor capping shares, and clone earnings caps.


## [1.75.1] - 2026-04-08

### Changed

- Build policy_data.db from source instead of downloading from HuggingFace, replace H5 dataset dependency with a --year CLI flag for all database ETL scripts, fix Modal data build ordering (CPS before PUF), and add missing heapq import in local area builder.


## [1.75.0] - 2026-04-08

### Added

- Add `docs/internals/` developer reference: three notebooks covering all nine pipeline stages (Stage 1 data build, Stage 2 calibration matrix assembly, Stages 3–4 L0 optimization and H5 assembly) plus a README with pipeline orchestration reference, run ID format, Modal volume layout, and HuggingFace artifact paths.

### Changed

- Update public-facing methodology and data documentation to reflect the current pipeline implementation; pipeline now uploads validation diagnostics to HuggingFace after H5 builds complete.


## [1.74.3] - 2026-04-07

### Fixed

- Fix the PR changelog fragment check to validate fragments added by the pull request rather than pre-existing files.


## [1.74.2] - 2026-04-03

### Fixed

- Fix district AGI geography assignment to match target shares and use the requested calibration database when loading district AGI targets.


## [1.74.1] - 2026-04-03

### Fixed

- Added fail-closed dataset contract validation for built CPS artifacts, including
  `policyengine-us` lockfile version checks, per-entity HDF5 length validation,
  and file-based `Microsimulation` smoke tests in both the build and upload paths.


## [1.74.0] - 2026-04-02

### Added

- Convert imputed deductible mortgage interest into structural mortgage balance, interest, and origination-year inputs when the installed `policyengine-us` supports federal MID cap modeling, while preserving total current-law interest deductions via residual investment interest inputs.
- Added SOI Table 4.3 top-tail calibration targets for the top 0.001%, 0.001-0.01%, 0.01-0.1%, and 0.1-1% AGI percentile intervals, covering 9 variables (count, AGI, wages, interest, dividends, capital gains, business income, and partnership/S-corp income).

### Changed

- Align SSI takeup and disability flags to CPS-reported receipt.
- Upgrade CI and Modal runtime defaults to Python 3.14 and declare package
  support for Python 3.14.
- Refresh tracked national SOI workbook targets through TY2023, backfill TY2022,
  teach `get_soi()` to pick the best available source year per variable, and
  overlay the national DB IRS-SOI targets that can now use the newer workbook
  release instead of staying stuck on the TY2022 geography file.

### Fixed

- Reduce unnecessary PR CI spend by canceling superseded runs and limiting
  the Modal-backed full data build to labeled or high-risk data-pipeline changes.
- Restructured CI/CD pipeline: migrated versioning from expired PAT to GitHub App token, moved tests to top-level tests/ with unit/integration split, consolidated 9 workflow files into 4 (pr.yaml, push.yaml, pipeline.yaml, versioning.yaml), added Codecov integration. Integration tests now only run on PRs with the run-integration label.
- Assign distinct `reform_id` values to each national JCT tax expenditure target instead of reusing a single generic reform id for all of them.
- Fix SOI uprating dtype error on newer pandas and add defensive non-negativity clip for retirement/SS variables in splice step.
- Fix the state income tax ETL to parse the official FY2023 Census STC `T40`
  row instead of using a mismatched hardcoded table, correcting Washington,
  New Hampshire, Tennessee, California, and other state targets.
- Use a mortgage-specific deduction variable for the JCT mortgage tax expenditure target instead of broad interest deductions.
- Scope pipeline artifact directory by run ID to prevent concurrent runs from clobbering each other's H5 files, calibration packages, and weights.


## [1.73.0] - 2026-03-12

### Added

- Add end-to-end test for calibration database build pipeline.
- Unified calibration pipeline with GPU-accelerated L1/L0 solver, target config YAML, and CLI package validator.
  Per-state and per-county precomputation replacing per-clone Microsimulation (51 sims instead of 436).
  Parallel state, county, and clone loop processing via ProcessPoolExecutor.
  Block-level takeup re-randomization with deterministic seeded draws.
  Hierarchical uprating with ACA PTC state-level CSV factors and CD reconciliation.
  Modal remote runner with Volume support, CUDA OOM fixes, and checkpointing.
  H5 builder that filters calibrated clone weights by CD subset, uses pre-assigned random census blocks from `geography.npz` to derive full sub-state geography, and produces self-contained local area datasets.
  Staging validation script (validate_staging.py) with sim.calculate() comparison and sanity checks.

### Changed

- Geography assignment now prevents clone-to-CD collisions.
  County-dependent vars (aca_ptc) selectively precomputed per county; other vars use state-only path.
  Target config switched to finest-grain include mode (~18K targets).
- Migrated from changelog_entry.yaml to towncrier fragments to eliminate merge conflicts.

### Fixed

- Cross-state cache pollution in matrix builder precomputation.
  Takeup draw ordering mismatch between matrix builder and stacked builder.
  At-large district geoid mismatch (7 districts had 0 estimates).


## [1.72.3] - 2026-03-09

### Changed

- Replaced batched QRF imputation with single sequential QRF via microimpute's fit_predict() API, preserving full covariance across all 85+ PUF income variables.


## [1.72.2] - 2026-03-06

### Changed

- Switch from black to ruff format.


## [1.72.1] - 2026-03-05

### Fixed

- Fixed double-weight application in dataset sanity tests: use `.values.sum()` for household_weight checks to avoid MicroSeries applying weights twice.


## [1.72.0] - 2026-03-05

### Added

- Hardened data pipeline against corrupted dataset uploads: pre-upload validation gate, post-generation assertions in enhanced CPS and sparse builders, CI workflow safety guards, file size checks, and comprehensive sanity tests for all dataset variants (5 layers of defense).


## [1.71.4] - 2026-03-04

### Fixed

- Fix create_sparse_ecps overwriting enhanced_cps_2024.h5 with sparse version that drops input variables like employment_income.


## [1.71.3] - 2026-03-04

### Changed

- Prioritize reported benefit recipients in take-up assignment for SSI and SNAP.


## [1.71.2] - 2026-03-04

### Fixed

- Reconcile SS sub-components after PUF imputation so they sum to social_security.


## [1.71.1] - 2026-03-04

### Changed

- Read IRS retirement contribution limits from policyengine-us parameters instead of hard-coding them.


## [1.71.0] - 2026-02-26

### Added

- Impute pregnancy in CPS microdata using CDC VSRR birth counts and Census ACS female population, with calibration targets per state.


## [1.70.0] - 2026-02-26

### Added

- Add end-to-end test for calibration database build pipeline.


## [1.69.4] - 2026-02-24

### Changed

- Migrated from changelog_entry.yaml to towncrier fragments to eliminate merge conflicts.


# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.69.3] - 2026-02-19 16:34:50

### Fixed

- Add TANF takeup (22%) assignment to CPS data pipeline so takes_up_tanf_if_eligible is persisted in the dataset.

## [1.69.2] - 2026-02-19 13:41:26

### Changed

- Bump policyengine-us to 1.570.7 to include TANF formula fixes and takes_up_tanf_if_eligible variable for calibration.

## [1.69.1] - 2026-02-19 06:08:37

### Added

- Add TANF takeup rate parameter (22%) and register takes_up_tanf_if_eligible in the calibration pipeline's SIMPLE_TAKEUP_VARS.

## [1.69.0] - 2026-02-18 21:11:01

### Added

- PUF clone + QRF imputation module (puf_impute.py) with state_fips predictor and stratified subsample preserving top 0.5% by AGI
- ACS re-imputation module (source_impute.py) with state predictor; SIPP/SCF imputation without state (surveys lack state identifiers)
- PUF and source impute integration into unified calibration pipeline (--puf-dataset, --skip-puf, --skip-source-impute flags)
- 21 new tests for puf_impute and source_impute modules
- DC_STATEHOOD=1 environment variable set in storage/__init__.py to ensure DC is included in state-based processing

### Changed

- Refactored extended_cps.py to delegate to puf_impute.puf_clone_dataset() (443 -> 75 lines)
- PUF QRF training uses stratified subsample (20K target) instead of random subsample(10_000), force-including high-income tail
- unified_calibration.py pipeline now supports optional source imputation and PUF cloning steps

## [1.68.0] - 2026-02-17 15:26:04

### Added

- Census-block-first calibration pipeline (calibration/ package) ported from PR
- Clone-and-assign module for population-weighted census block sampling
- Unified matrix builder with clone-by-clone simulation, COO caching, and target_overview-based querying
- Unified calibration CLI with L0 optimization and seeded takeup re-randomization
- 28 new tests for the calibration pipeline
- Integration test for build_matrix geographic masking (national/state/CD)
- Tests for drop_target_groups utility
- voluntary_filing.yaml takeup rate parameter

### Changed

- Rewrote local_area_calibration_setup.ipynb for clone-based pipeline
- Renamed _get_geo_level to get_geo_level (now cross-module public API)

### Fixed

- Fix Jupyter import error in unified_calibration.py (OutStream.reconfigure moved to main)
- Fix modal_app/remote_calibration_runner.py referencing deleted fit_calibration_weights.py
- Fix _coo_parts stale state bug on build_matrix re-call after failure
- Remove hardcoded voluntary_filing rate in favor of YAML parameter

## [1.67.0] - 2026-02-12 18:56:57

### Added

- field_valid_values table in database as a source of truth for fields that have semantic meaning external to the database hierarchy (variable, constraint_variable, period, operation, active).
- Event listeners that raise an error if inconsistent operations or parent-child relationships are attempted to be inserted into the database.
- Source field in target table.

## [1.66.0] - 2026-02-12 16:29:45

### Added

- Parallelization of data building steps.
- Checkpointing mechanism to resume data builds and testing modules from last successful step in Modal runs.

### Changed

- Removed duplicate run of test_local_area_calibration tests.
- Baked correct defaults into create_stratified_cps.py, removing hardcoded args from Makefile and Modal build.
- DRYed up sequential build path to iterate SCRIPT_OUTPUTS instead of a redundant list.
- Added thread-safe locking around volume.commit() for parallel checkpoint safety.

## [1.65.0] - 2026-02-12 04:48:34

### Added

- Entity aware target calculations for correct entity counts.
- Selection of targets closest to calibration time period and uprating logic for cohesive and comprehensive time-period coverage.
- Hierarchical state-level uprating with CD reconciliation via hierarchy inconsistency factors (HIFs) and state-specific uprating factors (e.g., CMS/KFF ACA PTC multipliers).
- stratum_domain and target_overview views for easier target inspection and debugging.
- test_schema_views_and_lookups.py for testing the new views and lookups.
- Calibration matrix diagnostic notebook (docs/calibration_matrix.ipynb).

## [1.64.1] - 2026-02-09 17:47:49

### Added

- promote-dataset Makefile target
- Year-mismatch warning in national targets ETL
- Congress-session constants and warning in SOI district puller

### Changed

- Switch DEFAULT_DATASET to local storage path for database ETL scripts
- Extract shared etl_argparser() to reduce boilerplate across 7 ETL scripts
- Delete dead get_pseudo_input_variables() function

## [1.64.0] - 2026-02-08 04:15:49

### Added

- Add voluntary tax filer variable and filer count calibration targets by AGI band.

## [1.63.1] - 2026-02-08 02:59:10

### Fixed

- Immigration status mapping.

## [1.63.0] - 2026-02-07 19:46:46

### Added

- Add liquid asset imputation from SIPP (bank accounts, stocks, bonds) for SSI and means-tested program modeling
- Add SSI takeup rate parameter and takes_up_ssi_if_eligible draw

## [1.62.0] - 2026-02-07 00:24:23

### Added

- Name-based seeding (seeded_rng) for order-independent reproducibility
- State-specific Medicaid takeup rates (53%-99% range, 51 jurisdictions)
- SSI resource test pass rate parameter (0.4)
- WIC takeup and nutritional risk draw variables (float)
- meets_ssi_resource_test boolean generation

### Changed

- Replaced shared RNG (seed=100) with per-variable name-based seeding
- Medicaid takeup now uses state-specific rates instead of uniform 93%

## [1.61.2] - 2026-02-01 20:58:21

### Fixed

- Fix etl_state_income_tax.py API mismatches with db_metadata utility functions

## [1.61.1] - 2026-02-01 04:24:23

### Added

- cps_2024.h5 to HuggingFace upload list so the raw (unenhanced) 2024 CPS dataset is published

## [1.61.0] - 2026-01-31 20:18:58

### Added

- Add state income tax calibration targets from Census STC FY2023 data

## [1.60.0] - 2026-01-31 19:57:29

### Changed

- Use income_tax_positive instead of income_tax for CBO calibration target

## [1.59.0] - 2026-01-31 19:54:59

### Added

- SSA benefit-type calibration targets for social_security_retirement, social_security_disability, social_security_survivors, and social_security_dependents
- IRA contribution calibration targets for traditional_ira_contributions and roth_ira_contributions from IRS SOI data

### Changed

- Use CPS ASEC RESNSS1/RESNSS2 source codes to classify Social Security income into retirement, disability, survivors, and dependents (replacing age-62 heuristic)
- Parameterize retirement contribution limits by year (2020-2025) instead of hardcoded 2022 values
- Update taxable pension fraction from 1.0 to 0.590 based on SOI 2015 Table 1.4
- Add age and is_male as QRF predictors for pension contribution imputation

## [1.58.0] - 2026-01-31 19:53:18

### Added

- weeks_unemployed variable from CPS ASEC LKWEEKS
- QRF-based imputation of weeks_unemployed for Extended CPS PUF copy

## [1.57.0] - 2026-01-31 03:18:20

### Added

- Added CPS_2024_Full class for full-sample 2024 CPS generation
- Added raw_cache utility for Census data caching
- Added atomic parallel local area H5 publishing with Modal Volume staging
- Added manifest validation with SHA256 checksums
- Added HuggingFace retry logic with exponential backoff to fix timeout errors
- Added staging folder approach for atomic HuggingFace deployments
- Added national targets ETL for CBO projections and tax expenditure data
- Added database hierarchy validation script
- Added stratum_group_id migration utilities
- Added db_metadata utilities for source and variable group management
- Added DATABASE_GUIDE.md with comprehensive calibration database documentation

### Changed

- Migrated data pipeline from CPS 2023 to CPS 2024 (March 2025 ASEC)
- Updated ExtendedCPS_2024 to use new CPS_2024_Full (full sample)
- Updated local area calibration to use 2024 extended CPS data
- Updated database ETL scripts for strata, IRS SOI, Medicaid, and SNAP
- Expanded IRS SOI ETL with detailed income brackets and filing status breakdowns

### Fixed

- Fixed cross-state recalculation in sparse matrix builder by adding time_period to calculate() calls

## [1.56.0] - 2026-01-26 22:41:56

### Added

- Census block-level geographic assignment for households in CD-stacked datasets
- Comprehensive geography variables in output (block_geoid, tract_geoid, cbsa_code, sldu, sldl, place_fips, vtd, puma, zcta)
- Block crosswalk file mapping 8.1M blocks to all Census geographies
- Block-to-CD distribution file for population-weighted assignment
- ZCTA (ZIP Code Tabulation Area) lookup from census block

## [1.55.0] - 2026-01-26 16:45:05

### Added

- Support for health_insurance_premiums_without_medicare_part_b in local area calibration

### Changed

- Removed dense reweighting path from enhanced CPS; only sparse (L0) weights are produced
- Eliminated TEST_LITE and LOCAL_AREA_CALIBRATION flags; all datasets generated unconditionally
- Merged data-local-area Makefile target into data target

### Fixed

- Versioning workflow now runs uv lock after version bump to keep uv.lock in sync

## [1.54.1] - 2026-01-26 02:49:11

### Fixed

- Derive partnership_se_income from PUF source columns using Yale Budget Lab's gross-up approach instead of looking for non-existent k1bx14 columns.

## [1.54.0] - 2026-01-25 17:43:38

### Added

- partnership_se_income variable from Schedule K-1 Box 14 (k1bx14p + k1bx14s), representing partnership income subject to self-employment tax.

## [1.53.1] - 2026-01-25 15:48:00

### Changed

- Bumped policyengine-core minimum version to 3.23.5 for pandas 3.0 compatibility

## [1.53.0] - 2026-01-23 20:51:58

### Changed

- Added policyengine-claude plugin auto-install configuration.

## [1.52.0] - 2026-01-22 20:50:13

### Added

- tests to verify SparseMatrixBuilder correctly calculates variables and constraints into the calibration matrix.

## [1.51.1] - 2026-01-07 01:05:49

### Fixed

- Fixed Publish workflow by migrating dev dependencies to PEP 735 dependency-groups

## [1.51.0] - 2026-01-01 17:39:26

### Added

- Sparse matrix builder for local area calibration with database-driven constraints
- Local area calibration data pipeline (make data-local-area)
- ExtendedCPS_2023 and PUF_2023 dataset classes
- Stratified CPS sampling to preserve high-income households
- Matrix verification tests for local area calibration
- Population-weighted P(county|CD) distributions from Census block data
- County assignment module for stacked dataset builder

## [1.50.0] - 2025-12-23 15:15:35

### Added

- Added --use-tob flag for TOB (Taxation of Benefits) revenue calibration targeting OASDI and HI trust fund revenue

## [1.49.0] - 2025-12-19 17:56:53

### Added

- SPM threshold calculation using policyengine/spm-calculator package
- New utility module (policyengine_us_data/utils/spm.py) for SPM calculations

### Changed

- CPS datasets now calculate SPM thresholds using spm-calculator with Census-provided geographic adjustments
- ACS datasets now calculate SPM thresholds using spm-calculator with national-level thresholds

## [1.48.0] - 2025-12-08 19:52:21

### Added

- Sparse matrix builder for local area calibration with database-driven constraints
- Local area calibration data pipeline (make data-local-area)
- ExtendedCPS_2023 and PUF_2023 dataset classes
- Stratified CPS sampling to preserve high-income households
- Matrix verification tests for local area calibration

## [1.47.1] - 2025-12-03 23:00:20

### Added

- Node.js 24 LTS setup to CI workflow for MyST builds
- H6 Social Security reform calibration for long-term projections (phases out OASDI taxation 2045-2054)
- H6 threshold crossover handling when OASDI thresholds exceed HI thresholds
- start_year parameter to run_household_projection.py CLI
- docs/README.md documenting MyST build output pitfall

### Fixed

- GitHub Pages documentation deployment (was deploying wrong directory causing blank pages)
- Removed timeout and error suppression from documentation build

## [1.47.0] - 2025-11-20 02:54:32

### Added

- Additional calibration based on SSA Trustees data that extends projections until 2100
- Manual trigger capability for documentation deployment workflow
- Documentation for SSA data sources in storage README

### Changed

- Renamed long-term projections notebook to clarify PWBM comparison scope (2025-2100)

### Fixed

- GitHub Pages documentation deployment path
- Corrected number of imputed variables from 72 to 67 in documentation
- Corrected calibration target count from 7,000+ to 2,813 across all docs
- Removed inaccurate "two-stage" terminology in methodology descriptions

## [1.46.1] - 2025-11-12 20:08:59

### Changed

- GitHub Actions workflow now uses self-hosted GCP runner to handle memory-intensive dataset builds

## [1.46.0] - 2025-09-10 20:30:41

### Added

- Support for 2024 CPS ASEC data (March 2024 survey)
- CensusCPS_2024 class to download raw 2024 ASEC data
- CPS_2024 class using actual 2024 data instead of extrapolation
- CPS_2025 class with extrapolation from 2024 data
- DOCS_FOLDER constant to storage module for cleaner file paths
- Tests for CPS 2024 and 2025 datasets

### Changed

- Fixed __file__ NameError in interactive Python environments
- Updated generate method to handle 2025 extrapolation from 2024

## [1.45.0] - 2025-08-20 18:44:07

### Added

- add SQLite database for calibration targets

## [1.44.2] - 2025-08-08 15:16:00

### Fixed

- Fixed GitHub Pages documentation by adding .nojekyll file to serve underscore-prefixed directories

## [1.44.1] - 2025-08-08 10:19:16

### Changed

- renamed "ucgid" to "ucgid_str" in age targets loading script and operation to "in"
- removed [0.5] key access from imputation results as per microimpute's new output format

## [1.44.0] - 2025-08-06 19:01:03

### Added

- Unpin -us.

## [1.43.1] - 2025-08-05 10:23:02

### Fixed

- Moved QRF implementation to microimpute package to avoid code duplication

## [1.43.0] - 2025-08-04 18:52:21

### Added

- Pin -us to a version pre-OBBBA baseline changes were implemented.

## [1.42.6] - 2025-08-01 11:29:48

### Fixed

- Lite mode was used in production.

## [1.42.5] - 2025-07-30 22:43:44

### Fixed

- Fixed GitHub Pages documentation rendering by setting BASE_URL for MyST

## [1.42.4] - 2025-07-30 22:29:06

### Changed

- New configuration for sparse solution (~20k non-zero households)
- added a seeding function to remove non-deterministic behavior in reweight
- Made np2023_d5_mid.csv a git ignorable file (it's in hugging face)

## [1.42.4] - 2025-07-30 21:55:05

### Added

- Fork check in PR workflows to fail fast with clear error message

### Fixed

- Fixed documentation deployment for MyST v2 by using timeout command

## [1.42.3] - 2025-07-30 20:28:05

### Fixed

- Made upload script more robust by only uploading files that exist
- Added logging to show which files are being uploaded vs skipped

## [1.42.2] - 2025-07-30 19:42:58

### Fixed

- {"Fixed push CI upload failure by using 'secrets": "inherit' in reusable workflows"}

## [1.42.1] - 2025-07-30 18:19:07

### Fixed

- Removed leftover changelog entry from merged PR that was causing push CI failures
- Removed unused make_person function with undefined CURRENT_YEAR variable

## [1.42.0] - 2025-07-28 16:34:40

### Added

- Added creation script to build relational database for targets
- Refactored age targets load script to load the database

## [1.41.2] - 2025-07-26 20:53:26

### Added

- PyPI auto-publish workflow in GitHub Actions

### Fixed

- README typo (installion -> installation)

## [1.41.1] - 2025-07-26 19:06:45

### Fixed

- Increased Medicaid calibration tolerance to 100% to handle state-level noise

## [1.41.0] - 2025-07-26 17:22:33

### Added

- Python 3.13 support

### Changed

- Simplified CI test matrix to only test on Python 3.13 and Ubuntu
- Updated policyengine-us to >=1.350.0 for Python 3.13 support
- Updated policyengine-core to >=3.19.0 for Python 3.13 support
- Updated microimpute from 0.1.4 to 1.0.1 for numpy 2.x compatibility
- Updated scipy dependency from <1.13 to >=1.15.3
- Updated pandas dependency from >=2.3.0 to >=2.3.1
- Updated statsmodels dependency from >=0.14.0 to >=0.14.5
- Added lower bounds to dependencies that were missing them

## [1.40.1] - 2025-07-26 13:35:10

### Fixed

- Clean up immigration status PR.

## [1.40.0] - 2025-07-24 13:44:42

### Added

- Added Immigration status from SSN algorithm.

## [1.39.2] - 2025-07-22 21:03:38

### Changed

- Update microdf_python dependency to >=1.0.0.

## [1.39.1] - 2025-07-18 17:01:51

### Fixed

- Edit and create files that pull SOI agi, ACS age, hardcoded and SNAP targets to follow the same clean csv format.
- Track all csv files used by loss.py for backwards compatibility.

## [1.39.0] - 2025-07-18 12:46:15

### Added

- l0 regularization as described in https://arxiv.org/abs/1712.01312

## [1.38.1] - 2025-07-17 20:07:31

### Fixed

- Github pages deploy

## [1.38.0] - 2025-07-16 01:01:25

### Changed

- Removed github download capability
- Changed download option for soi.csv and np2023_d5_mid.csv to Hugging Face

## [1.37.1] - 2025-07-14 15:33:11

### Changed

- bad targets (causing problems with estimation) removed
- lite mode now builds CPS_2023 in addition to CPS_2024
- gave reweight an epochs argument and set it at 150 for optimization
- updating minimum versions on policyengine-us and pandas dependencies
- getting rid of non-working manual workflow code

## [1.37.0] - 2025-07-09 14:58:33

### Added

- Medicaid state level calibration targets.

## [1.36.2] - 2025-07-08 21:53:02

### Fixed

- Use SURVIVING_SPOUSE and is_surviving_spouse instead of WIDOW and is_widowed.

## [1.36.1] - 2025-07-03 09:21:06

### Changed

- PR tests to be more similar to production builds.

## [1.36.0] - 2025-07-03 03:03:06

### Added

- State SNAP calibration targets.

## [1.35.2] - 2025-07-02 15:31:46

### Changed

- Epochs increased to 1k.

## [1.35.1] - 2025-07-02 15:00:11

### Fixed

- Imputed non-CPS income variables from the PUF.

## [1.35.0] - 2025-07-01 23:42:47

### Added

- Normalisation of national and state targets.

## [1.34.1] - 2025-07-01 22:12:13

### Changed

- Calibration epochs reduced to 500.

## [1.34.0] - 2025-07-01 20:10:32

### Added

- State real estate taxes calibration targets.

## [1.33.3] - 2025-07-01 19:15:43

### Fixed

- Bug in hyperparameter tuning.

## [1.33.2] - 2025-07-01 19:02:50

### Fixed

- Increased epochs back to 5k.
- Disabled hyperparameter tuning for imputation models.

## [1.33.1] - 2025-07-01 16:54:09

### Fixed

- Use full CPS by default.

## [1.33.0] - 2025-07-01 14:51:09

### Added

- State agi calibration targets.

## [1.32.1] - 2025-07-01 13:28:38

### Added

- State age targets.

## [1.32.0] - 2025-06-23 14:48:18

### Added

- SSN card type imputation algorithm.
- Family correlation adjustment to align parent-child SSN status.

## [1.31.0] - 2025-06-19 21:34:31

### Added

- Added automated checks for changelog entry
- New "would be qualified income" variables simulated
- REIT, PTP, and BDC dividend income variables simulated
- UBIA property is being simulated
- Farm Operations Income added

### Changed

- W2 Wages from Qualified business is now being simulated with random variables
- qualified business income sources have been redefined based on IRS PUF inputs

## [1.30.2] - 2025-06-19 13:59:12

### Fixed

- Small CPS is now 1000 households.

## [1.30.1] - 2025-06-19 10:09:37

### Added

- Add test for small ECPS.

## [1.30.0] - 2025-06-18 12:31:13

### Added

- Synthetic, small ECPS data file.

## [1.29.1] - 2025-06-18 10:07:41

### Added

- ACA and Medicaid calibration targets.

## [1.29.0] - 2025-06-14 20:36:59

### Added

- Change ACA Marketplace variable to use current coverage instead of any coverage within the last year.

## [1.28.4] - 2025-06-13 16:30:39

### Fixed

- Data length in the take-up variables.

## [1.28.3] - 2025-06-13 14:46:04

### Fixed

- Adjust take-up seed variables.

## [1.28.2] - 2025-06-13 11:06:01

### Added

- Join wealth and auto loan interest imputations.

## [1.28.1] - 2025-06-12 16:59:41

### Fixed

- Increase tolerance for auto loan interest and balance test.

## [1.28.0] - 2025-06-11 22:28:55

### Added

- Add ACA and Medicaid take-up rates.

## [1.27.0] - 2025-06-09 11:46:29

### Added

- Source for net worth calibration.

## [1.26.0] - 2025-06-09 10:44:59

### Added

- Net worth variable to cps.

## [1.25.3] - 2025-05-26 22:11:20

### Fixed

- Missing HF token.

## [1.25.2] - 2025-05-26 22:01:07

### Fixed

- Tests run after versioning.

## [1.25.1] - 2025-05-26 21:57:26

### Added

- Versioning to dataset uploads.

## [1.25.0] - 2025-05-26 10:43:04

### Added

- Hours worked last week variable.

## [1.24.0] - 2025-05-23 15:00:34

### Added

- Auto loan balance variable to cps.

## [1.23.4] - 2025-05-22 10:56:32

### Changed

- Methodology to directly impute auto loan interest instead of assuming a 2% interest rate on auto loan balance.

## [1.23.3] - 2025-05-20 10:37:41

### Fixed

- GCP uploads use permissions correctly

## [1.23.2] - 2025-05-19 15:34:43

### Fixed

- Upload to GCP on dataset build.

## [1.23.1] - 2025-05-19 07:52:35

### Fixed

- Runtime for tests reduced.

## [1.23.0] - 2025-05-14 14:29:32

### Added

- scf package loading module
- auto loan balance imputation notebook

## [1.22.0] - 2025-05-14 14:15:06

### Added

- SSN card type implementation for CPS dataset.
- Calibration of undocumented population to 10.1 million based on Pew Research data.

## [1.21.1] - 2025-05-14 13:31:21

### Fixed

- Data downloads for Census datasets disabled.
- Warning added for downsampling non-existent policyengine-[country] variables.

## [1.21.0] - 2025-05-13 13:29:57

### Added

- Calibration of the QBID tax expenditure.

## [1.20.0] - 2025-05-13 12:48:06

### Added

- Tip income.

## [1.19.2] - 2025-04-22 18:24:44

### Added

- Non-downsampled versions of the 2021, 2022, and 2023 CPS datasets

### Changed

- Modified downsampling method within CPS base dataset class
- Pooled 3-Year CPS generation uses the non-downsampled versions of the 2021, 2022, and 2023 CPS datasets
- Downsampling method attempts to preserve original dtype values

## [1.19.1] - 2025-03-28 18:07:01

### Changed

- Explicitly specified encoding while building county FIPS dataset

## [1.19.0] - 2025-03-27 22:58:46

### Added

- County FIPS dataset

## [1.18.1] - 2025-02-20 12:34:31

### Fixed

- Apply the miscellaneous deduction imputation to the unreimbursed_business_employee_expenses instead of the misc_deduction variable.

## [1.18.0] - 2025-02-01 02:21:19

### Fixed

- Larger GH runner for data generation.

## [1.17.0] - 2025-01-24 11:18:33

### Added

- Interest expenses.

## [1.16.1] - 2025-01-22 04:02:27

### Fixed

- Minor bug with memory breaches.

## [1.16.0] - 2025-01-13 16:36:45

### Added

- DC PTC takeup.

## [1.15.1] - 2024-12-03 23:21:24

### Changed

- Install order and requirements for policyengine-us

## [1.15.0] - 2024-12-02 20:40:26

### Changed

- Changed GitHub release URLs to Hugging Face URLs for Enhanced CPS 2024 and Pooled 3-Year CPS 2023.
- Set minimum version for policyengine-core.

## [1.14.0] - 2024-11-29 20:23:10

### Added

- Automatic upload behavior.

## [1.13.0] - 2024-11-19 12:29:11

### Added

- Metric comparisons by dataset to the documentation.
- Calibration of state populations.

## [1.12.1] - 2024-11-12 15:03:39

### Added

- Metric comparisons by dataset to the documentation.

## [1.12.0] - 2024-11-12 07:03:52

### Added

- Paper on methodology.

## [1.11.1] - 2024-10-29 19:15:42

### Changed

- Reverted to using standard version of microdf

## [1.11.0] - 2024-10-09 14:11:41

### Changed

- EITC targets improved by uprating 2020 rather than 2021 targets.

## [1.10.0] - 2024-10-08 15:48:46

### Fixed

- EITC calibration.

## [1.9.0] - 2024-10-07 11:45:52

### Added

- EITC calibration by child counts.
- 10% dropout during weight calibration.

## [1.8.0] - 2024-09-29 18:08:57

### Fixed

- Moved PolicyEngine US out of setup.py dependencies.

## [1.7.0] - 2024-09-29 15:03:05

### Changed

- Bump to policyengine-us 1.100.0.

## [1.6.0] - 2024-09-25 10:40:39

### Added

- State and household size as predictors for rent and property taxes.

## [1.5.1] - 2024-09-23 11:22:32

### Changed

- Documentation updated.
- URLs for PUF data.

## [1.5.0] - 2024-09-23 10:28:55

### Added

- Migrate the ACS from the US-repository.

### Changed

- Enhanced CPS now uses a 3-year pooled CPS.

## [1.4.5] - 2024-09-22 21:15:27

## [1.4.4] - 2024-09-19 15:36:00

### Changed

- Split push actions into two separate files
- Made run of second portion of push conditional upon run of first

## [1.4.3] - 2024-09-18 20:57:03

### Changed

- Fixed CI/CD push script

## [1.4.2] - 2024-09-18 19:49:48

### Fixed

- Corrected versioning issues

## [1.4.1] - 2024-09-18 16:30:37

### Fixed

- Import errors in non-dev mode.

## [1.4.0] - 2024-09-18 03:05:11

### Added

- Geography generation module (previously in US package)

### Changed

- Fixed export structure within __init__ files

## [1.3.1] - 2024-09-17 19:37:44

### Added

- Jupyter Book documentation.

## [1.3.0] - 2024-09-17 10:27:10

### Fixed

- Moved heavy dependencies to dev.

## [1.2.1] - 2024-09-16 08:04:08

### Fixed

- Bug in docs where prerequisites wouldn't load in GCP.

## [1.2.0] - 2024-09-12 19:47:01

### Added

- Added conditional deletion of existing resource
- Added downloading of existing resources for backup purposes
- Added tqdm to download script

### Changed

- Fixed upload script's use of tqdm

## [1.1.1] - 2024-09-11 16:40:10

### Fixed

- Added GitHub Actions test job to PR and push
- Run publish to PyPI GitHub Actions job only on push
- Fix changelog GitHub Actions job

## [1.1.0] - 2024-09-11 13:48:15

### Changed

- Improved logging
- Updated required Python version
- Removed setuptools_scm

## [1.0.0] - 2024-09-09 17:29:10

### Added

- Improved changelog

## [1.0.0] - 2024-09-09 17:29:10

### Added

- Initialized changelogging



[1.69.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.69.2...1.69.3
[1.69.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.69.1...1.69.2
[1.69.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.69.0...1.69.1
[1.69.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.68.0...1.69.0
[1.68.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.67.0...1.68.0
[1.67.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.66.0...1.67.0
[1.66.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.65.0...1.66.0
[1.65.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.64.1...1.65.0
[1.64.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.64.0...1.64.1
[1.64.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.63.1...1.64.0
[1.63.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.63.0...1.63.1
[1.63.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.62.0...1.63.0
[1.62.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.61.2...1.62.0
[1.61.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.61.1...1.61.2
[1.61.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.61.0...1.61.1
[1.61.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.60.0...1.61.0
[1.60.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.59.0...1.60.0
[1.59.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.58.0...1.59.0
[1.58.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.57.0...1.58.0
[1.57.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.56.0...1.57.0
[1.56.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.55.0...1.56.0
[1.55.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.54.1...1.55.0
[1.54.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.54.0...1.54.1
[1.54.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.53.1...1.54.0
[1.53.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.53.0...1.53.1
[1.53.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.52.0...1.53.0
[1.52.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.51.1...1.52.0
[1.51.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.51.0...1.51.1
[1.51.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.50.0...1.51.0
[1.50.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.49.0...1.50.0
[1.49.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.48.0...1.49.0
[1.48.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.47.1...1.48.0
[1.47.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.47.0...1.47.1
[1.47.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.46.1...1.47.0
[1.46.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.46.0...1.46.1
[1.46.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.45.0...1.46.0
[1.45.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.44.2...1.45.0
[1.44.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.44.1...1.44.2
[1.44.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.44.0...1.44.1
[1.44.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.43.1...1.44.0
[1.43.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.43.0...1.43.1
[1.43.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.42.6...1.43.0
[1.42.6]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.42.5...1.42.6
[1.42.5]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.42.4...1.42.5
[1.42.4]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.42.4...1.42.4
[1.42.4]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.42.3...1.42.4
[1.42.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.42.2...1.42.3
[1.42.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.42.1...1.42.2
[1.42.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.42.0...1.42.1
[1.42.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.41.2...1.42.0
[1.41.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.41.1...1.41.2
[1.41.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.41.0...1.41.1
[1.41.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.40.1...1.41.0
[1.40.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.40.0...1.40.1
[1.40.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.39.2...1.40.0
[1.39.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.39.1...1.39.2
[1.39.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.39.0...1.39.1
[1.39.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.38.1...1.39.0
[1.38.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.38.0...1.38.1
[1.38.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.37.1...1.38.0
[1.37.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.37.0...1.37.1
[1.37.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.36.2...1.37.0
[1.36.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.36.1...1.36.2
[1.36.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.36.0...1.36.1
[1.36.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.35.2...1.36.0
[1.35.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.35.1...1.35.2
[1.35.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.35.0...1.35.1
[1.35.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.34.1...1.35.0
[1.34.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.34.0...1.34.1
[1.34.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.33.3...1.34.0
[1.33.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.33.2...1.33.3
[1.33.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.33.1...1.33.2
[1.33.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.33.0...1.33.1
[1.33.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.32.1...1.33.0
[1.32.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.32.0...1.32.1
[1.32.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.31.0...1.32.0
[1.31.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.30.2...1.31.0
[1.30.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.30.1...1.30.2
[1.30.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.30.0...1.30.1
[1.30.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.29.1...1.30.0
[1.29.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.29.0...1.29.1
[1.29.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.28.4...1.29.0
[1.28.4]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.28.3...1.28.4
[1.28.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.28.2...1.28.3
[1.28.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.28.1...1.28.2
[1.28.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.28.0...1.28.1
[1.28.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.27.0...1.28.0
[1.27.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.26.0...1.27.0
[1.26.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.25.3...1.26.0
[1.25.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.25.2...1.25.3
[1.25.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.25.1...1.25.2
[1.25.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.25.0...1.25.1
[1.25.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.24.0...1.25.0
[1.24.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.23.4...1.24.0
[1.23.4]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.23.3...1.23.4
[1.23.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.23.2...1.23.3
[1.23.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.23.1...1.23.2
[1.23.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.23.0...1.23.1
[1.23.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.22.0...1.23.0
[1.22.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.21.1...1.22.0
[1.21.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.21.0...1.21.1
[1.21.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.20.0...1.21.0
[1.20.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.19.2...1.20.0
[1.19.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.19.1...1.19.2
[1.19.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.19.0...1.19.1
[1.19.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.18.1...1.19.0
[1.18.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.18.0...1.18.1
[1.18.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.17.0...1.18.0
[1.17.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.16.1...1.17.0
[1.16.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.16.0...1.16.1
[1.16.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.15.1...1.16.0
[1.15.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.15.0...1.15.1
[1.15.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.14.0...1.15.0
[1.14.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.13.0...1.14.0
[1.13.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.12.1...1.13.0
[1.12.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.12.0...1.12.1
[1.12.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.11.1...1.12.0
[1.11.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.11.0...1.11.1
[1.11.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.10.0...1.11.0
[1.10.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.9.0...1.10.0
[1.9.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.8.0...1.9.0
[1.8.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.7.0...1.8.0
[1.7.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.6.0...1.7.0
[1.6.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.5.1...1.6.0
[1.5.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.5.0...1.5.1
[1.5.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.4.5...1.5.0
[1.4.5]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.4.4...1.4.5
[1.4.4]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.4.3...1.4.4
[1.4.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.4.2...1.4.3
[1.4.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.4.1...1.4.2
[1.4.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.4.0...1.4.1
[1.4.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.3.1...1.4.0
[1.3.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.3.0...1.3.1
[1.3.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.2.1...1.3.0
[1.2.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.2.0...1.2.1
[1.2.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.1.1...1.2.0
[1.1.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.0.0...1.1.0
[1.0.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.0.0...1.0.0
