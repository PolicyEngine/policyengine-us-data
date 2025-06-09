# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
