# Overview

The Enhanced CPS dataset provides a comprehensive microdata foundation for US tax and benefit microsimulation. By combining the demographic detail of the Current Population Survey with the tax precision of the IRS Public Use File, we create a dataset that enables accurate policy analysis across the full spectrum of government programs.

## Key Features

- **Demographic detail**: Preserves the CPS's household structure, state identifiers, and demographic characteristics
- **Tax precision**: Incorporates 72 detailed tax variables from IRS administrative data
- **Administrative consistency**: Calibrated to over 7,000 targets from six authoritative sources
- **Open source**: All code, data, and documentation freely available
- **Reproducible**: Full pipeline from raw data to final weights

## Use Cases

The Enhanced CPS supports:

- Federal tax reform analysis
- State and local tax modeling
- Benefit program evaluation
- Distributional impact assessment
- Cross-program interaction studies

## Data Sources

### Primary Sources
1. **Current Population Survey (CPS)**: Annual Social and Economic Supplement providing household demographics
2. **IRS Public Use File (PUF)**: De-identified tax return data with detailed income and deduction information

### Calibration Sources
- IRS Statistics of Income: 5,300+ income distribution targets
- Census population projections: Age and state demographics
- Congressional Budget Office: Program participation totals
- Joint Committee on Taxation: Tax expenditure estimates
- Treasury Department: Revenue projections
- Healthcare spending data: Age-specific expenditures

## Access

The Enhanced CPS dataset is available through:
- Direct download from [GitHub releases](https://github.com/PolicyEngine/policyengine-us-data/releases)
- Python API via `policyengine-us-data` package
- Integration with PolicyEngine microsimulation model

## Citation

If you use this dataset in your research, please cite:

```
Woodruff, N. and Ghenis, M. (2025). "Enhanced CPS: A Microsimulation Dataset 
Combining Survey and Administrative Data." PolicyEngine Working Paper.
```