# Enhanced CPS: Combining Survey and Administrative Data for Microsimulation

The Enhanced CPS dataset provides the microdata foundation for PolicyEngine's US tax and benefit microsimulation model. This documentation describes how we combine the Current Population Survey with IRS administrative data to create a dataset that accurately represents American households for policy analysis.

## Quick Start

```python
from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.enhanced_cps import EnhancedCPS

# Load the Enhanced CPS for 2024
sim = Microsimulation(dataset=EnhancedCPS(2024))

# Calculate poverty rate
poverty = sim.calculate("in_poverty", period=2024)
poverty_rate = poverty.mean()
print(f"Poverty rate: {poverty_rate:.1%}")
```

## Documentation Structure

This documentation covers:

- **[Overview](overview)**: Key features, use cases, and data sources
- **[Methodology](methodology)**: Two-stage enhancement process (imputation and reweighting)
- **[Technical Details](technical_details)**: Implementation details for researchers
- **[Validation](validation)**: Performance metrics against official statistics
- **[Results](results)**: Example analyses and policy simulations
- **[SSN Status Imputation](ssn_statuses_imputation)**: Modeling undocumented populations

## Contributing

The Enhanced CPS is open source and welcomes contributions. Visit our [GitHub repository](https://github.com/PolicyEngine/policyengine-us-data) to:
- Report issues
- Submit improvements
- Access source code
- Download datasets

