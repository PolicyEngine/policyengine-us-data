# Reproduction Guide for Enhanced CPS Dataset

This guide provides step-by-step instructions for reproducing the Enhanced CPS dataset and validation results.

## Prerequisites

### System Requirements
- **Memory**: Minimum 16GB RAM, recommended 32GB for full dataset generation
- **Storage**: At least 50GB free disk space
- **OS**: Linux (Ubuntu 20.04+), macOS (11+), or Windows with WSL2
- **Time**: Full reproduction takes approximately 4-6 hours

### Software Dependencies
```bash
# Python 3.9-3.11 required
python --version

# Install system dependencies
# macOS
brew install gcc git-lfs

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential git-lfs

# Install LaTeX (required for paper generation)
# macOS
brew install --cask mactex

# Ubuntu/Debian  
sudo apt-get install texlive-full
```

## Data Access Setup

### 1. CPS Data Access
The CPS data is publicly available but requires specific version:
```bash
# CPS 2022 ASEC (March supplement)
# Downloaded automatically via Census API
# No authentication required
```

### 2. PUF Data Access
The IRS Public Use File requires prior approval:
1. Visit https://www.irs.gov/statistics/soi-tax-stats-individual-public-use-microdata-files
2. Complete data use agreement
3. Download 2015 PUF file
4. Place in `data/raw/puf_2015.csv`

### 3. API Credentials
```bash
# Create .env file
cat > .env << EOF
# GitHub token for downloading CPS extracts
POLICYENGINE_GITHUB_MICRODATA_AUTH_TOKEN=your_github_token

# Census API key (free registration)
CENSUS_API_KEY=your_census_api_key
EOF
```

To obtain tokens:
- GitHub: Create at https://github.com/settings/tokens (no special permissions needed)
- Census: Register at https://api.census.gov/data/key_signup.html

## Installation

```bash
# Clone repository
git clone https://github.com/policyengine/policyengine-us-data.git
cd policyengine-us-data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
make install

# Verify installation
python -c "import policyengine_us_data; print('Installation successful')"
```

## Data Generation

### Quick Test (Subset)
```bash
# Generate small test dataset (10 minutes, 4GB RAM)
make test-data

# This creates:
# - data/enhanced_cps_2022_test.h5 (1,000 households)
# - Validates basic pipeline functionality
```

### Full Dataset
```bash
# Generate complete dataset (4-6 hours, 32GB RAM)
make data

# Monitor progress
tail -f logs/enhancement_*.log

# Expected outputs:
# - data/extended_cps_2022.h5 (2.1GB)
# - data/enhanced_cps_2022.h5 (2.3GB)
# - logs/calibration_targets.csv
```

### Troubleshooting Data Generation

Common issues and solutions:

1. **Memory Error**
   ```
   Error: Unable to allocate array with shape...
   ```
   Solution: Use subset generation or increase swap space

2. **API Rate Limit**
   ```
   Error: 403 Forbidden from api.github.com
   ```
   Solution: Add GitHub token to .env file

3. **Missing PUF File**
   ```
   Error: FileNotFoundError: data/raw/puf_2015.csv
   ```
   Solution: Download PUF data following instructions above

## Validation

### Generate Validation Results
```bash
# Run all validation scripts
make validate

# This generates:
# - tables/tax_unit_metrics.tex
# - tables/household_metrics.tex
# - figures/income_distribution.png
# - validation_report.html
```

### Expected Results

Key metrics to verify:
```
Tax Unit Gini Coefficient: 0.521 ± 0.005
Top 10% Income Share: 47.2% ± 0.5%
Top 1% Income Share: 19.8% ± 0.3%
Poverty Rate: 11.6% ± 0.2%
```

### Validation Dashboard
```bash
# Start local validation dashboard
make dashboard

# Access at http://localhost:8080
# Compare your results against paper figures
```

## Paper Reproduction

### Generate Paper
```bash
# Build paper with updated results
make paper

# Output: paper/main.pdf
# Build log: paper/main.log
```

### Update Results
```python
# To regenerate tables with your data
python paper/scripts/generate_all_tables.py

# To update specific figures
python paper/scripts/calculate_distributional_metrics.py
```

## Continuous Validation

### Automated Tests
```bash
# Run test suite
pytest tests/

# Run specific test
pytest tests/test_imputation.py::test_qrf_preserves_distribution

# Check reproducibility
pytest tests/test_reproducibility.py -v
```

### Expected Test Output
```
tests/test_reproducibility.py::test_deterministic_imputation PASSED
tests/test_reproducibility.py::test_weight_optimization_convergence PASSED
tests/test_reproducibility.py::test_validation_metrics_stable PASSED
```

## Docker Option

For guaranteed reproducibility:
```bash
# Build container
docker build -t enhanced-cps .

# Run with data mounted
docker run -v $(pwd)/data:/app/data enhanced-cps make data

# Interactive session
docker run -it -v $(pwd):/app enhanced-cps bash
```

## Checksums

Verify output integrity:
```bash
# Generate checksums
make checksums

# Verify against reference
diff checksums.txt reference_checksums.txt
```

Expected checksums (SHA-256):
```
enhanced_cps_2022.h5: a3f8b2c1d4e5f6789abcdef0123456789...
validation_results.json: b2c3d4e5f67890abcdef1234567890ab...
```

## Performance Notes

### Resource Usage by Stage
1. **CPS Loading**: 5 min, 4GB RAM
2. **PUF Processing**: 10 min, 8GB RAM  
3. **Imputation**: 60-90 min, 16GB RAM
4. **Reweighting**: 120-180 min, 32GB RAM
5. **Validation**: 30 min, 8GB RAM

### Optimization Tips
- Use `--njobs=4` for parallel processing
- Set `--sample_frac=0.1` for testing
- Enable `--cache` to avoid recomputation

## Support

- Issues: https://github.com/policyengine/policyengine-us-data/issues
- Documentation: https://policyengine.github.io/policyengine-us-data/
- Discussion: https://github.com/policyengine/policyengine-us-data/discussions

## Citation

If you use this dataset, please cite:
```bibtex
@article{enhanced_cps_2024,
  title={Enhanced CPS: Combining Survey and Administrative Data},
  author={Author Names},
  journal={International Journal of Microsimulation},
  year={2024}
}
```