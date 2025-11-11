# Long-Term Income Tax Revenue Projection Methodology
## Integrating Economic Uprating with Demographic Reweighting

### Quick Start

Run projections using `run_household_projection.py`:

```bash
# Save calibrated datasets as .h5 files for each year
python run_household_projection.py 2100 --greg --use-ss --save-h5
```

**Arguments:**
- `END_YEAR`: Target year for projection (default: 2035)
- `--greg`: Use GREG calibration instead of IPF (optional)
- `--use-ss`: Include Social Security benefit totals as calibration target (requires --greg)
- `--save-h5`: Save year-specific .h5 files to `./projected_datasets/` directory

---

### Documentation

For detailed methodology, implementation, and analysis, see the [Long-Term Projections Notebook](../../../../docs/long_term_projections.ipynb).
