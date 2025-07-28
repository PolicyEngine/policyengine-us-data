# Response to Reviewers

We thank the reviewers for their careful reading and constructive feedback. We have substantially revised the manuscript to address their concerns.

## Reviewer 1 (Jon Bakija)

### Temporal Consistency Concern
**Comment:** "The use of 2015 PUF data to enhance 2024 CPS raises serious temporal consistency issues..."

**Response:** We acknowledge this limitation and have added discussion in Section 4.2 (Limitations). The 2015 PUF remains the most recent publicly available tax microdata. While we uprate dollar amounts using IRS SOI growth factors, demographic shifts are not fully captured. We note that our calibration to 7,000+ contemporary targets partially mitigates this issue by forcing consistency with current administrative totals.

### Poverty Rate Discrepancy
**Comment:** "The poverty metrics require careful examination and validation..."

**Response:** We have removed all specific poverty rate claims and replaced them with a more cautious discussion. The poverty metrics require further investigation, and we now advise users to interpret poverty results cautiously. We acknowledge that poverty measurement in enhanced datasets is complex due to the interaction between imputed tax variables and poverty thresholds. Future work will include specific poverty rate calibration targets.

## Reviewer 2 (Nora Lustig)

### Methodological Transparency
**Comment:** "The paper would benefit from more detailed discussion of the QRF hyperparameters..."

**Response:** We have added technical details about the QRF implementation in Section 3.1. The implementation uses the quantile-forest package with standard hyperparameters. Specific values are documented in the source code at `policyengine_us_data/datasets/cps/extended_cps.py`.

### State Tax Modeling
**Comment:** "Given the importance of state-level analysis, how does the methodology handle state tax modeling?"

**Response:** We have added Section 4.4 discussing state tax capabilities. The enhanced dataset preserves CPS state identifiers while incorporating federal tax precision from the PUF. We calibrate to state-level targets including demographics, income, and program participation. The dataset supports state tax calculators that typically use federal AGI as a starting point.

## Reviewer 3 (Gijs Dekkers)

### Validation and Robustness
**Comment:** "Additional validation exercises would strengthen confidence..."

**Response:** We have added discussion of validation approaches including cross-validation methodology and the importance of stability testing. We provide reference to our comprehensive online validation dashboard at https://policyengine.github.io/policyengine-us-data/validation.html where users can examine performance across all calibration targets.

### Reproducibility
**Comment:** "...encourage the authors to ensure full reproducibility..."

**Response:** We have implemented a complete reproducibility framework:
- All results now generated from Python scripts in `paper/scripts/`
- Added `make paper-results` target to generate all tables programmatically
- Fixed random seeds throughout the pipeline
- All code and data publicly available on GitHub

## Additional Changes

1. **Removed all adjectives and adverbs** from the text to maintain direct, evidence-based writing
2. **Created unified content system** allowing single-source documentation for both Jupyter Book and LaTeX paper
3. **Implemented strict data integrity protocols** to ensure all reported results come from actual computations
4. **Added placeholder markers** ([TO BE CALCULATED]) for metrics requiring full dataset generation

We believe these revisions substantially improve the manuscript's rigor, transparency, and reproducibility. We look forward to your further feedback.