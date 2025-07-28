# Response to Reviewers

We thank the reviewers for their careful reading and constructive feedback. We have substantially revised the manuscript to address their concerns.

## Reviewer 1 (Jon Bakija)

### Temporal Consistency Concern
**Comment:** "The use of 2015 PUF data to enhance 2024 CPS raises serious temporal consistency issues..."

**Response:** We acknowledge this limitation and have added discussion in Section 4.2 (Limitations). The 2015 PUF remains the most recent publicly available tax microdata. While we uprate dollar amounts using IRS SOI growth factors, demographic shifts are not fully captured. We note that our calibration to 7,000+ contemporary targets partially mitigates this issue by forcing consistency with current administrative totals.

### Poverty Rate Discrepancy
**Comment:** "The reported SPM poverty rate of 24.9% seems implausibly high..."

**Response:** We have removed the specific poverty rate claims and replaced them with a more cautious discussion. The poverty metrics require further investigation, and we now advise users to interpret poverty results cautiously. We have removed the decomposition analysis that was not based on actual computed results.

## Reviewer 2 (Nora Lustig)

### Methodological Transparency
**Comment:** "The paper would benefit from more detailed discussion of the QRF hyperparameters..."

**Response:** We have added comprehensive technical details in Section 3.1.3:
- Number of trees: 100
- Maximum depth: None (grown to purity)
- Bootstrap: True with replacement
- Random state: 0 for reproducibility
- Training on 10,000 PUF record subsample

### State Tax Modeling
**Comment:** "Given the importance of state-level analysis, how does the methodology handle state tax modeling?"

**Response:** We have added Section 4.4 discussing state tax capabilities. The enhanced dataset preserves CPS state identifiers while incorporating federal tax precision from the PUF. We calibrate to state-level targets including demographics, income, and program participation. The dataset supports state tax calculators that typically use federal AGI as a starting point.

## Reviewer 3 (Gijs Dekkers)

### Validation and Robustness
**Comment:** "Additional validation exercises would strengthen confidence..."

**Response:** We have added:
- Cross-validation methodology (Section 3.2.6)
- Stability analysis across random seeds
- Sensitivity testing of hyperparameters
- Reference to our comprehensive online validation dashboard

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