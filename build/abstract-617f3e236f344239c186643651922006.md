# Abstract

We present a methodology for creating enhanced microsimulation datasets by combining the
Current Population Survey (CPS) with the IRS Public Use File (PUF). Our approach uses
quantile regression forests to impute 67 tax variables from the PUF onto CPS records,
preserving distributional characteristics while maintaining household composition and member
relationships. The imputation process alone does not guarantee consistency with official
statistics, necessitating a reweighting step to align the combined dataset with known
population totals and administrative benchmarks. We apply a reweighting algorithm that calibrates the dataset to 2,813 targets from the IRS Statistics of Income, Census population projections, Congressional Budget Office benefit program estimates, Treasury expenditure data, Joint Committee on Taxation tax expenditure estimates, healthcare spending patterns, and other benefit program costs. The reweighting employs dropout-regularized gradient descent optimization to ensure consistency with administrative benchmarks. The dataset maintains the CPS's demographic detail and geographic granularity while
incorporating tax reporting data from administrative sources. We release the enhanced
dataset, source code, and documentation to support policy analysis.
