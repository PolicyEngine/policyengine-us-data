# Introduction

Microsimulation models require high-quality microdata that accurately represents both demographic characteristics and economic outcomes. The ideal dataset would combine the demographic richness and household structure of surveys with the income precision of administrative tax records. However, publicly available datasets typically excel in one dimension while lacking in the other.

The Current Population Survey (CPS) Annual Social and Economic Supplement provides detailed household demographics, family relationships, and program participation data for a representative sample of US households. However, it suffers from well-documented income underreporting, particularly at the top of the distribution. The IRS Public Use File (PUF) contains accurate tax return information but lacks household structure, demographic detail, and state identifiers needed for comprehensive policy analysis.

This paper presents a methodology for creating an Enhanced CPS dataset that combines the strengths of both sources. Through a two-stage enhancement process—imputation followed by reweighting—we create a dataset suitable for analyzing both tax and transfer policies at federal and state levels.

## Related Work

Several approaches have been developed to address the limitations of survey data for microsimulation:

Statistical matching techniques have long been used to combine datasets. Early work by [citations needed] established methods for matching records based on common variables. More recent advances use machine learning approaches for imputation.

Reweighting methods to align survey data with administrative totals have been employed by statistical agencies and researchers. The Luxembourg Income Study uses calibration to improve cross-national comparability. The Urban-Brookings Tax Policy Center employs reweighting in their microsimulation model.

Our approach builds on these foundations while introducing several innovations:
- Use of quantile regression forests to preserve distributional characteristics
- Calibration to over 7,000 targets from multiple administrative sources  
- Open-source implementation enabling reproducibility and collaboration
- Integration with a comprehensive tax-benefit microsimulation model

## Contributions

This paper makes three main contributions:

1. **Methodological**: We demonstrate how quantile regression forests can effectively impute detailed tax variables while preserving their joint distribution and relationship to demographics.

2. **Empirical**: We create and validate a publicly available enhanced dataset that outperforms existing alternatives across multiple dimensions.

3. **Practical**: We provide open-source tools and documentation enabling researchers to apply these methods, modify the approach, or build upon our work.

The remainder of this paper is organized as follows. Section 2 describes our data sources. Section 3 details the enhancement methodology. Section 4 presents validation results. Section 5 discusses limitations and future directions. Section 6 concludes.