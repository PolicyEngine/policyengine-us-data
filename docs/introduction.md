# Introduction

```{admonition} Key Points
:class: tip
- Combines CPS demographic richness with PUF tax precision
- Two-stage enhancement: imputation + reweighting
- Calibrated to 7,000+ administrative targets
- Fully open-source implementation
```

Microsimulation models require high-quality microdata that accurately represents both demographic characteristics and economic outcomes. The ideal dataset would combine the demographic richness and household structure of surveys with the income precision of administrative tax records. However, publicly available datasets typically excel in one dimension while lacking in the other.

The Current Population Survey (CPS) Annual Social and Economic Supplement provides detailed household demographics, family relationships, and program participation data for a representative sample of US households. However, it suffers from well-documented income underreporting, particularly at the top of the distribution. The IRS Public Use File (PUF) contains accurate tax return information but lacks household structure, demographic detail, and state identifiers needed for comprehensive policy analysis.

This paper presents a methodology for creating an Enhanced CPS dataset that combines the strengths of both sources. Through a two-stage enhancement process—imputation followed by reweighting—we create a dataset suitable for analyzing both tax and transfer policies at federal and state levels.

## Related Work

Several approaches have been developed to address the limitations of survey data for microsimulation. Statistical matching techniques have long been used to combine datasets. {cite:t}`radner1978` pioneered exact matching methods for combining survey and administrative data, while {cite:t}`rodgers1984` developed statistical matching based on common variables. More recently, {cite:t}`dorazio2006` provided a comprehensive framework for modern statistical matching methods.

Economic studies have addressed dataset limitations through various strategies. The Congressional Budget Office combines CPS data with tax return information through statistical matching {cite}`cbo2022`. The Tax Policy Center creates synthetic datasets by statistically matching the CPS to a subset of tax returns {cite}`rohaly2005`. However, these approaches often sacrifice either demographic detail or tax precision, limiting their utility for comprehensive policy analysis.

Reweighting methods to align survey data with administrative totals have been employed by statistical agencies and researchers. The Luxembourg Income Study uses calibration to improve cross-national comparability {cite}`gornick2013`. The Urban-Brookings Tax Policy Center employs reweighting in their microsimulation model but relies on proprietary data that cannot be shared publicly {cite}`khitatrakun2016`.

Our approach differs from previous efforts in three key ways. First, we employ quantile regression forests to preserve distributional characteristics during imputation, improving upon traditional hot-deck and regression-based methods that may distort variable relationships. We conduct robustness checks comparing QRF performance to gradient boosting and neural network approaches, finding QRF provides the best balance of accuracy and interpretability. Second, we calibrate to over 7,000 targets from multiple administrative sources, far exceeding the scope of previous calibration efforts which typically use fewer than 100 targets. Third, we provide a fully open-source implementation enabling reproducibility and collaborative improvement, addressing the transparency limitations of existing proprietary models.

## Contributions

This paper makes three main contributions to the economic and public policy literature. Methodologically, we demonstrate how quantile regression forests can effectively impute detailed tax variables while preserving their joint distribution and relationship to demographics. This advances the statistical matching literature by showing how modern machine learning methods can overcome limitations of traditional hot-deck and parametric approaches. The preservation of distributional characteristics is particularly important for tax policy analysis where outcomes often depend on complex interactions between income sources and household characteristics.

Our empirical contribution involves creating and validating a publicly available enhanced dataset that addresses longstanding data limitations in microsimulation modeling. By combining the demographic richness of the CPS with the tax precision of the PUF, we enable analyses that were previously infeasible with public data. The dataset's calibration to over 7,000 administrative targets ensures consistency with official statistics across multiple dimensions simultaneously.

From a practical perspective, we provide open-source tools and comprehensive documentation that enable researchers to apply these methods, modify the approach, or build upon our work. This transparency contrasts with existing proprietary models and supports reproducible research. Government agencies could use our framework to enhance their own microsimulation capabilities, while academic researchers gain access to data suitable for analyzing distributional impacts of tax and transfer policies. The modular design allows incremental improvements as new data sources become available.

The remainder of this paper is organized as follows. Section 2 describes our data sources including the primary datasets and calibration targets. Section 3 details the enhancement methodology including both the imputation and reweighting stages. Section 4 presents validation results comparing performance across datasets. Section 5 discusses limitations, applications, and future directions. Section 6 concludes with implications for policy analysis.