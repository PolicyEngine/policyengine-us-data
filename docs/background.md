# Background

## The Microsimulation Landscape

Tax and benefit microsimulation models play a role in policy analysis by projecting the distributional and revenue impacts of proposed reforms. Institutions maintaining these models include government agencies like the Congressional Budget Office (CBO), Joint Committee on Taxation (JCT), and Treasury's Office of Tax Analysis (OTA), as well as non-governmental organizations including the Urban-Brookings Tax Policy Center (TPC), Tax Foundation, Penn Wharton Budget Model (PWBM), Institute on Taxation and Economic Policy (ITEP), Yale Budget Lab, and the open-source Policy Simulation Library (PSL). Each model serves specific institutional needs but faces common data challenges.

The core challenges these models face stem from the tradeoff between data comprehensiveness and accessibility. Administrative tax data provides income reporting but lacks the household context needed to model benefit programs and analyze family-level impacts {cite}`sabelhaus2020`. Survey data captures household relationships and program participation but suffers from income underreporting that worsens at higher income levels {cite}`meyer2021`. The need to protect taxpayer privacy limits data availability, as administrative microdata cannot be publicly released.

## Data Enhancement Approaches

Different microsimulation models use various approaches to enhance their underlying data:

Government models (CBO, JCT, Treasury) have access to confidential administrative data but cannot share their enhanced microdata. Non-governmental models work with public data, leading to various enhancement strategies. Some use proprietary extracts of tax returns, while others rely on survey data with different enhancement methods.

Our enhanced dataset provides an open-source methodology with state identifiers and calibration to state-level targets. This enables analysis of federal-state tax interactions. The dataset can be used with PolicyEngine or other microsimulation models.

The open-source nature promotes methodological transparency. The modular design allows researchers to substitute alternative imputation or calibration methods while maintaining the overall framework. Regular updates as new CPS and administrative data become available ensure the dataset remains current.