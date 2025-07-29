# Background

## The Microsimulation Landscape

Tax and benefit microsimulation models play a crucial role in policy analysis by projecting the distributional and revenue impacts of proposed reforms. Major institutions maintaining these models include government agencies like the Congressional Budget Office (CBO), Joint Committee on Taxation (JCT), and Treasury's Office of Tax Analysis (OTA), as well as non-governmental organizations including the Urban-Brookings Tax Policy Center (TPC), Tax Foundation, Penn Wharton Budget Model (PWBM), Institute on Taxation and Economic Policy (ITEP), Yale Budget Lab, and the open-source Policy Simulation Library (PSL). Each model serves specific institutional needs but faces common data challenges.

The core challenges these models face stem from the fundamental tradeoff between data comprehensiveness and accessibility. Administrative tax data provides accurate income reporting but lacks the household context needed to model benefit programs and analyze family-level impacts {cite}`sabelhaus2020`. Survey data captures household relationships and program participation but suffers from income underreporting that worsens at higher income levels {cite}`meyer2021`. The need to protect taxpayer privacy further limits data availability, as administrative microdata cannot be publicly released.

## Comparison of Major Microsimulation Models

Table 1 summarizes the data preparation approaches used by major US microsimulation models. Each model balances different priorities regarding data sources, enhancement methods, and transparency. Sources for this comparison include {cite}`cbo2018`, {cite}`jct2023`, {cite}`ota2012`, {cite}`tpc2024`, {cite}`tf2024`, {cite}`pwbm2024`, {cite}`policy2024`, {cite}`budgetlab2024`, and {cite}`psl2024`.

| Model | Primary Data | Enhancement Method | Geographic Detail | Transfer Programs | Public Access |
|-------|--------------|-------------------|-------------------|-------------------|---------------|
| CBO | CPS + tax data | Statistical matching | National | Yes | No |
| JCT | Tax returns | Aging/extrapolation | National | Limited | No |
| Treasury OTA | Tax returns | Administrative linking | National | Limited | No |
| TPC | CPS + PUF extract | Statistical matching | State | Yes | Limited |
| Tax Foundation | CPS | Reweighting | State | No | Yes |
| PWBM | CPS + admin | Dynamic aging | National | Yes | Limited |
| ITEP | CPS | Imputation | State | Limited | No |
| Yale Budget Lab | CPS | Reweighting | National | Yes | Yes |
| PSL Tax-Calculator | CPS/PUF | User choice | National | No | Yes |
| PolicyEngine (this paper) | CPS + PUF | QRF imputation + reweighting | State | Yes | Yes |

Government models (CBO, JCT, Treasury) benefit from access to confidential administrative data but cannot share their enhanced microdata. Non-governmental models must work with public data, leading to various enhancement strategies. The Tax Policy Center uses a proprietary extract of tax returns, while others rely entirely on survey data with different enhancement methods.

Our enhanced dataset addresses these institutional needs in several ways. For government agencies like CBO and Treasury, our open-source methodology provides a transparent alternative that could supplement or validate their internal models. The dataset's state identifiers and calibration to state-level targets enables state revenue departments to analyze federal-state tax interactions. Academic researchers gain access to data suitable for analyzing both tax and transfer policies without restrictive data use agreements. Policy organizations can use the enhanced dataset with PolicyEngine or other microsimulation models to provide rapid analysis of reform proposals.

The integration of our dataset with the broader microsimulation ecosystem offers several advantages. The open-source nature promotes methodological transparency and peer review, addressing concerns about "black box" models in policy analysis. The modular design allows researchers to substitute alternative imputation or calibration methods while maintaining the overall framework. Regular updates as new CPS and administrative data become available ensure the dataset remains current. Most importantly, the public availability democratizes access to high-quality microsimulation capabilities previously restricted to well-resourced institutions.