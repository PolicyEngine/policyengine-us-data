# Survey of Consumer Finances (SCF)

This folder contains the tooling that ingests the Federal Reserve Board's
Survey of Consumer Finances (SCF) summary extract into PolicyEngine's US
microdata pipeline (`fed_scf.py`, `scf.py`).

The SCF is the Fed's triennial household-level survey of wealth, debt,
and income. PolicyEngine uses the summary extract to inform net-worth
and asset-related calibration targets.

## Documentation

The Federal Reserve Board publishes a codebook for each SCF survey wave
describing every summary variable, derivation, and weight. These are the
canonical reference for the code in this folder:

- [2022 SCF main-survey codebook (TXT)](https://www.federalreserve.gov/econres/files/codebk2022.txt)
- [2019 SCF main-survey codebook (TXT)](https://www.federalreserve.gov/econres/files/codebk2019.txt)
- [2016 SCF main-survey codebook (TXT)](https://www.federalreserve.gov/econres/files/codebk2016.txt)
- [SCF summary-extract variable-definition macro (bulletin.macro.txt)](https://www.federalreserve.gov/econres/files/bulletin.macro.txt)

See also:

- [SCF landing page](https://www.federalreserve.gov/econres/scfindex.htm)
- [SCF documentation (working papers, methodology)](https://www.federalreserve.gov/econres/scf-documentation.htm)

## Data products in this folder

- `fed_scf.py` — downloads the Fed's SAS summary-extract ZIPs
  (`SummarizedFedSCF_2016`, `SummarizedFedSCF_2019`, `SummarizedFedSCF_2022`)
  and reads them into a pandas DataFrame.
- `scf.py` — wraps the raw summary extract in a PolicyEngine `Dataset`
  (`SCF`) with the standard ARRAYS format used downstream.
