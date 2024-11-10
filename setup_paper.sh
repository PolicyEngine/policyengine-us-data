#!/bin/bash

# Create paper directory structure
mkdir -p paper/{sections,figures,tables,bibliography}

# Create main.tex
cat >paper/main.tex <<'EOL'
\documentclass[12pt]{article}

\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{natbib}
\usepackage{hyperref}

\input{macros}

\title{Enhancing Survey Microdata with Administrative Records: \\ A Novel Approach to Microsimulation Dataset Construction}
\author{PolicyEngine Team}
\date{\today}

\begin{document}

\maketitle

\input{sections/abstract}
\input{sections/introduction}
\input{sections/background}
\input{sections/data}
\input{sections/methodology}
\input{sections/results}
\input{sections/discussion}
\input{sections/conclusion}

\bibliography{bibliography/references}
\bibliographystyle{plainnat}

\end{document}
EOL

# Create macros.tex
cat >paper/macros.tex <<'EOL'
% Custom commands and mathematics macros
\newcommand{\policyengine}{\textsc{PolicyEngine}}
\newcommand{\cps}{\textsc{CPS}}
\newcommand{\puf}{\textsc{PUF}}
EOL

# Create section files with proper capitalization
cat >paper/sections/abstract.tex <<'EOL'
\section*{Abstract}
EOL

cat >paper/sections/introduction.tex <<'EOL'
\section{Introduction}
EOL

cat >paper/sections/background.tex <<'EOL'
\section{Background}
EOL

cat >paper/sections/data.tex <<'EOL'
\section{Data}
EOL

cat >paper/sections/methodology.tex <<'EOL'
\section{Methodology}

Our approach combines survey and administrative data through a multi-step process of aging, imputation, and reweighting. The procedure preserves the detailed demographic information from survey data while incorporating the precision of administrative tax records.

\subsection{Dataset Aging and Uprating}

We first age both datasets to the target year through demographic aging and economic uprating. For each variable $y$ in the set of economic variables, we apply an uprating factor:

\begin{equation}
y_{t} = y_{t_0} \cdot \frac{f(t)}{f(t_0)}
\end{equation}

where $f(t)$ represents the appropriate uprating factor time series (e.g., wage growth, price indices) for time $t$. The uprating factors are derived from official projections and historical series.

\subsection{Demographics Integration}

A key challenge is preserving household structure while incorporating tax unit information. We train a quantile regression forest model on PUF records with observed demographics to impute missing values:

\begin{equation}
\hat{d}_i = QRF(x_i; \theta)
\end{equation}

where $d_i$ represents demographic variables (age, sex, relationship status) and $x_i$ represents observed tax characteristics. The model preserves distributional relationships while avoiding deterministic assignments.

\subsection{Income and Tax Variable Enhancement}

We enhance CPS income and tax variables using information from the PUF through a two-stage process:

1. Variable Mapping: We establish correspondence between CPS and PUF variables through a detailed crosswalk accounting for definitional differences.

2. Distribution Matching: For each mapped variable pair, we fit a quantile regression forest:

\begin{equation}
\hat{y}_{PUF} = QRF(X_{CPS}; \theta)
\end{equation}

where $X_{CPS}$ includes relevant predictors from the CPS like employment status, age, and existing income measures.

\subsection{Reweighting Procedure}

The final step adjusts household weights to match key population and economic targets while preserving micro-level relationships. We solve:

\begin{equation}
\min_w \sum_j \left(\frac{\sum_i w_i x_{ij} - t_j}{t_j}\right)^2
\end{equation}

subject to:
\begin{equation}
w_i \geq 0 \quad \forall i
\end{equation}

where:
\begin{itemize}
\item $w_i$ is the weight for household $i$
\item $x_{ij}$ is the value of variable $j$ for household $i$
\item $t_j$ is the target total for variable $j$
\end{itemize}
EOL

cat >paper/sections/results.tex <<'EOL'
\section{Results}
EOL

cat >paper/sections/discussion.tex <<'EOL'
\section{Discussion}
EOL

cat >paper/sections/conclusion.tex <<'EOL'
\section{Conclusion}
EOL

# Create empty bibliography file
touch paper/bibliography/references.bib

# Create .gitignore for LaTeX
cat >paper/.gitignore <<'EOL'
## Core latex/pdflatex auxiliary files:
*.aux
*.lof
*.log
*.lot
*.fls
*.out
*.toc
*.fmt
*.fot
*.cb
*.cb2
.*.lb

## Generated if empty string is given at "Please type another file name for output:"
.pdf

## Bibliography auxiliary files (bibtex/biblatex/biber):
*.bbl
*.bcf
*.blg
*-blx.aux
*-blx.bib
*.run.xml

## Build tool auxiliary files:
*.fdb_latexmk
*.synctex
*.synctex(busy)
*.synctex.gz
*.synctex.gz(busy)
*.pdfsync
EOL

echo "Paper directory structure created successfully!"
