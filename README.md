# PolicyEngine US Data

## Installation

While it is possible to install via PyPi:
```bash
pip install policyengine-us-data
```
the recommended installation is 
```
pip install -e .[dev]
```
which installs the development dependencies in a reference-only manner (so that changes
to the package code will be reflected immediately); `policyengine-us-data` is a dev package
and not intended for direct access.

## Pull Requests

PRs must come from branches pushed to `PolicyEngine/policyengine-us-data`, not from
personal forks. The PR workflow hard-fails fork-based PRs before the real test suite
runs because the required secrets are unavailable there.

Before opening a PR, push the current branch to the upstream repo:

```bash
make push-pr-branch
```

That target pushes the current branch to the `upstream` remote and sets tracking so
`gh pr create` opens the PR from `PolicyEngine/policyengine-us-data`.

## SSA Data Sources

The following SSA data sources are used in this project:

- [Latest Trustee's Report (2025)](https://www.ssa.gov/oact/TR/2025/index.html) - Source for `social_security_aux.csv` (extracted via `extract_ssa_costs.py`)
- [Single Year Supplementary Tables (2025)](https://www.ssa.gov/oact/tr/2025/lrIndex.html) - Long-range demographic and economic projections
- [Single Year Age Demographic Projections (2024 - latest published)](https://www.ssa.gov/oact/HistEst/Population/2024/Population2024.html) - Source for `SSPopJul_TR2024.csv` population data

## Pipeline Overview

PolicyEngine constructs its representative household datasets through a multi-step pipeline. Public survey data is merged, stratified, and cloned to geographic variants per household. Each clone is simulated through PolicyEngine US with stochastic take-up, then calibrated via L0-regularized optimization against administrative targets at the national, state, and congressional district levels, producing geographically representative datasets.

The Enhanced CPS (`make data-legacy`) produces a national-only calibrated dataset. For the current geography-specific pipeline, see [docs/calibration.md](docs/calibration.md).

The repo currently contains two calibration tracks:
- Legacy Enhanced CPS (`make data-legacy`), which uses the older `EnhancedCPS` / `build_loss_matrix()` path for national-only calibration.
- Unified calibration (`docs/calibration.md`), which uses `storage/calibration/policy_data.db` and the sparse matrix + L0 pipeline for current national and geography-specific builds.

For detailed calibration usage, see [docs/calibration.md](docs/calibration.md) and [modal_app/README.md](modal_app/README.md).

### Running the Full Pipeline

The pipeline runs as sequential steps in Modal:

```bash
make pipeline   # prints the steps below

# 1. Build data (CPS/PUF/ACS → source-imputed stratified CPS)
make build-data-modal

# 2. Build calibration matrices (CPU, ~10h)
make build-matrices

# 3. Fit weights (GPU, county + national in parallel)
make calibrate-both

# 4. Build H5 files (state/district/city + national in parallel)
make stage-all-h5s

# 5. Promote to versioned HF paths
make promote
```

## Building the Paper

### Prerequisites

The paper requires a LaTeX distribution (e.g., TeXLive or MiKTeX) with the following packages:

- graphicx (for figures)
- amsmath (for mathematical notation)
- natbib (for bibliography management)
- hyperref (for PDF links)
- booktabs (for tables)
- geometry (for page layout)
- microtype (for typography)
- xcolor (for colored links)

On Ubuntu/Debian, you can install these with:

```bash
sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended
```

On macOS with Homebrew:

```bash
brew install --cask mactex
```

### Building

To build the paper:

```bash
make paper
```

To clean LaTeX build files:

```bash
make clean-paper
```

The output PDF will be at `paper/main.pdf`.

## Building the Documentation

### Prerequisites

The documentation uses Jupyter Book 2 (pre-release) with MyST. To install:

```bash
# Install Jupyter Book 2 pre-release
pip install --pre "jupyter-book==2.*"

# Install MyST CLI
npm install -g mystmd
```

### Building

To build and serve the documentation locally:

```bash
cd docs
myst start
```

Or alternatively from the project root:

```bash
jupyter book start docs
```

Both commands will start a local server at http://localhost:3001 where you can view the documentation.

The legacy Makefile command:

```bash
make documentation
```

Note: The Makefile uses the older `jb` command syntax which may not work with Jupyter Book 2. Use `myst start` or `jupyter book start docs` instead.
