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

## SSA Data Sources

The following SSA data sources are used in this project:

- [Latest Trustee's Report (2025)](https://www.ssa.gov/oact/TR/2025/index.html) - Source for `social_security_aux.csv` (extracted via `extract_ssa_costs.py`)
- [Single Year Supplementary Tables (2025)](https://www.ssa.gov/oact/tr/2025/lrIndex.html) - Long-range demographic and economic projections
- [Single Year Age Demographic Projections (2024 - latest published)](https://www.ssa.gov/oact/HistEst/Population/2024/Population2024.html) - Source for `SSPopJul_TR2024.csv` population data

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
