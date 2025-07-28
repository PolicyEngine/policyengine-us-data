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
