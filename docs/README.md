# Documentation

This project uses [MyST Markdown](https://mystmd.org/) for documentation.

## Building Locally

### Requirements
- Python 3.13+ with dev dependencies: `uv pip install -e .[dev] --system`
- Node.js 20+ (required by MyST)

### Commands
```bash
make documentation        # Build static HTML files
make documentation-serve  # Serve locally on http://localhost:8080
```

## Important: MyST Build Outputs

**MyST creates two different outputs - DO NOT confuse them:**

- `_build/html/` - **Static HTML files (use for GitHub Pages deployment)**
- `_build/site/` - Dynamic content for `myst start` development server only

**GitHub Pages must deploy `_build/html/`**, not `_build/site/`. The `_build/site/` directory contains JSON files for MyST's development server and will result in a blank page on GitHub Pages.

## GitHub Pages Deployment

- Site URL: https://policyengine.github.io/policyengine-us-data/
- Deployed from: `docs/_build/html/` directory
- Propagation time: 5-10 minutes after push to gh-pages branch
- Workflow: `.github/workflows/code_changes.yaml` (on main branch only)

## Troubleshooting

**Blank page after deployment:**
- Check that workflow deploys `folder: docs/_build/html` (not `_build/site`)
- Wait 5-10 minutes for GitHub Pages propagation
- Hard refresh browser (Ctrl+Shift+R / Cmd+Shift+R)

**Build fails in CI:**
- Ensure Node.js setup step exists in workflow (MyST requires Node.js)
- Never add timeouts or `|| true` to build commands - they mask failures

**Missing index.html:**
- MyST auto-generates index.html in `_build/html/`
- Do not create manual index.html in docs/
