# CLAUDE.md - Guidelines for PolicyEngine US Data

## Build Commands
- `make install` - Install dependencies and dev environment
- `make build` - Build the package using Python build
- `make data` - Generate project datasets

## Testing
- `pytest` - Run all tests
- `pytest path/to/test_file.py::test_function` - Run a specific test
- `make test` - Also runs all tests

## Formatting
- `make format` - Format all code using Black with 79 char line length
- `black . -l 79 --check` - Check formatting without changing files

## Code Style Guidelines
- **Imports**: Standard libraries first, then third-party, then internal
- **Type Hints**: Use for all function parameters and return values
- **Naming**: Classes: PascalCase, Functions/Variables: snake_case, Constants: UPPER_SNAKE_CASE
- **Documentation**: Google-style docstrings with Args and Returns sections
- **Error Handling**: Use validation checks with specific error messages
- **Line Length**: 79 characters max (Black configured in pyproject.toml)
- **Python Version**: Targeting Python 3.11

## CRITICAL RULES FOR ACADEMIC INTEGRITY

### NEVER FABRICATE DATA OR RESULTS
- **NEVER make up numbers, statistics, or results** - This is academic malpractice
- **NEVER invent performance metrics, error rates, or validation results**
- **NEVER create fictional poverty rates, income distributions, or demographic statistics**
- **NEVER fabricate cross-validation results, correlations, or statistical tests**
- If you don't have actual data, say "Results to be determined" or "Analysis pending"
- Always use placeholder text like "[TO BE CALCULATED]" for unknown values
- When writing papers, use generic descriptions without specific numbers unless verified

### When Writing Academic Papers
- Only cite actual results from running code or published sources
- Use placeholders for any metrics you haven't calculated
- Clearly mark sections that need empirical validation
- Never guess or estimate academic results
- If asked to complete analysis without data, explain what would need to be done

### Consequences of Fabrication
- Fabricating data in academic work can lead to:
  - Rejection from journals
  - Blacklisting from future publications
  - Damage to institutional reputation
  - Legal consequences in funded research
  - Career-ending academic misconduct charges