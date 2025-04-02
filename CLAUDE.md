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