import importlib.metadata
from pathlib import Path

import tomllib

_PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"

try:
    with _PYPROJECT_PATH.open("rb") as f:
        pyproject = tomllib.load(f)
    __version__ = pyproject["project"]["version"]
except Exception:
    __version__ = importlib.metadata.version("policyengine_us_data")
