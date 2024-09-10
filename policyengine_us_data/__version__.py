import importlib.metadata
import tomli

try:
    # We're doing this instead of using setuptools_scm because
    # setuptools_scm makes it very difficult to not append ".devX"
    # to the version number, and because importlib.metadata.version
    # occurs at runtime, it does not read pyproject.toml on call, returning
    # older version number
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    __version__ = pyproject["project"]["version"]
except Exception as e:
    __version__ = importlib.metadata.version("policyengine_us_data")
