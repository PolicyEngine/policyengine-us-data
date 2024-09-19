# Note: Action must be run in Python 3.11 or later
import tomllib


def fetch_version():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    return pyproject["project"]["version"]


if __name__ == "__main__":
    print(fetch_version())
