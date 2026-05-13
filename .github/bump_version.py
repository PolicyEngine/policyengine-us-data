"""Infer semver bump from towncrier fragment types and update version."""

import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?$")


def get_current_version(pyproject_path: Path) -> str:
    text = pyproject_path.read_text()
    match = VERSION_RE.search(text)
    if not match:
        print(
            "Could not find version in pyproject.toml",
            file=sys.stderr,
        )
        sys.exit(1)
    return match.group(1)


def get_package_name(pyproject_path: Path) -> str:
    text = pyproject_path.read_text()
    match = re.search(r'^name\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        print("Could not find project name in pyproject.toml", file=sys.stderr)
        sys.exit(1)
    return match.group(1)


def infer_bump(changelog_dir: Path) -> str:
    fragments = [
        f for f in changelog_dir.iterdir() if f.is_file() and f.name != ".gitkeep"
    ]
    if not fragments:
        print("No changelog fragments found", file=sys.stderr)
        sys.exit(1)

    categories = {f.suffix.lstrip(".") for f in fragments}
    for f in fragments:
        parts = f.stem.split(".")
        if len(parts) >= 2:
            categories.add(parts[-1])

    if "breaking" in categories:
        return "major"
    if "added" in categories or "removed" in categories:
        return "minor"
    return "patch"


def bump_version(version: str, bump: str) -> str:
    match = SEMVER_RE.match(version)
    if not match:
        print(f"Unsupported version format: {version}", file=sys.stderr)
        sys.exit(1)
    major, minor, patch = (int(x) for x in match.groups()[:3])
    if bump == "major":
        return f"{major + 1}.0.0"
    elif bump == "minor":
        return f"{major}.{minor + 1}.0"
    else:
        return f"{major}.{minor}.{patch + 1}"


def next_rc_version(package_name: str, final_version: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", package_name).lower()
    url = f"https://pypi.org/pypi/{normalized}/json"
    highest = 0
    try:
        with urlopen(url, timeout=20) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code != 404:
            raise
        payload = {"releases": {}}
    except URLError as exc:
        print(f"Could not fetch PyPI release history: {exc}", file=sys.stderr)
        sys.exit(1)
    prefix = re.escape(final_version)
    rc_re = re.compile(rf"^{prefix}rc(\d+)$")
    for version in payload.get("releases", {}):
        match = rc_re.match(version)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{final_version}rc{highest + 1}"


def update_file(path: Path, old_version: str, new_version: str):
    text = path.read_text()
    updated = text.replace(
        f'version = "{old_version}"',
        f'version = "{new_version}"',
    )
    if updated != text:
        path.write_text(updated)
        print(f"  Updated {path}")


def main():
    root = Path(__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"
    changelog_dir = root / "changelog.d"

    package_name = get_package_name(pyproject)
    current = get_current_version(pyproject)
    bump = infer_bump(changelog_dir)
    final_version = bump_version(current, bump)
    candidate_version = next_rc_version(package_name, final_version)

    print(f"Version: {current} -> {candidate_version} ({bump})")
    print(f"Final release version: {final_version}")

    update_file(pyproject, current, candidate_version)


if __name__ == "__main__":
    main()
