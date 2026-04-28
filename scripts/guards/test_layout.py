"""Guardrails for pytest layout and test helper imports."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_LANES = {
    "tests/unit": Path("tests/unit"),
    "tests/integration": Path("tests/integration"),
    "tests/optimized": Path("tests/optimized"),
}
ALLOWED_TEST_ROOTS = tuple(TEST_LANES.values())
PYTEST_FILE_PREFIX = "test_"
PYTEST_FILE_SUFFIX = "_test.py"


def _git_files() -> list[Path]:
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [
            path.relative_to(REPO_ROOT)
            for path in REPO_ROOT.rglob("*")
            if path.is_file()
        ]

    return [
        Path(line)
        for line in result.stdout.splitlines()
        if line and (REPO_ROOT / line).is_file()
    ]


def _is_pytest_file(path: Path) -> bool:
    return path.suffix == ".py" and (
        path.name.startswith(PYTEST_FILE_PREFIX)
        or path.name.endswith(PYTEST_FILE_SUFFIX)
    )


def _is_under(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _test_lane(path: Path) -> str | None:
    for name, root in TEST_LANES.items():
        if _is_under(path, root):
            return name
    return None


def _module_root(module: str) -> str | None:
    for name in TEST_LANES:
        if module == name.replace("/", ".") or module.startswith(
            f"{name.replace('/', '.')}."
        ):
            return name
    return None


def _check_test_placement(files: list[Path]) -> list[str]:
    violations = []
    for path in files:
        if not _is_pytest_file(path):
            continue

        if _is_under(path, Path("policyengine_us_data/tests")):
            violations.append(
                f"{path}: package-internal tests are not collected by CI; "
                "move tests under tests/unit, tests/integration, or tests/optimized."
            )
            continue

        if path.parts and path.parts[0] == "tests":
            if not any(_is_under(path, root) for root in ALLOWED_TEST_ROOTS):
                violations.append(
                    f"{path}: pytest files under tests/ must live under "
                    "tests/unit, tests/integration, or tests/optimized."
                )

    return violations


def _check_test_imports(files: list[Path]) -> list[str]:
    violations = []
    for path in files:
        if path.suffix != ".py" or not _is_under(path, Path("tests")):
            continue

        source = (REPO_ROOT / path).read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{path}: could not parse Python source: {exc}")
            continue

        current_lane = _test_lane(path)
        for node in ast.walk(tree):
            module_names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                module_names.append(node.module)
            elif isinstance(node, ast.Import):
                module_names.extend(alias.name for alias in node.names)

            for module in module_names:
                if module == "tests.conftest" or module.startswith("tests.conftest."):
                    violations.append(
                        f"{path}: import from {module!r} couples tests to global "
                        "pytest setup; move helpers into a local support module."
                    )
                    continue

                imported_lane = _module_root(module)
                if imported_lane and imported_lane != current_lane:
                    violations.append(
                        f"{path}: imports {module!r} across test lanes; move shared "
                        "helpers to tests/support or colocate them with the tests."
                    )

    return violations


def check() -> list[str]:
    files = _git_files()
    return [
        *_check_test_placement(files),
        *_check_test_imports(files),
    ]


def main() -> int:
    violations = check()
    if not violations:
        print("test-layout guard passed")
        return 0

    print("test-layout guard failed:")
    for violation in violations:
        print(f"  - {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
