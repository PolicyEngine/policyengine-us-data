"""Guardrails for pytest layout and test helper imports."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_LANES = {
    "tests/unit": Path("tests/unit"),
    "tests/integration": Path("tests/integration"),
}
VALIDATION_ROOT = Path("validation")
VALIDATION_STAGE_PATTERN = re.compile(r"^stage_[1-9]\d*$")
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


def _validation_stage_root(path: Path) -> Path | None:
    if len(path.parts) < 2 or path.parts[0] != VALIDATION_ROOT.name:
        return None

    stage = path.parts[1]
    if not VALIDATION_STAGE_PATTERN.fullmatch(stage):
        return None

    return VALIDATION_ROOT / stage


def _allowed_test_root(path: Path) -> Path | None:
    for root in TEST_LANES.values():
        if _is_under(path, root):
            return root

    return _validation_stage_root(path)


def _test_lane(path: Path) -> str | None:
    for name, root in TEST_LANES.items():
        if _is_under(path, root):
            return name

    validation_root = _validation_stage_root(path)
    if validation_root is not None:
        return validation_root.as_posix()

    return None


def _module_root(module: str) -> str | None:
    for name in TEST_LANES:
        if module == name.replace("/", ".") or module.startswith(
            f"{name.replace('/', '.')}."
        ):
            return name

    if module == "validation" or module.startswith("validation."):
        parts = module.split(".")
        if len(parts) >= 2 and VALIDATION_STAGE_PATTERN.fullmatch(parts[1]):
            return f"validation/{parts[1]}"

    return None


def _check_test_placement(files: list[Path]) -> list[str]:
    violations = []
    for path in files:
        if not _is_pytest_file(path):
            continue

        if _is_under(path, Path("policyengine_us_data/tests")):
            violations.append(
                f"{path}: package-internal tests are not collected by CI; "
                "move tests under tests/unit, tests/integration, or validation."
            )
            continue

        if path.parts and path.parts[0] in {"tests", "validation"}:
            if _allowed_test_root(path) is None:
                violations.append(
                    f"{path}: pytest files under tests/ or validation/ must live "
                    "under tests/unit, tests/integration, or a stage-specific "
                    "validation/stage_<n>/ folder."
                )

    return violations


def _check_test_imports(files: list[Path]) -> list[str]:
    violations = []
    for path in files:
        if path.suffix != ".py" or _allowed_test_root(path) is None:
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
