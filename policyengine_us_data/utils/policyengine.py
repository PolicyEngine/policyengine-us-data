from __future__ import annotations

import json
import subprocess
import tomllib
import hashlib
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from typing import Any
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
UV_LOCK_PATH = REPO_ROOT / "uv.lock"


@dataclass(frozen=True)
class PolicyEngineUSBuildInfo:
    version: str
    locked_version: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    package_file_sha256: str | None = None
    package_tree_sha256: str | None = None
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {"version": self.version}
        if self.locked_version is not None:
            result["locked_version"] = self.locked_version
        if self.git_commit is not None:
            result["git_commit"] = self.git_commit
            result["commit_id"] = self.git_commit
            result["direct_url"] = {
                "vcs_info": {
                    "commit_id": self.git_commit,
                    "vcs": "git",
                }
            }
        if self.git_dirty is not None:
            result["git_dirty"] = self.git_dirty
        if self.package_file_sha256 is not None:
            result["package_file_sha256"] = self.package_file_sha256
        if self.package_tree_sha256 is not None:
            result["package_tree_sha256"] = self.package_tree_sha256
        if self.source_path is not None:
            result["source_path"] = self.source_path
        return result

    def to_metadata_dict(self) -> dict[str, Any]:
        result = self.to_dict()
        result.pop("source_path", None)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyEngineUSBuildInfo":
        direct_url = data.get("direct_url") or {}
        vcs_info = direct_url.get("vcs_info") or {}
        return cls(
            version=data["version"],
            locked_version=data.get("locked_version"),
            git_commit=(
                data.get("git_commit")
                or data.get("commit_id")
                or vcs_info.get("commit_id")
            ),
            git_dirty=data.get("git_dirty"),
            package_file_sha256=data.get("package_file_sha256"),
            package_tree_sha256=data.get("package_tree_sha256"),
            source_path=data.get("source_path"),
        )


def _find_git_root(start_path: Path | None) -> Path | None:
    current = start_path
    while current is not None:
        if (current / ".git").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent
    return None


def _get_git_commit(path: Path | None) -> str | None:
    if path is None:
        return None
    git_root = _find_git_root(path)
    if git_root is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(git_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_git_dirty(path: Path | None) -> bool | None:
    if path is None:
        return None
    git_root = _find_git_root(path)
    if git_root is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(git_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(completed.stdout.strip())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        if "__pycache__" in file_path.parts or file_path.suffix in {".pyc", ".pyo"}:
            continue
        relative_path = file_path.relative_to(path).as_posix()
        contents = file_path.read_bytes()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(contents)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    return digest.hexdigest()


def _policyengine_us_package_file() -> Path | None:
    spec = importlib.util.find_spec("policyengine_us")
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin)
    return path if path.exists() else None


@lru_cache(maxsize=None)
def get_locked_dependency_version(package_name: str) -> str | None:
    if not UV_LOCK_PATH.exists():
        return None
    lock_data = tomllib.loads(UV_LOCK_PATH.read_text())
    for package in lock_data.get("package", []):
        if package.get("name") == package_name:
            return package.get("version")
    return None


@lru_cache(maxsize=1)
def get_policyengine_us_build_info() -> PolicyEngineUSBuildInfo:
    version = metadata.version("policyengine-us")
    distribution = metadata.distribution("policyengine-us")

    source_path = None
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        source_path = direct_url.get("url")
        if source_path and source_path.startswith("file://"):
            source_path = source_path.removeprefix("file://")
    if source_path is None:
        try:
            import policyengine_us

            source_path = str(Path(policyengine_us.__file__).resolve().parent)
        except Exception:
            source_path = None

    source = Path(source_path) if source_path else None
    git_commit = _get_git_commit(source)
    git_dirty = _get_git_dirty(source)
    package_file = _policyengine_us_package_file()
    return PolicyEngineUSBuildInfo(
        version=version,
        locked_version=get_locked_dependency_version("policyengine-us"),
        git_commit=git_commit,
        git_dirty=git_dirty,
        package_file_sha256=(
            _sha256_file(package_file) if package_file is not None else None
        ),
        package_tree_sha256=(
            _sha256_directory(package_file.parent) if package_file is not None else None
        ),
        source_path=source_path,
    )


def assert_locked_policyengine_us_version() -> PolicyEngineUSBuildInfo:
    build_info = get_policyengine_us_build_info()
    if (
        build_info.locked_version is not None
        and build_info.version != build_info.locked_version
    ):
        raise RuntimeError(
            "Installed policyengine-us version does not match uv.lock: "
            f"found {build_info.version}, expected {build_info.locked_version}."
        )
    return build_info


@lru_cache(maxsize=1)
def _policyengine_us_variable_names() -> frozenset[str]:
    from policyengine_us import CountryTaxBenefitSystem

    return frozenset(CountryTaxBenefitSystem().variables)


def has_policyengine_us_variables(*variables: str) -> bool:
    try:
        available_variables = _policyengine_us_variable_names()
    except Exception:
        return False

    return set(variables).issubset(available_variables)
