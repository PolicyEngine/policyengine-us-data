"""Shared pre-baked Modal images for policyengine-us-data.

Bakes source code and dependencies into image layers at build time.
Modal caches layers by content hash of copied files -- if code
changes, the image rebuilds; if not, the cached layer is reused.
"""

import subprocess
from pathlib import Path
from typing import Callable

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

GIT_ENV = {}
try:
    GIT_ENV["GIT_COMMIT"] = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        .decode()
        .strip()
    )
    GIT_ENV["GIT_BRANCH"] = (
        subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        .decode()
        .strip()
    )
    GIT_ENV["BUILD_COMMIT_SHA"] = GIT_ENV["GIT_COMMIT"]
except Exception:
    pass


# Extra paths the Modal image must never include, beyond what .gitignore
# already covers. `.git` holds hundreds of MB of pack data that Modal never
# reads; `paper` and `presentations` are authoring directories.
_MODAL_EXTRA_IGNORE = {".git", "paper", "presentations"}


def _build_ignore_callable(repo_root: Path) -> Callable[[Path], bool]:
    """Return an ignore predicate that mirrors .gitignore for Modal.

    Modal's `add_local_dir(ignore=...)` uses dockerignore semantics, where
    bare patterns like `*.h5` only match root-level files. Our `.gitignore`
    mixes gitignore semantics (bare names match at any depth). Translating
    patterns by hand is error-prone and drifts over time. Instead, we ask
    git directly for the set of locally-ignored paths and exclude those.
    Untracked-but-not-ignored files still ship so uncommitted edits to
    Modal code (e.g. `modal_app/images.py` itself) make it into the image.
    """
    repo_root = repo_root.resolve()
    ignored_paths: set[Path] = set()
    try:
        out = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "--directory",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            entry = line.strip().rstrip("/")
            if entry:
                ignored_paths.add((repo_root / entry).resolve())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    for name in _MODAL_EXTRA_IGNORE:
        ignored_paths.add((repo_root / name).resolve())

    def should_ignore(path: Path) -> bool:
        try:
            resolved = path.resolve()
        except (OSError, ValueError):
            return False
        if resolved in ignored_paths:
            return True
        for parent in resolved.parents:
            if parent in ignored_paths:
                return True
            if parent == repo_root:
                return False
        return False

    return should_ignore


def _base_image(extras: list[str] | None = None):
    return (
        modal.Image.debian_slim(python_version="3.14")
        .apt_install("git", "make")
        .uv_sync(
            uv_project_dir=str(REPO_ROOT),
            frozen=True,
            extras=extras,
        )
        .add_local_dir(
            str(REPO_ROOT),
            remote_path="/root/policyengine-us-data",
            copy=True,
            ignore=_build_ignore_callable(REPO_ROOT),
        )
        .workdir("/root/policyengine-us-data")
        .env(GIT_ENV)
    )


cpu_image = _base_image()
gpu_image = _base_image(extras=["l0"])
