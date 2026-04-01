"""Shared pre-baked Modal images for policyengine-us-data.

Bakes source code and dependencies into image layers at build time.
Modal caches layers by content hash of copied files -- if code
changes, the image rebuilds; if not, the cached layer is reused.

Uses `uv pip install --system` to install packages directly into
the system Python (no venv). This matches the policyengine-api-v2
simulation-api pattern: containers start with everything already
importable, no `uv run` wrapper needed.
"""

import subprocess
import modal
from pathlib import Path

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

_IGNORE = [
    ".git",
    "__pycache__",
    "*.egg-info",
    ".pytest_cache",
    "*.h5",
    "*.npy",
    "*.pkl",
    "*.db",
    "node_modules",
    "venv",
    ".venv",
    "docs/_build",
    "paper",
    "presentations",
]


def _base_image(extras: list[str] | None = None):
    extra_flags = " ".join(f"--extra {e}" for e in (extras or []))
    return (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("git")
        .pip_install("uv>=0.8")
        .add_local_dir(
            str(REPO_ROOT),
            remote_path="/root/policyengine-us-data",
            copy=True,
            ignore=_IGNORE,
        )
        .env(GIT_ENV)
        .run_commands(
            f"cd /root/policyengine-us-data && "
            f"UV_HTTP_TIMEOUT=300 uv pip install --system -e '.[dev]' {extra_flags}"
        )
    )


cpu_image = _base_image()
gpu_image = _base_image(extras=["l0"])
