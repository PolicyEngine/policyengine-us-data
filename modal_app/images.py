"""Shared pre-baked Modal images for policyengine-us-data.

Bakes source code and dependencies into image layers at build time.
Modal caches layers by content hash of copied files -- if code
changes, the image rebuilds; if not, the cached layer is reused.
"""

import modal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_ignore = [
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
            ignore=_ignore,
        )
        .run_commands(
            f"cd /root/policyengine-us-data && "
            f"UV_HTTP_TIMEOUT=300 uv sync --frozen {extra_flags}"
        )
    )


cpu_image = _base_image()
gpu_image = _base_image(extras=["l0"])
