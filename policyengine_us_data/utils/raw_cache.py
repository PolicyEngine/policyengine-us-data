import json
import logging
import os
from pathlib import Path

from policyengine_us_data.storage import STORAGE_FOLDER

logger = logging.getLogger(__name__)

RAW_INPUTS_DIR = STORAGE_FOLDER / "calibration" / "raw_inputs"
RAW_INPUTS_DIR.mkdir(parents=True, exist_ok=True)

REFRESH = os.environ.get("PE_REFRESH_RAW", "0") == "1"


def cache_path(filename: str) -> Path:
    """Return the absolute path of a raw-input cache file.

    ``filename`` must be a plain relative path under
    :data:`RAW_INPUTS_DIR` — no absolute paths and no parent-directory
    (``..``) components. Without this guard a caller that builds
    ``filename`` from a URL or any external input (for example
    ``cache_path(url.split("/")[-1])`` in a future ETL script) could
    escape ``RAW_INPUTS_DIR`` and either read or overwrite arbitrary
    files elsewhere on the filesystem.

    The resolve-and-check below fails closed if the resolved path is
    not inside ``RAW_INPUTS_DIR``.
    """
    if not isinstance(filename, (str, os.PathLike)):
        raise TypeError(
            f"cache_path expects a string or PathLike filename; got {type(filename)!r}"
        )
    if filename == "" or filename is None:
        raise ValueError("cache_path requires a non-empty filename")
    path_filename = Path(filename)
    if path_filename.is_absolute():
        raise ValueError(f"cache_path refuses absolute filenames; got {filename!r}")
    if ".." in path_filename.parts:
        raise ValueError(
            f"cache_path refuses filenames that traverse with '..'; got {filename!r}"
        )
    raw_inputs_dir_resolved = RAW_INPUTS_DIR.resolve()
    candidate = (RAW_INPUTS_DIR / path_filename).resolve()
    try:
        candidate.relative_to(raw_inputs_dir_resolved)
    except ValueError as exc:
        raise ValueError(
            f"cache_path refuses filenames that escape RAW_INPUTS_DIR "
            f"({raw_inputs_dir_resolved}): got {filename!r} -> {candidate}"
        ) from exc
    return RAW_INPUTS_DIR / path_filename


def is_cached(filename: str) -> bool:
    return cache_path(filename).exists() and not REFRESH


def save_json(filename: str, data):
    cache_path(filename).write_text(json.dumps(data, ensure_ascii=False))


def load_json(filename: str):
    return json.loads(cache_path(filename).read_text())


def save_bytes(filename: str, data: bytes):
    cache_path(filename).write_bytes(data)


def load_bytes(filename: str) -> bytes:
    return cache_path(filename).read_bytes()
