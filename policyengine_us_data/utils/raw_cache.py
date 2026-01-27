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
    return RAW_INPUTS_DIR / filename


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
