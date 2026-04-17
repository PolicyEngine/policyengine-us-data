"""Regression tests for ``utils.raw_cache.cache_path`` path-traversal
guard (N12).

Previously::

    def cache_path(filename: str) -> Path:
        return RAW_INPUTS_DIR / filename

No validation that ``filename`` stays under ``RAW_INPUTS_DIR``. All
current callers pass literal filenames, so the real-world risk is
low, but a future ETL script building ``cache_path(url.split("/")[-1])``
or similar would escape silently. Fail closed.
"""

import pytest

from policyengine_us_data.utils import raw_cache


def test_cache_path_accepts_plain_basenames():
    result = raw_cache.cache_path("snap_state.csv")
    assert result.name == "snap_state.csv"
    assert result.parent == raw_cache.RAW_INPUTS_DIR


def test_cache_path_accepts_nested_relative_paths_under_raw_inputs_dir():
    result = raw_cache.cache_path("soi/raw/2024.csv")
    assert result.parent.parent == raw_cache.RAW_INPUTS_DIR / "soi"


def test_cache_path_rejects_parent_dot_dot_traversal():
    with pytest.raises(ValueError, match=r"\.\."):
        raw_cache.cache_path("../escape.csv")


def test_cache_path_rejects_nested_parent_dot_dot_traversal():
    with pytest.raises(ValueError):
        raw_cache.cache_path("subdir/../../escape.csv")


def test_cache_path_rejects_absolute_path():
    with pytest.raises(ValueError, match="absolute"):
        raw_cache.cache_path("/etc/passwd")


def test_cache_path_rejects_empty_filename():
    with pytest.raises(ValueError, match="non-empty"):
        raw_cache.cache_path("")


def test_cache_path_rejects_non_string_filenames():
    with pytest.raises(TypeError):
        raw_cache.cache_path(123)  # type: ignore[arg-type]


def test_cache_path_rejects_escape_via_symlink_style_resolve(tmp_path, monkeypatch):
    """An absolute ``file://``-like prefix (``/tmp/other/foo``) must
    fail both the absolute-path guard and the resolve() containment
    check."""
    with pytest.raises(ValueError):
        raw_cache.cache_path(str(tmp_path / "unreachable.csv"))


def test_is_cached_refuses_traversal_filenames():
    """The public wrappers inherit the guard via cache_path."""
    with pytest.raises(ValueError):
        raw_cache.is_cached("../escape.csv")


def test_save_and_load_json_round_trip_under_raw_inputs_dir():
    """Sanity: the guard does not break the happy path."""
    filename = "test_raw_cache_path_traversal_roundtrip.json"
    try:
        raw_cache.save_json(filename, {"hello": "world"})
        assert raw_cache.load_json(filename) == {"hello": "world"}
    finally:
        path = raw_cache.cache_path(filename)
        if path.exists():
            path.unlink()
