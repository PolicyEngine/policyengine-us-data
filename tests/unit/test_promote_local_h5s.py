import pytest

import policyengine_us_data.calibration.promote_local_h5s as promote_module


def test_promote_preflight_failure_stops_before_production_writes(
    tmp_path, monkeypatch
):
    local_file = tmp_path / "AL.h5"
    local_file.write_bytes(b"state")
    files = [(local_file, "states/AL.h5")]
    rel_paths = ["states/AL.h5"]
    promote_calls = []

    monkeypatch.setattr(
        promote_module,
        "preflight_release_manifest_publish",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("blocked")),
    )
    monkeypatch.setattr(
        promote_module,
        "promote_staging_to_production_hf",
        lambda *args, **kwargs: promote_calls.append(("hf", args, kwargs)),
    )
    monkeypatch.setattr(
        promote_module,
        "upload_from_hf_staging_to_gcs",
        lambda *args, **kwargs: promote_calls.append(("gcs", args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="blocked"):
        promote_module.promote(files, rel_paths, version="1.73.0")

    assert promote_calls == []
