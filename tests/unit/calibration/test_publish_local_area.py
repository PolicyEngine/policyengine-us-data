import numpy as np
from types import SimpleNamespace

import policyengine_us_data.calibration.publish_local_area as publish_local_area
from policyengine_us_data.build_outputs.payload import H5Payload
from policyengine_us_data.calibration.publish_local_area import (
    build_h5,
    compute_input_fingerprint,
    load_calibration_geography,
)


def test_compute_input_fingerprint_uses_loader_canonical_geography_identity(
    tmp_path, monkeypatch
):
    weights_path = tmp_path / "weights.npy"
    dataset_path = tmp_path / "dataset.h5"
    geo_one = tmp_path / "geography-one.npz"
    geo_two = tmp_path / "geography-two.npz"

    np.save(weights_path, np.array([1.0, 2.0, 3.0, 4.0]))
    dataset_path.write_bytes(b"dataset")
    geo_one.write_bytes(b"first-raw-geometry")
    geo_two.write_bytes(b"second-raw-geometry")

    monkeypatch.setattr(
        "policyengine_us_data.calibration.publish_local_area.CalibrationGeographyLoader.resolve_source",
        lambda self, **kwargs: SimpleNamespace(
            kind="saved_geography",
            path=kwargs["geography_path"],
        ),
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.publish_local_area.CalibrationGeographyLoader.compute_canonical_checksum",
        lambda self, **kwargs: "sha256:canonical-geometry",
    )

    first = compute_input_fingerprint(
        weights_path,
        dataset_path,
        n_clones=2,
        geography_path=geo_one,
    )
    second = compute_input_fingerprint(
        weights_path,
        dataset_path,
        n_clones=2,
        geography_path=geo_two,
    )

    assert first == second


def test_compute_input_fingerprint_passes_calibration_package_path_to_loader(
    tmp_path, monkeypatch
):
    weights_path = tmp_path / "weights.npy"
    dataset_path = tmp_path / "dataset.h5"
    package_path = tmp_path / "calibration_package.pkl"

    np.save(weights_path, np.array([1.0, 2.0, 3.0, 4.0]))
    dataset_path.write_bytes(b"dataset")
    package_path.write_bytes(b"package")

    seen = {}

    def fake_resolve_source(self, **kwargs):
        seen["resolve"] = kwargs
        return SimpleNamespace(
            kind="calibration_package",
            path=kwargs["calibration_package_path"],
        )

    def fake_compute_canonical_checksum(self, **kwargs):
        seen["checksum"] = kwargs
        return "sha256:canonical-package"

    monkeypatch.setattr(
        "policyengine_us_data.calibration.publish_local_area.CalibrationGeographyLoader.resolve_source",
        fake_resolve_source,
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.publish_local_area.CalibrationGeographyLoader.compute_canonical_checksum",
        fake_compute_canonical_checksum,
    )

    compute_input_fingerprint(
        weights_path,
        dataset_path,
        n_clones=2,
        calibration_package_path=package_path,
    )

    assert seen["resolve"]["calibration_package_path"] == package_path
    assert seen["checksum"]["calibration_package_path"] == package_path


def test_load_calibration_geography_passes_calibration_package_path_to_loader(
    tmp_path, monkeypatch
):
    weights_path = tmp_path / "weights.npy"
    package_path = tmp_path / "calibration_package.pkl"

    np.save(weights_path, np.array([1.0, 2.0, 3.0, 4.0]))
    package_path.write_bytes(b"package")

    seen = {}

    def fake_resolve_source(self, **kwargs):
        seen["resolve"] = kwargs
        return None

    def fake_load(self, **kwargs):
        seen["load"] = kwargs
        return "geography"

    monkeypatch.setattr(
        "policyengine_us_data.calibration.publish_local_area.CalibrationGeographyLoader.resolve_source",
        fake_resolve_source,
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.publish_local_area.CalibrationGeographyLoader.load",
        fake_load,
    )

    result = load_calibration_geography(
        weights_path=weights_path,
        n_records=2,
        n_clones=2,
        calibration_package_path=package_path,
    )

    assert result == "geography"
    assert seen["resolve"]["calibration_package_path"] == package_path
    assert seen["load"]["calibration_package_path"] == package_path


def test_build_h5_facade_delegates_to_builder_and_writer(tmp_path, monkeypatch):
    output_path = tmp_path / "NC-01.h5"
    source = SimpleNamespace(n_households=2, time_period=2024)
    seen = {}

    class FakeBuilder:
        def __init__(self, **kwargs):
            seen["builder_init"] = kwargs

        def build(self, **kwargs):
            seen["build"] = kwargs
            payload = H5Payload(
                data={"household_id": {2024: np.array([0])}},
                time_period=2024,
                entity_lengths={"household": 1},
            )
            return FakeBuildResult(payload)

    class FakeBuildResult:
        def __init__(self, payload):
            self.payload = payload
            self.data = payload.data
            self.time_period = payload.time_period
            self.selection = SimpleNamespace(
                n_selected_clones=1,
                block_geoids=np.array(["block-1"]),
            )
            self.reindexed = SimpleNamespace(
                person_ids=np.array([0]),
                subentity_source_indices={"tax_unit": np.array([0])},
            )
            self.variables_saved = 1
            self.summary = {"total_weight": 2.0}

        def postprocessor_result(self, postprocessor):
            return SimpleNamespace(takeup_variables=("takes_up_snap",))

    class FakeWriter:
        def write(self, **kwargs):
            seen["write"] = kwargs
            return SimpleNamespace(
                households=1,
                persons=1,
                household_weight_sum=2.0,
                person_weight_sum=None,
            )

    monkeypatch.setattr(
        publish_local_area,
        "Microsimulation",
        lambda dataset: SimpleNamespace(dataset=dataset),
    )
    monkeypatch.setattr(
        publish_local_area.SourceDatasetSnapshot,
        "from_simulation",
        classmethod(lambda cls, dataset_path, simulation: source),
    )
    monkeypatch.setattr(
        publish_local_area,
        "LocalAreaDatasetBuilder",
        FakeBuilder,
    )
    monkeypatch.setattr(publish_local_area, "H5Writer", lambda: FakeWriter())

    result = build_h5(
        weights=np.array([1.0, 2.0, 3.0, 4.0]),
        geography=SimpleNamespace(),
        dataset_path=tmp_path / "source.h5",
        output_path=output_path,
        cd_subset=["3701"],
        county_fips_filter={"37183"},
        takeup_filter=["takes_up_snap"],
    )

    assert result == output_path
    assert [
        type(postprocessor).__name__
        for postprocessor in seen["builder_init"]["postprocessors"]
    ] == [
        "USEntityPostProcessor",
        "USGeographyPostProcessor",
        "USTakeupPostProcessor",
        "USMedicaidCostPostProcessor",
    ]
    assert seen["build"]["source"] is source
    assert seen["build"]["takeup_filter"] == ("takes_up_snap",)
    request = seen["build"]["request"]
    assert request.area_id == "NC-01"
    assert [area_filter.geography_field for area_filter in request.filters] == [
        "cd_geoid",
        "county_fips",
    ]
    assert seen["write"]["output_path"] == output_path
    assert seen["write"]["payload"].time_period == 2024
    np.testing.assert_array_equal(
        seen["write"]["payload"].data["household_id"][2024],
        np.array([0]),
    )
