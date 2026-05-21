from pathlib import Path

from scripts.extract_pipeline_docs import (
    load_pipeline_map,
    merge_map,
    render_markdown,
    scan_decorated_objects,
)


def test_scan_decorated_objects_finds_pipeline_metadata():
    objects = scan_decorated_objects()

    assert "run_calibration" in objects
    assert objects["run_calibration"].object_path.endswith(
        "unified_calibration.run_calibration"
    )
    assert "def run_calibration(" in objects["run_calibration"].signature


def test_merge_map_preserves_api_nodes():
    objects = scan_decorated_objects()
    manifest = {
        "stages": [
            {
                "id": "example",
                "label": "Example",
                "title": "Example Stage",
                "description": "Example",
                "edges": [
                    {
                        "source": "run_calibration",
                        "target": "fit_model",
                        "edge_type": "data_flow",
                    }
                ],
            }
        ]
    }

    bundle = merge_map(manifest, objects)

    assert bundle["metadata"]["canonical_stage_count"] == 0
    assert bundle["metadata"]["stage_count"] == 1
    assert bundle["metadata"]["substage_count"] == 1
    assert {node["id"] for node in bundle["stages"][0]["nodes"]} == {
        "run_calibration",
        "fit_model",
    }
    assert bundle["metadata"]["api_node_count"] >= 1


def test_render_markdown_includes_stage_edges():
    objects = scan_decorated_objects()
    bundle = merge_map(
        {
            "stages": [
                {
                    "id": "example",
                    "label": "Example",
                    "title": "Example Stage",
                    "description": "Example",
                    "edges": [
                        {
                            "source": "run_calibration",
                            "target": "fit_model",
                            "edge_type": "data_flow",
                        }
                    ],
                }
            ]
        },
        objects,
    )

    markdown = render_markdown(bundle, objects)

    assert "# Pipeline Map" in markdown
    assert "`run_calibration` -> `fit_model`" in markdown


def test_pipeline_map_manifest_exists():
    assert Path("docs/pipeline_map.yaml").exists()


def test_pipeline_map_manifest_validates():
    objects = scan_decorated_objects()
    manifest = load_pipeline_map(Path("docs/pipeline_map.yaml"))

    bundle = merge_map(manifest, objects)

    assert bundle["metadata"]["canonical_stage_count"] == 5
    assert bundle["metadata"]["stage_count"] == 17
    assert bundle["metadata"]["substage_count"] == 17
    assert [stage["id"] for stage in bundle["canonical_stages"]] == [
        "1_build_datasets",
        "2_build_calibration_package",
        "3_fit_weights",
        "4_build_outputs",
        "5_validate_and_promote_release",
    ]
    assert [stage["id"] for stage in bundle["stages"]] == [
        "1a_raw_data_download",
        "1b_base_dataset_construction",
        "1c_extended_cps_puf_clone",
        "1d_enhanced_cps_reweighting",
        "1e_stratified_cps",
        "1f_source_imputation",
        "1g_stage_base_datasets",
        "2a_matrix_build_calibration_target_construction",
        "3a_weight_fitting_regional",
        "3b_weight_fitting_national",
        "4a_local_area_h5_regional",
        "4b_local_area_h5_national",
        "4d_upload_diagnostics",
        "5a_validate_outputs",
        "5b_promote_huggingface",
        "5c_promote_gcs",
        "5d_write_version_manifest",
    ]
    assert bundle["metadata"]["decorated_object_count"] >= 70
    assert bundle["metadata"]["mapped_decorated_node_count"] >= 45
    assert sum(len(stage["nodes"]) for stage in bundle["stages"]) >= 160
    assert sum(len(stage["edges"]) for stage in bundle["stages"]) >= 170
    stage2 = next(
        stage
        for stage in bundle["stages"]
        if stage["id"] == "2a_matrix_build_calibration_target_construction"
    )
    stage2_node_ids = {node["id"] for node in stage2["nodes"]}
    assert {
        "stage2_input_bundle",
        "stage2_build_context",
        "stage2_target_config_identity",
        "stage2_target_catalog_load",
        "stage2_target_catalog_reader",
        "stage2_target_selection_policy",
        "stage2_target_selection_result",
        "stage2_geography_assignment_spec",
        "stage2_geography_assignment_result",
        "stage2_matrix_build_spec",
        "stage2_matrix_build_service",
        "stage2_chunk_build_request",
        "stage2_chunk_worker_result",
        "stage2_matrix_build_result",
        "build_matrix",
        "build_matrix_chunked",
        "stage2_payload_boundary",
        "stage2_payload_reader",
        "stage2_payload_writer",
        "stage2_calibration_package_writer",
        "stage2_calibration_package_contract_writer",
        "stage2_calibration_package_contract_validator",
    } <= stage2_node_ids
