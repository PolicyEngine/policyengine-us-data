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

    assert bundle["metadata"]["stage_count"] == 1
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

    assert bundle["metadata"]["stage_count"] == 5
    assert bundle["metadata"]["decorated_object_count"] >= 70
    assert bundle["metadata"]["mapped_decorated_node_count"] >= 15
