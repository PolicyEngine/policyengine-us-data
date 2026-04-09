from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import extract_pipeline


class ExtractPipelineValidationTest(unittest.TestCase):
    def test_build_pipeline_json_rejects_missing_edge_target_without_groups(self):
        stages_yaml = {
            "stages": [
                {
                    "id": 0,
                    "label": "Stage 0",
                    "title": "Stage 0",
                    "description": "Validation test stage",
                    "extra_nodes": [
                        {
                            "id": "existing",
                            "label": "Existing node",
                            "node_type": "input",
                            "description": "Present in the stage",
                        }
                    ],
                    "edges": [
                        {
                            "source": "existing",
                            "target": "missing_target",
                            "edge_type": "data_flow",
                        }
                    ],
                }
            ]
        }

        with TemporaryDirectory() as temp_dir:
            with patch.object(
                extract_pipeline,
                "scan_source_files",
                return_value={},
            ):
                with patch.object(
                    extract_pipeline,
                    "load_stages_yaml",
                    return_value=stages_yaml,
                ):
                    with self.assertRaises(SystemExit) as exc:
                        extract_pipeline.build_pipeline_json(
                            Path(temp_dir) / "pipeline.json"
                        )

        self.assertEqual(str(exc.exception), "\n  1 validation error(s) found")
