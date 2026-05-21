from policyengine_us_data.fit_weights import FitScope, fit_artifacts_for_scope
from scripts.extract_pipeline_docs import scan_decorated_objects


def test_fit_weights_identity_nodes_are_in_generated_pipeline_docs() -> None:
    decorated = scan_decorated_objects()

    assert "fitted_weights_spec" in decorated
    assert "fitted_weights_artifacts" in decorated
    assert "fitted_weights_output_bundle" in decorated
    assert "fit_weights" in decorated["fitted_weights_spec"].metadata["pathways"]
    assert "fit_weights" in decorated["fitted_weights_artifacts"].metadata["pathways"]
    assert (
        "fit_weights" in decorated["fitted_weights_output_bundle"].metadata["pathways"]
    )


def test_stage_3_pipeline_map_labels_match_scoped_artifacts(stage_3_substage) -> None:
    regional_artifacts = fit_artifacts_for_scope(FitScope.REGIONAL)
    national_artifacts = fit_artifacts_for_scope(FitScope.NATIONAL)
    regional = stage_3_substage("3a_weight_fitting_regional")
    national = stage_3_substage("3b_weight_fitting_national")

    regional_nodes = {node["id"]: node for node in regional["extra_nodes"]}
    national_nodes = {node["id"]: node for node in national["extra_nodes"]}

    assert "fit_spec_regional" in regional["groups"][0]["node_ids"]
    assert "fit_artifacts_regional" in regional["groups"][0]["node_ids"]
    assert regional_nodes["out_weights"]["label"] == regional_artifacts.weights.filename
    assert regional_nodes["out_geo_s6"]["label"] == (
        regional_artifacts.geography.filename
    )
    assert regional_nodes["out_config_s6"]["label"] == (
        regional_artifacts.run_config.filename
    )

    assert "fit_spec_national" in national["groups"][0]["node_ids"]
    assert "fit_artifacts_national" in national["groups"][0]["node_ids"]
    assert national_nodes["out_national_weights"]["label"] == (
        national_artifacts.weights.filename
    )
    assert national_nodes["out_national_geo_s6"]["label"] == (
        national_artifacts.geography.filename
    )
    assert national_nodes["out_national_config_s6"]["label"] == (
        national_artifacts.run_config.filename
    )
