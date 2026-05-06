from policyengine_us_data.utils.run_context import (
    RunContext,
    build_modal_resource_name,
    build_run_id,
    resolve_run_id,
    sanitize_run_id,
    staging_prefix,
)


def test_run_id_from_github_identity() -> None:
    assert (
        build_run_id(
            github_run_id="123456789",
            github_run_attempt="2",
            github_sha="abcdef123456",
        )
        == "usdata-gha123456789-a2-abcdef12"
    )


def test_run_id_sanitizes_for_modal_and_hf_paths() -> None:
    assert sanitize_run_id("Feature/Some PR #12!") == "feature-some-pr-12"


def test_modal_resource_name_uses_safe_prefix_and_truncates() -> None:
    run_id = "usdata-gha123456789-a1-" + ("a" * 80)

    name = build_modal_resource_name(run_id, prefix="policyengine-us-data-pub")

    assert name.startswith("policyengine-us-data-pub-usdata-gha123456789-a1")
    assert len(name) <= 64


def test_resolve_run_id_prefers_explicit_value() -> None:
    env = {
        "US_DATA_RUN_ID": "from-env",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SHA": "abcdef12",
    }

    assert resolve_run_id("Explicit Value", env=env) == "explicit-value"


def test_resolve_run_id_ignores_raw_github_actions_identity() -> None:
    env = {
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SHA": "abcdef12",
    }

    assert resolve_run_id(env=env) == ""


def test_resolve_run_id_ignores_generic_run_id_alias() -> None:
    assert resolve_run_id(env={"RUN_ID": "alias-run"}) == ""


def test_run_context_from_env_records_cross_system_identity() -> None:
    run_id = build_run_id(
        github_run_id="123456789",
        github_run_attempt="1",
        github_sha="abcdef123456",
    )
    env = {
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "PolicyEngine/policyengine-us-data",
        "GITHUB_WORKFLOW": "Run Pipeline",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_REF_NAME": "main",
        "GITHUB_SHA": "abcdef123456",
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_RUN_ATTEMPT": "1",
        "US_DATA_RUN_ID": run_id,
        "US_DATA_PIPELINE_VOLUME_NAME": "pipeline-artifacts-test",
        "US_DATA_STAGING_VOLUME_NAME": "local-area-staging-test",
        "US_DATA_CHECKPOINT_VOLUME_NAME": "data-build-checkpoints-test",
    }

    context = RunContext.from_env(env=env)

    assert context.run_id == run_id
    assert context.modal_app_name == (
        "policyengine-us-data-pub-usdata-gha123456789-a1-abcdef12"
    )
    assert context.modal_environment == "main"
    assert context.hf_staging_prefix == staging_prefix(context.run_id)
    assert context.github_run_url == (
        "https://github.com/PolicyEngine/policyengine-us-data/actions/runs/123456789"
    )
    assert context.pipeline_volume_name == "pipeline-artifacts-test"
    assert context.staging_volume_name == "local-area-staging-test"
    assert context.checkpoint_volume_name == "data-build-checkpoints-test"


def test_run_context_export_env_includes_modal_and_hf_values() -> None:
    context = RunContext.from_env(
        env={"US_DATA_RUN_ID": "run-123"},
        modal_app_name="policyengine-us-data-pub-run-123",
        modal_environment="main",
    )

    exported = context.export_env()

    assert exported["US_DATA_RUN_ID"] == "run-123"
    assert exported["MODAL_APP_NAME"] == "policyengine-us-data-pub-run-123"
    assert exported["MODAL_ENVIRONMENT"] == "main"
    assert exported["US_DATA_HF_STAGING_PREFIX"] == "staging/run-123"
