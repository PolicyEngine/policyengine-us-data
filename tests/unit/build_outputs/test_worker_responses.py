from policyengine_us_data.build_outputs.worker_responses import (
    normalize_worker_response,
)


def test_normalize_worker_response_marks_fatal_and_nonfatal_issues():
    result = normalize_worker_response(
        worker_index=2,
        result={
            "completed": ["district:NC-01"],
            "failed": [],
            "errors": [{"error": "Failed to parse worker output"}],
            "issues": [
                {
                    "item": "district:NC-01",
                    "phase": "validation",
                    "error": "validation warning",
                }
            ],
            "validation_rows": [],
        },
    )

    assert result.completed == ("district:NC-01",)
    assert result.failed == ()
    assert result.fatal_errors == (
        {
            "error": "Failed to parse worker output",
            "worker": 2,
            "severity": "worker_failure",
        },
    )
    assert result.issues == (
        {
            "item": "district:NC-01",
            "phase": "validation",
            "error": "validation warning",
            "worker": 2,
            "severity": "validation",
        },
    )


def test_normalize_worker_response_marks_malformed_fields_as_protocol_errors():
    result = normalize_worker_response(
        worker_index=1,
        result={
            "completed": "district:NC-01",
            "failed": [],
            "errors": [],
            "validation_rows": [],
        },
    )

    assert result.completed == ()
    assert result.fatal_errors == (
        {
            "phase": "protocol",
            "error": "Worker result field 'completed' must be a list",
            "worker": 1,
            "severity": "protocol",
        },
    )


def test_normalize_worker_response_marks_failed_items_without_errors():
    result = normalize_worker_response(
        worker_index=0,
        result={
            "completed": [],
            "failed": ["district:NC-01"],
            "errors": [],
            "issues": [],
            "validation_rows": [],
        },
    )

    assert result.fatal_errors == (
        {
            "item": "district:NC-01",
            "phase": "worker",
            "error": "Worker reported failed item without a matching error",
            "worker": 0,
            "severity": "worker_failure",
        },
    )


def test_normalize_worker_response_keeps_validation_errors_nonfatal():
    result = normalize_worker_response(
        worker_index=3,
        result={
            "completed": ["district:NC-01"],
            "failed": [],
            "errors": [
                {
                    "item": "district:NC-01",
                    "phase": "validation",
                    "error": "validation failed",
                    "severity": "validation",
                }
            ],
            "issues": [
                {
                    "item": "district:NC-01",
                    "phase": "validation",
                    "error": "validation failed",
                    "severity": "validation",
                }
            ],
            "validation_rows": [],
        },
    )

    assert result.completed == ("district:NC-01",)
    assert result.failed == ()
    assert result.fatal_errors == ()
    assert result.issues == (
        {
            "item": "district:NC-01",
            "phase": "validation",
            "error": "validation failed",
            "severity": "validation",
            "worker": 3,
        },
    )
