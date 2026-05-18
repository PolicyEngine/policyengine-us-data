"""Shared validation execution primitives for stage contract validation."""

from .checks import ValidationCheck, ValidationSuite
from .context import ValidationArtifactResolver, ValidationContext
from .runner import ValidationRunner
from .writers import (
    ValidationFindingsJsonlOutputStrategy,
    ValidationReportJsonOutputStrategy,
    ValidationReportOutput,
    ValidationReportOutputStrategy,
    ValidationReportWriter,
    ValidationResultWriter,
    ValidationSummaryJsonOutputStrategy,
)

__all__ = [
    "ValidationArtifactResolver",
    "ValidationCheck",
    "ValidationContext",
    "ValidationFindingsJsonlOutputStrategy",
    "ValidationReportJsonOutputStrategy",
    "ValidationReportOutput",
    "ValidationReportOutputStrategy",
    "ValidationReportWriter",
    "ValidationResultWriter",
    "ValidationRunner",
    "ValidationSummaryJsonOutputStrategy",
    "ValidationSuite",
]
