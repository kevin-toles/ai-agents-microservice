"""Schemas for validate_against_spec function.

WBS-AGT10: validate_against_spec Function

This module defines the input/output schemas for the validate_against_spec
function, which compares artifacts against specifications and acceptance criteria.

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 5
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ViolationSeverity(str, Enum):
    """Severity level for a validation violation."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Violation(BaseModel):
    """A single violation detected during validation.

    AC-10.5: Violations include line_number, expected, actual.
    Each violation must have an expected vs actual comparison.

    Attributes:
        expected: What was expected according to the specification
        actual: What was actually found in the artifact
        description: Human-readable description of the violation
        line_number: Optional line number where violation occurs
        severity: Severity level of the violation
        criterion_id: Optional ID linking to specific acceptance criterion
    """

    expected: str = Field(
        ...,
        description="What was expected according to the specification",
    )
    actual: str = Field(
        ...,
        description="What was actually found in the artifact",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the violation",
    )
    line_number: int | None = Field(
        default=None,
        description="Line number where violation occurs",
    )
    severity: ViolationSeverity = Field(
        default=ViolationSeverity.ERROR,
        description="Severity level of the violation",
    )
    criterion_id: str | None = Field(
        default=None,
        description="ID linking to specific acceptance criterion",
    )


class ValidationResult(BaseModel):
    """Result of validating an artifact against a specification.

    AC-10.2: Returns ValidationResult with compliance %, violations.

    Exit Criteria:
    - compliance_percentage is 0-100 float
    - Each Violation has expected vs actual comparison
    - Empty violations list → compliance_percentage = 100.0

    Attributes:
        valid: Whether the artifact passes validation
        violations: List of violations found
        compliance_percentage: Percentage of criteria met (0-100)
        confidence: Confidence level of the validation (0.0-1.0)
        remediation_hints: List of hints to fix violations
    """

    valid: bool = Field(
        default=True,
        description="Whether the artifact passes validation",
    )
    violations: list[Violation] = Field(
        default_factory=list,
        description="List of violations found during validation",
    )
    compliance_percentage: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Percentage of criteria met (0-100)",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level of the validation (0.0-1.0)",
    )
    remediation_hints: list[str] = Field(
        default_factory=list,
        description="List of hints to fix violations",
    )

    @field_validator("compliance_percentage")
    @classmethod
    def validate_compliance_percentage(cls, v: float) -> float:
        """Validate compliance_percentage is between 0 and 100."""
        if v < 0.0 or v > 100.0:
            raise ValueError("compliance_percentage must be between 0 and 100")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is between 0 and 1."""
        if v < 0.0 or v > 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v


class ValidateAgainstSpecInput(BaseModel):
    """Input schema for validate_against_spec function.

    AC-10.1: Compares artifact against specification.

    Attributes:
        artifact: The code or content to validate
        specification: The original requirement/specification
        invariants: List of invariants from upstream summarize_content
        acceptance_criteria: List of acceptance criteria to check
    """

    artifact: str = Field(
        ...,
        description="The code or content to validate",
    )
    specification: str = Field(
        ...,
        description="The original requirement/specification",
    )
    invariants: list[str] = Field(
        default_factory=list,
        description="List of invariants from upstream summarize_content",
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="List of acceptance criteria to check",
    )


__all__ = [
    "ValidateAgainstSpecInput",
    "ValidationResult",
    "Violation",
    "ViolationSeverity",
]
