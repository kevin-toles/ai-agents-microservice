"""Analysis schemas for ai-agents (Finding, Violation, Results).

Implements AC-4.4 from WBS-AGT4.

Models:
- Severity: Enum for finding severity levels
- Finding: Individual analysis result for analyze_artifact
- Violation: Spec compliance failure for validate_against_spec
- AnalysisResult: Container for findings with metrics
- ValidationResult: Container for violations with compliance %

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Functions (analyze_artifact, validate_against_spec)

Anti-Pattern Compliance:
- AP-1.5: No mutable default arguments (uses Field(default_factory=list/dict))
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================

class Severity(str, Enum):
    """Severity levels for findings and violations.
    
    Ordered from most to least severe for sorting.
    """
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# Severity ordering for comparison
SEVERITY_ORDER: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}


# =============================================================================
# AC-4.4: Finding
# =============================================================================

class Finding(BaseModel):
    """Individual analysis finding from analyze_artifact.
    
    Represents a quality, security, pattern, or dependency issue
    detected during artifact analysis.
    
    Attributes:
        severity: Issue severity (critical, high, medium, low, info)
        category: Type of finding (quality, security, patterns, etc.)
        description: Human-readable description of the issue
        location: Optional file:line or path reference
        fix_hint: Optional suggestion for remediation
        code_snippet: Optional relevant code excerpt
        rule_id: Optional SonarQube/linter rule ID
    
    Example:
        >>> finding = Finding(
        ...     severity="high",
        ...     category="security",
        ...     description="SQL injection vulnerability",
        ...     location="src/db/queries.py:42",
        ...     rule_id="S3649",
        ... )
    """
    
    severity: Severity = Field(
        ...,
        description="Severity level: critical, high, medium, low, info",
    )
    category: str = Field(
        ...,
        description="Category: quality, security, patterns, dependencies, performance",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the issue",
    )
    location: str | None = Field(
        default=None,
        description="File path and line number, e.g., 'src/db.py:42'",
    )
    fix_hint: str | None = Field(
        default=None,
        description="Suggested fix or remediation",
    )
    code_snippet: str | None = Field(
        default=None,
        description="Relevant code excerpt",
    )
    rule_id: str | None = Field(
        default=None,
        description="SonarQube or linter rule ID, e.g., 'S3776'",
    )
    
    @property
    def severity_order(self) -> int:
        """Return numeric severity for sorting (lower = more severe).
        
        Returns:
            Integer for sorting: 0=critical, 1=high, 2=medium, 3=low, 4=info
        """
        return SEVERITY_ORDER.get(self.severity.value, 5)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "severity": "high",
                    "category": "security",
                    "description": "SQL injection vulnerability detected",
                    "location": "src/db/queries.py:42",
                    "fix_hint": "Use parameterized queries",
                    "rule_id": "S3649",
                }
            ]
        }
    }


# =============================================================================
# AC-4.4: Violation
# =============================================================================

class Violation(BaseModel):
    """Specification compliance violation from validate_against_spec.
    
    Represents a mismatch between an artifact and its specification
    or acceptance criteria.
    
    Attributes:
        requirement_id: Identifier for the violated requirement
        description: Human-readable description of the violation
        expected: What the spec requires
        actual: What was found in the artifact
        line_number: Optional line number in artifact
        severity: Violation severity (defaults to medium)
        suggested_fix: Optional fix suggestion
    
    Example:
        >>> violation = Violation(
        ...     requirement_id="REQ-001",
        ...     description="Missing required field",
        ...     expected="user_id: str field present",
        ...     actual="user_id field not found",
        ...     line_number=42,
        ... )
    """
    
    requirement_id: str = Field(
        ...,
        description="Requirement or criteria identifier",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the violation",
    )
    expected: str = Field(
        ...,
        description="What the specification requires",
    )
    actual: str = Field(
        ...,
        description="What was found in the artifact",
    )
    line_number: int | None = Field(
        default=None,
        description="Line number in artifact where violation occurs",
        ge=1,
    )
    severity: Severity = Field(
        default=Severity.MEDIUM,
        description="Violation severity",
    )
    suggested_fix: str | None = Field(
        default=None,
        description="Suggested fix for the violation",
    )
    
    def diff_format(self) -> str:
        """Format violation as expected vs actual diff.
        
        Returns:
            Formatted diff string
        """
        lines = [
            f"Requirement: {self.requirement_id}",
            f"Expected: {self.expected}",
            f"Actual: {self.actual}",
        ]
        if self.line_number:
            lines.insert(1, f"Line: {self.line_number}")
        return "\n".join(lines)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "requirement_id": "REQ-001",
                    "description": "Missing required field 'user_id'",
                    "expected": "user_id: str field present",
                    "actual": "user_id field not found",
                    "line_number": 42,
                    "severity": "medium",
                }
            ]
        }
    }


# =============================================================================
# AnalysisResult Container
# =============================================================================

class AnalysisResult(BaseModel):
    """Container for analyze_artifact output.
    
    Aggregates findings from artifact analysis along with
    metrics and overall pass/fail status.
    
    Attributes:
        findings: List of findings from analysis
        metrics: Optional metrics dict (loc, cc_avg, etc.)
        passed: Overall analysis pass/fail
        compressed_report: Optional summary for downstream
    
    Example:
        >>> result = AnalysisResult(
        ...     findings=[finding1, finding2],
        ...     metrics={"loc": 500, "cc_avg": 8.5},
        ...     passed=False,
        ... )
    """
    
    # AP-1.5: Use Field(default_factory=list) instead of default=[]
    findings: list[Finding] = Field(
        default_factory=list,
        description="List of analysis findings",
    )
    # AP-1.5: Use Field(default_factory=dict) instead of default={}
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis metrics (loc, cc_avg, etc.)",
    )
    passed: bool = Field(
        default=True,
        description="Overall analysis pass/fail status",
    )
    compressed_report: str | None = Field(
        default=None,
        description="Compressed summary for downstream consumption",
    )
    
    @property
    def critical_count(self) -> int:
        """Count of critical severity findings.
        
        Returns:
            Number of critical findings
        """
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)
    
    @property
    def high_count(self) -> int:
        """Count of high severity findings.
        
        Returns:
            Number of high findings
        """
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)
    
    def findings_by_severity(self) -> dict[str, list[Finding]]:
        """Group findings by severity level.
        
        Returns:
            Dict mapping severity to list of findings
        """
        result: dict[str, list[Finding]] = defaultdict(list)
        for finding in self.findings:
            result[finding.severity.value].append(finding)
        return dict(result)
    
    def findings_by_category(self) -> dict[str, list[Finding]]:
        """Group findings by category.
        
        Returns:
            Dict mapping category to list of findings
        """
        result: dict[str, list[Finding]] = defaultdict(list)
        for finding in self.findings:
            result[finding.category].append(finding)
        return dict(result)
    
    def sorted_findings(self) -> list[Finding]:
        """Return findings sorted by severity (most severe first).
        
        Returns:
            Sorted list of findings
        """
        return sorted(self.findings, key=lambda f: f.severity_order)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "findings": [
                        {
                            "severity": "high",
                            "category": "security",
                            "description": "SQL injection vulnerability",
                        }
                    ],
                    "metrics": {"loc": 500, "cc_avg": 8.5},
                    "passed": False,
                }
            ]
        }
    }


# =============================================================================
# ValidationResult Container
# =============================================================================

class ValidationResult(BaseModel):
    """Container for validate_against_spec output.
    
    Aggregates violations from spec validation along with
    compliance percentage and confidence.
    
    Attributes:
        valid: Overall validation pass/fail
        violations: List of spec violations
        compliance_percentage: Percentage of requirements met (0-100)
        confidence: Model confidence in validation (0.0-1.0)
        remediation_hints: Optional list of fix suggestions
    
    Example:
        >>> result = ValidationResult(
        ...     valid=False,
        ...     violations=[violation1],
        ...     compliance_percentage=80.0,
        ...     confidence=0.95,
        ... )
    """
    
    valid: bool = Field(
        ...,
        description="Overall validation pass/fail status",
    )
    # AP-1.5: Use Field(default_factory=list) instead of default=[]
    violations: list[Violation] = Field(
        default_factory=list,
        description="List of spec violations",
    )
    compliance_percentage: float = Field(
        default=100.0,
        description="Percentage of requirements met (0-100)",
        ge=0.0,
        le=100.0,
    )
    confidence: float = Field(
        default=1.0,
        description="Model confidence in validation result (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    # AP-1.5: Use Field(default_factory=list) instead of default=[]
    remediation_hints: list[str] = Field(
        default_factory=list,
        description="Suggested remediation steps",
    )
    
    @property
    def is_fully_compliant(self) -> bool:
        """Check if artifact is fully compliant with spec.
        
        Returns:
            True if no violations and 100% compliance
        """
        return self.valid and not self.violations and self.compliance_percentage == 100.0
    
    def violations_by_severity(self) -> dict[str, list[Violation]]:
        """Group violations by severity level.
        
        Returns:
            Dict mapping severity to list of violations
        """
        result: dict[str, list[Violation]] = defaultdict(list)
        for violation in self.violations:
            result[violation.severity.value].append(violation)
        return dict(result)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "valid": False,
                    "violations": [
                        {
                            "requirement_id": "REQ-001",
                            "description": "Missing field",
                            "expected": "present",
                            "actual": "missing",
                        }
                    ],
                    "compliance_percentage": 80.0,
                    "confidence": 0.95,
                }
            ]
        }
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "Severity",
    # Models
    "Finding",
    "Violation",
    "AnalysisResult",
    "ValidationResult",
    # Constants
    "SEVERITY_ORDER",
]
