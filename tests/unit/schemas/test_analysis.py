"""Tests for analysis schemas (Finding, Violation).

TDD RED Phase: Tests written before implementation.

Acceptance Criteria Coverage:
- AC-4.4: Finding/Violation models for analysis
- AC-4.5: All schemas have JSON schema export

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Functions (analyze_artifact, validate_against_spec)
"""

import pytest
from datetime import datetime, timezone
from typing import Any


# =============================================================================
# AC-4.4: Finding Tests
# =============================================================================

class TestFinding:
    """Tests for Finding model used in analyze_artifact output."""
    
    def test_finding_creation(self) -> None:
        """Finding captures analysis result fields."""
        from src.schemas.analysis import Finding
        
        finding = Finding(
            severity="high",
            category="security",
            description="SQL injection vulnerability detected",
            location="src/db/queries.py:42",
            fix_hint="Use parameterized queries instead of string concatenation",
        )
        
        assert finding.severity == "high"
        assert finding.category == "security"
        assert finding.description == "SQL injection vulnerability detected"
        assert finding.location == "src/db/queries.py:42"
        assert finding.fix_hint == "Use parameterized queries instead of string concatenation"
    
    def test_finding_severity_levels(self) -> None:
        """Finding supports severity levels: critical, high, medium, low, info."""
        from src.schemas.analysis import Finding, Severity
        
        for severity in ["critical", "high", "medium", "low", "info"]:
            finding = Finding(
                severity=severity,
                category="quality",
                description=f"Test {severity} finding",
            )
            assert finding.severity == severity
    
    def test_finding_severity_validation(self) -> None:
        """Finding validates severity enum values."""
        from src.schemas.analysis import Finding
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            Finding(
                severity="invalid_severity",
                category="quality",
                description="Test",
            )
    
    def test_finding_categories(self) -> None:
        """Finding supports multiple category types."""
        from src.schemas.analysis import Finding
        
        categories = ["quality", "security", "patterns", "dependencies", "performance"]
        
        for category in categories:
            finding = Finding(
                severity="medium",
                category=category,
                description=f"Test {category} finding",
            )
            assert finding.category == category
    
    def test_finding_optional_fields(self) -> None:
        """Finding allows optional location and fix_hint."""
        from src.schemas.analysis import Finding
        
        finding = Finding(
            severity="low",
            category="quality",
            description="Minor code smell detected",
        )
        
        assert finding.location is None
        assert finding.fix_hint is None
    
    def test_finding_with_code_snippet(self) -> None:
        """Finding can include code snippet context."""
        from src.schemas.analysis import Finding
        
        finding = Finding(
            severity="medium",
            category="quality",
            description="Function too complex",
            location="src/utils.py:100-150",
            code_snippet="def process_data(...):\n    # 50 lines of nested logic",
        )
        
        assert finding.code_snippet is not None
        assert "def process_data" in finding.code_snippet
    
    def test_finding_with_rule_id(self) -> None:
        """Finding can include rule/check ID reference."""
        from src.schemas.analysis import Finding
        
        finding = Finding(
            severity="high",
            category="quality",
            description="Cognitive complexity too high",
            rule_id="S3776",
        )
        
        assert finding.rule_id == "S3776"
    
    def test_finding_comparison_by_severity(self) -> None:
        """Findings can be compared by severity for sorting."""
        from src.schemas.analysis import Finding
        
        critical = Finding(severity="critical", category="security", description="A")
        high = Finding(severity="high", category="security", description="B")
        low = Finding(severity="low", category="quality", description="C")
        
        # severity_order property for sorting
        assert critical.severity_order < high.severity_order
        assert high.severity_order < low.severity_order
    
    def test_finding_json_schema_export(self) -> None:
        """Finding exports valid JSON Schema (AC-4.5)."""
        from src.schemas.analysis import Finding
        
        schema = Finding.model_json_schema()
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "severity" in schema["properties"]
        assert "category" in schema["properties"]
        assert "description" in schema["properties"]


# =============================================================================
# AC-4.4: Violation Tests
# =============================================================================

class TestViolation:
    """Tests for Violation model used in validate_against_spec output."""
    
    def test_violation_creation(self) -> None:
        """Violation captures spec compliance failure."""
        from src.schemas.analysis import Violation
        
        violation = Violation(
            requirement_id="REQ-001",
            description="Missing required field 'user_id'",
            expected="user_id: str field present",
            actual="user_id field not found",
            line_number=42,
        )
        
        assert violation.requirement_id == "REQ-001"
        assert violation.description == "Missing required field 'user_id'"
        assert violation.expected == "user_id: str field present"
        assert violation.actual == "user_id field not found"
        assert violation.line_number == 42
    
    def test_violation_without_line_number(self) -> None:
        """Violation allows missing line_number."""
        from src.schemas.analysis import Violation
        
        violation = Violation(
            requirement_id="REQ-002",
            description="Overall structure mismatch",
            expected="Repository pattern",
            actual="Active Record pattern",
        )
        
        assert violation.line_number is None
    
    def test_violation_with_suggested_fix(self) -> None:
        """Violation can include suggested fix."""
        from src.schemas.analysis import Violation
        
        violation = Violation(
            requirement_id="REQ-003",
            description="Incorrect return type",
            expected="list[User]",
            actual="User",
            suggested_fix="Change return type annotation from User to list[User]",
        )
        
        assert violation.suggested_fix is not None
        assert "list[User]" in violation.suggested_fix
    
    def test_violation_severity_inferred(self) -> None:
        """Violation has severity based on requirement type."""
        from src.schemas.analysis import Violation
        
        # Critical requirement
        critical_violation = Violation(
            requirement_id="REQ-CRITICAL-001",
            severity="critical",
            description="Missing auth check",
            expected="Authorization required",
            actual="No auth",
        )
        
        # Optional requirement
        optional_violation = Violation(
            requirement_id="OPT-001",
            severity="low",
            description="Missing optional logging",
            expected="Debug logging",
            actual="No logging",
        )
        
        assert critical_violation.severity == "critical"
        assert optional_violation.severity == "low"
    
    def test_violation_default_severity(self) -> None:
        """Violation defaults to medium severity."""
        from src.schemas.analysis import Violation
        
        violation = Violation(
            requirement_id="REQ-001",
            description="Test",
            expected="A",
            actual="B",
        )
        
        assert violation.severity == "medium"
    
    def test_violation_diff_format(self) -> None:
        """Violation.diff_format() returns expected vs actual comparison."""
        from src.schemas.analysis import Violation
        
        violation = Violation(
            requirement_id="REQ-001",
            description="Type mismatch",
            expected="int",
            actual="str",
        )
        
        diff = violation.diff_format()
        
        assert "expected:" in diff.lower() or "Expected:" in diff
        assert "actual:" in diff.lower() or "Actual:" in diff
        assert "int" in diff
        assert "str" in diff
    
    def test_violation_json_schema_export(self) -> None:
        """Violation exports valid JSON Schema (AC-4.5)."""
        from src.schemas.analysis import Violation
        
        schema = Violation.model_json_schema()
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "requirement_id" in schema["properties"]
        assert "expected" in schema["properties"]
        assert "actual" in schema["properties"]


# =============================================================================
# AnalysisResult Tests (Container for Findings)
# =============================================================================

class TestAnalysisResult:
    """Tests for AnalysisResult container model."""
    
    def test_analysis_result_creation(self) -> None:
        """AnalysisResult aggregates findings and metrics."""
        from src.schemas.analysis import AnalysisResult, Finding
        
        findings = [
            Finding(severity="high", category="security", description="SQL injection"),
            Finding(severity="low", category="quality", description="Long function"),
        ]
        
        result = AnalysisResult(
            findings=findings,
            metrics={"loc": 500, "cc_avg": 8.5},
            passed=False,
        )
        
        assert len(result.findings) == 2
        assert result.metrics["loc"] == 500
        assert result.passed is False
    
    def test_analysis_result_empty_findings(self) -> None:
        """AnalysisResult with no findings indicates pass."""
        from src.schemas.analysis import AnalysisResult
        
        result = AnalysisResult(
            findings=[],
            metrics={"loc": 100, "cc_avg": 3.0},
            passed=True,
        )
        
        assert result.findings == []
        assert result.passed is True
    
    def test_analysis_result_findings_by_severity(self) -> None:
        """AnalysisResult.findings_by_severity() groups findings."""
        from src.schemas.analysis import AnalysisResult, Finding
        
        findings = [
            Finding(severity="critical", category="security", description="A"),
            Finding(severity="high", category="security", description="B"),
            Finding(severity="high", category="quality", description="C"),
            Finding(severity="low", category="quality", description="D"),
        ]
        
        result = AnalysisResult(findings=findings, passed=False)
        by_severity = result.findings_by_severity()
        
        assert len(by_severity["critical"]) == 1
        assert len(by_severity["high"]) == 2
        assert len(by_severity["low"]) == 1
    
    def test_analysis_result_findings_by_category(self) -> None:
        """AnalysisResult.findings_by_category() groups findings."""
        from src.schemas.analysis import AnalysisResult, Finding
        
        findings = [
            Finding(severity="high", category="security", description="A"),
            Finding(severity="medium", category="security", description="B"),
            Finding(severity="low", category="quality", description="C"),
        ]
        
        result = AnalysisResult(findings=findings, passed=False)
        by_category = result.findings_by_category()
        
        assert len(by_category["security"]) == 2
        assert len(by_category["quality"]) == 1
    
    def test_analysis_result_critical_count(self) -> None:
        """AnalysisResult.critical_count returns count of critical findings."""
        from src.schemas.analysis import AnalysisResult, Finding
        
        findings = [
            Finding(severity="critical", category="security", description="A"),
            Finding(severity="critical", category="security", description="B"),
            Finding(severity="high", category="quality", description="C"),
        ]
        
        result = AnalysisResult(findings=findings, passed=False)
        
        assert result.critical_count == 2
    
    def test_analysis_result_compressed_report(self) -> None:
        """AnalysisResult.compressed_report for downstream consumption."""
        from src.schemas.analysis import AnalysisResult, Finding
        
        result = AnalysisResult(
            findings=[Finding(severity="high", category="security", description="Issue")],
            metrics={"loc": 100},
            passed=False,
            compressed_report="SECURITY: 1 high finding in src/db.py",
        )
        
        assert result.compressed_report is not None
        assert "SECURITY" in result.compressed_report
    
    def test_analysis_result_json_schema_export(self) -> None:
        """AnalysisResult exports valid JSON Schema (AC-4.5)."""
        from src.schemas.analysis import AnalysisResult
        
        schema = AnalysisResult.model_json_schema()
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "findings" in schema["properties"]
        assert "passed" in schema["properties"]


# =============================================================================
# ValidationResult Tests (Container for Violations)
# =============================================================================

class TestValidationResult:
    """Tests for ValidationResult container model."""
    
    def test_validation_result_creation(self) -> None:
        """ValidationResult aggregates violations and compliance."""
        from src.schemas.analysis import ValidationResult, Violation
        
        violations = [
            Violation(
                requirement_id="REQ-001",
                description="Missing field",
                expected="present",
                actual="missing",
            ),
        ]
        
        result = ValidationResult(
            valid=False,
            violations=violations,
            compliance_percentage=80.0,
            confidence=0.95,
        )
        
        assert result.valid is False
        assert len(result.violations) == 1
        assert result.compliance_percentage == pytest.approx(80.0)
        assert result.confidence == pytest.approx(0.95)
    
    def test_validation_result_fully_valid(self) -> None:
        """ValidationResult with no violations is 100% compliant."""
        from src.schemas.analysis import ValidationResult
        
        result = ValidationResult(
            valid=True,
            violations=[],
            compliance_percentage=100.0,
            confidence=0.99,
        )
        
        assert result.valid is True
        assert result.violations == []
        assert result.compliance_percentage == pytest.approx(100.0)
    
    def test_validation_result_compliance_range(self) -> None:
        """ValidationResult.compliance_percentage is 0-100."""
        from src.schemas.analysis import ValidationResult
        from pydantic import ValidationError
        
        # Valid range
        result = ValidationResult(valid=True, compliance_percentage=50.0)
        assert 0.0 <= result.compliance_percentage <= 100.0
        
        # Invalid: below 0
        with pytest.raises(ValidationError):
            ValidationResult(valid=False, compliance_percentage=-5.0)
        
        # Invalid: above 100
        with pytest.raises(ValidationError):
            ValidationResult(valid=False, compliance_percentage=105.0)
    
    def test_validation_result_confidence_range(self) -> None:
        """ValidationResult.confidence is 0.0-1.0."""
        from src.schemas.analysis import ValidationResult
        from pydantic import ValidationError
        
        # Valid range
        result = ValidationResult(valid=True, confidence=0.85)
        assert 0.0 <= result.confidence <= 1.0
        
        # Invalid: below 0
        with pytest.raises(ValidationError):
            ValidationResult(valid=True, confidence=-0.1)
        
        # Invalid: above 1
        with pytest.raises(ValidationError):
            ValidationResult(valid=True, confidence=1.5)
    
    def test_validation_result_remediation_hints(self) -> None:
        """ValidationResult can include remediation hints."""
        from src.schemas.analysis import ValidationResult
        
        result = ValidationResult(
            valid=False,
            compliance_percentage=70.0,
            remediation_hints=[
                "Add missing 'user_id' field to request schema",
                "Implement error handling for network failures",
            ],
        )
        
        assert len(result.remediation_hints) == 2
    
    def test_validation_result_json_schema_export(self) -> None:
        """ValidationResult exports valid JSON Schema (AC-4.5)."""
        from src.schemas.analysis import ValidationResult
        
        schema = ValidationResult.model_json_schema()
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "valid" in schema["properties"]
        assert "violations" in schema["properties"]
        assert "compliance_percentage" in schema["properties"]
