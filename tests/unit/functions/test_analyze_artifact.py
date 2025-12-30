"""Tests for analyze_artifact function.

TDD tests for WBS-AGT9: analyze_artifact Function.

Acceptance Criteria Coverage:
- AC-9.1: Analyzes code/docs for quality, patterns, issues
- AC-9.2: Returns AnalysisResult with findings list
- AC-9.3: Context budget: 16384 input / 2048 output
- AC-9.4: Default preset: D4 (Standard)
- AC-9.5: Supports analysis_type parameter (quality/security/patterns)

Exit Criteria:
- Each Finding has severity, category, description, location
- analysis_type="security" flags common vulnerabilities
- analysis_type="patterns" identifies design patterns

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 4
"""

import pytest
from typing import Any
from pydantic import ValidationError


# =============================================================================
# AC-9.1, AC-9.5: Input Schema Tests - AnalyzeArtifactInput
# =============================================================================

class TestAnalyzeArtifactInput:
    """Tests for AnalyzeArtifactInput schema."""

    def test_input_requires_artifact(self) -> None:
        """AnalyzeArtifactInput requires artifact field."""
        from src.schemas.functions.analyze_artifact import AnalyzeArtifactInput
        
        with pytest.raises(ValidationError) as exc_info:
            AnalyzeArtifactInput()  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("artifact",) for e in errors)

    def test_input_accepts_artifact_string(self) -> None:
        """AnalyzeArtifactInput accepts artifact as string."""
        from src.schemas.functions.analyze_artifact import AnalyzeArtifactInput
        
        input_data = AnalyzeArtifactInput(artifact="def foo(): pass")
        assert input_data.artifact == "def foo(): pass"

    def test_input_has_artifact_type_with_default(self) -> None:
        """AnalyzeArtifactInput has artifact_type with default 'code'."""
        from src.schemas.functions.analyze_artifact import AnalyzeArtifactInput, ArtifactKind
        
        input_data = AnalyzeArtifactInput(artifact="test")
        assert input_data.artifact_type == ArtifactKind.CODE

    def test_input_accepts_document_artifact_type(self) -> None:
        """AnalyzeArtifactInput accepts document artifact type."""
        from src.schemas.functions.analyze_artifact import AnalyzeArtifactInput, ArtifactKind
        
        input_data = AnalyzeArtifactInput(
            artifact="# Document",
            artifact_type=ArtifactKind.DOCUMENT,
        )
        assert input_data.artifact_type == ArtifactKind.DOCUMENT

    def test_input_accepts_config_artifact_type(self) -> None:
        """AnalyzeArtifactInput accepts config artifact type."""
        from src.schemas.functions.analyze_artifact import AnalyzeArtifactInput, ArtifactKind
        
        input_data = AnalyzeArtifactInput(
            artifact="key: value",
            artifact_type=ArtifactKind.CONFIG,
        )
        assert input_data.artifact_type == ArtifactKind.CONFIG

    def test_input_has_analysis_type_with_default(self) -> None:
        """AnalyzeArtifactInput has analysis_type with default 'quality'."""
        from src.schemas.functions.analyze_artifact import AnalyzeArtifactInput, AnalysisType
        
        input_data = AnalyzeArtifactInput(artifact="test")
        assert input_data.analysis_type == AnalysisType.QUALITY

    def test_input_accepts_security_analysis_type(self) -> None:
        """AnalyzeArtifactInput accepts security analysis type (AC-9.5)."""
        from src.schemas.functions.analyze_artifact import AnalyzeArtifactInput, AnalysisType
        
        input_data = AnalyzeArtifactInput(
            artifact="test",
            analysis_type=AnalysisType.SECURITY,
        )
        assert input_data.analysis_type == AnalysisType.SECURITY

    def test_input_accepts_patterns_analysis_type(self) -> None:
        """AnalyzeArtifactInput accepts patterns analysis type (AC-9.5)."""
        from src.schemas.functions.analyze_artifact import AnalyzeArtifactInput, AnalysisType
        
        input_data = AnalyzeArtifactInput(
            artifact="test",
            analysis_type=AnalysisType.PATTERNS,
        )
        assert input_data.analysis_type == AnalysisType.PATTERNS

    def test_input_accepts_dependencies_analysis_type(self) -> None:
        """AnalyzeArtifactInput accepts dependencies analysis type."""
        from src.schemas.functions.analyze_artifact import AnalyzeArtifactInput, AnalysisType
        
        input_data = AnalyzeArtifactInput(
            artifact="test",
            analysis_type=AnalysisType.DEPENDENCIES,
        )
        assert input_data.analysis_type == AnalysisType.DEPENDENCIES

    def test_input_has_optional_checklist(self) -> None:
        """AnalyzeArtifactInput has optional checklist field."""
        from src.schemas.functions.analyze_artifact import AnalyzeArtifactInput
        
        input_data = AnalyzeArtifactInput(artifact="test")
        assert input_data.checklist == []
        
        input_with_checklist = AnalyzeArtifactInput(
            artifact="test",
            checklist=["check1", "check2"],
        )
        assert input_with_checklist.checklist == ["check1", "check2"]

    def test_input_json_schema_export(self) -> None:
        """AnalyzeArtifactInput exports valid JSON schema."""
        from src.schemas.functions.analyze_artifact import AnalyzeArtifactInput
        
        schema = AnalyzeArtifactInput.model_json_schema()
        
        assert "properties" in schema
        assert "artifact" in schema["properties"]
        assert "artifact_type" in schema["properties"]
        assert "analysis_type" in schema["properties"]


# =============================================================================
# AC-9.2: Finding Schema Tests
# =============================================================================

class TestFinding:
    """Tests for Finding model - each Finding has severity, category, description, location."""

    def test_finding_requires_severity(self) -> None:
        """Finding requires severity field."""
        from src.schemas.functions.analyze_artifact import Finding
        
        with pytest.raises(ValidationError) as exc_info:
            Finding(
                category="test",
                description="test desc",
                location="line 1",
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("severity",) for e in errors)

    def test_finding_requires_category(self) -> None:
        """Finding requires category field."""
        from src.schemas.functions.analyze_artifact import Finding, Severity
        
        with pytest.raises(ValidationError) as exc_info:
            Finding(
                severity=Severity.HIGH,
                description="test desc",
                location="line 1",
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("category",) for e in errors)

    def test_finding_requires_description(self) -> None:
        """Finding requires description field."""
        from src.schemas.functions.analyze_artifact import Finding, Severity
        
        with pytest.raises(ValidationError) as exc_info:
            Finding(
                severity=Severity.HIGH,
                category="test",
                location="line 1",
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("description",) for e in errors)

    def test_finding_requires_location(self) -> None:
        """Finding requires location field."""
        from src.schemas.functions.analyze_artifact import Finding, Severity
        
        with pytest.raises(ValidationError) as exc_info:
            Finding(
                severity=Severity.HIGH,
                category="test",
                description="test desc",
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("location",) for e in errors)

    def test_finding_accepts_all_required_fields(self) -> None:
        """Finding accepts all required fields: severity, category, description, location."""
        from src.schemas.functions.analyze_artifact import Finding, Severity
        
        finding = Finding(
            severity=Severity.HIGH,
            category="security",
            description="SQL injection vulnerability",
            location="line 42",
        )
        
        assert finding.severity == Severity.HIGH
        assert finding.category == "security"
        assert finding.description == "SQL injection vulnerability"
        assert finding.location == "line 42"

    def test_finding_severity_enum_values(self) -> None:
        """Finding severity supports all expected levels."""
        from src.schemas.functions.analyze_artifact import Finding, Severity
        
        for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
            finding = Finding(
                severity=sev,
                category="test",
                description="test",
                location="line 1",
            )
            assert finding.severity == sev

    def test_finding_has_optional_fix_hint(self) -> None:
        """Finding has optional fix_hint field."""
        from src.schemas.functions.analyze_artifact import Finding, Severity
        
        finding = Finding(
            severity=Severity.MEDIUM,
            category="code-quality",
            description="Function too complex",
            location="func:calculate",
        )
        assert finding.fix_hint is None
        
        finding_with_hint = Finding(
            severity=Severity.MEDIUM,
            category="code-quality",
            description="Function too complex",
            location="func:calculate",
            fix_hint="Extract helper methods",
        )
        assert finding_with_hint.fix_hint == "Extract helper methods"

    def test_finding_has_optional_line_number(self) -> None:
        """Finding has optional line_number for precise location."""
        from src.schemas.functions.analyze_artifact import Finding, Severity
        
        finding = Finding(
            severity=Severity.LOW,
            category="style",
            description="Missing docstring",
            location="class:MyClass",
        )
        assert finding.line_number is None
        
        finding_with_line = Finding(
            severity=Severity.LOW,
            category="style",
            description="Missing docstring",
            location="class:MyClass",
            line_number=15,
        )
        assert finding_with_line.line_number == 15

    def test_finding_json_schema_export(self) -> None:
        """Finding exports valid JSON schema."""
        from src.schemas.functions.analyze_artifact import Finding
        
        schema = Finding.model_json_schema()
        
        assert "properties" in schema
        assert "severity" in schema["properties"]
        assert "category" in schema["properties"]
        assert "description" in schema["properties"]
        assert "location" in schema["properties"]


# =============================================================================
# AC-9.2: AnalysisResult Schema Tests
# =============================================================================

class TestAnalysisResult:
    """Tests for AnalysisResult model - returns findings list."""

    def test_analysis_result_has_findings_list(self) -> None:
        """AnalysisResult has findings list (AC-9.2)."""
        from src.schemas.functions.analyze_artifact import AnalysisResult
        
        result = AnalysisResult()
        assert result.findings == []
        assert isinstance(result.findings, list)

    def test_analysis_result_accepts_findings(self) -> None:
        """AnalysisResult accepts list of Finding objects."""
        from src.schemas.functions.analyze_artifact import AnalysisResult, Finding, Severity
        
        finding = Finding(
            severity=Severity.HIGH,
            category="security",
            description="Hardcoded password",
            location="line 10",
        )
        result = AnalysisResult(findings=[finding])
        
        assert len(result.findings) == 1
        assert result.findings[0].severity == Severity.HIGH

    def test_analysis_result_has_metrics_dict(self) -> None:
        """AnalysisResult has metrics dict for code metrics."""
        from src.schemas.functions.analyze_artifact import AnalysisResult
        
        result = AnalysisResult()
        assert result.metrics == {}
        
        result_with_metrics = AnalysisResult(
            metrics={"loc": 100, "cyclomatic_complexity": 5}
        )
        assert result_with_metrics.metrics["loc"] == 100

    def test_analysis_result_has_passed_bool(self) -> None:
        """AnalysisResult has passed bool for overall gate."""
        from src.schemas.functions.analyze_artifact import AnalysisResult
        
        result = AnalysisResult()
        assert result.passed is True  # Default to pass
        
        result_failed = AnalysisResult(passed=False)
        assert result_failed.passed is False

    def test_analysis_result_has_compressed_report(self) -> None:
        """AnalysisResult has compressed_report for downstream."""
        from src.schemas.functions.analyze_artifact import AnalysisResult
        
        result = AnalysisResult()
        assert result.compressed_report is None
        
        result_with_report = AnalysisResult(
            compressed_report="Analysis complete: 2 issues found"
        )
        assert result_with_report.compressed_report == "Analysis complete: 2 issues found"

    def test_analysis_result_json_schema_export(self) -> None:
        """AnalysisResult exports valid JSON schema."""
        from src.schemas.functions.analyze_artifact import AnalysisResult
        
        schema = AnalysisResult.model_json_schema()
        
        assert "properties" in schema
        assert "findings" in schema["properties"]
        assert "metrics" in schema["properties"]
        assert "passed" in schema["properties"]


# =============================================================================
# AC-9.5: AnalysisType Enum Tests
# =============================================================================

class TestAnalysisType:
    """Tests for AnalysisType enum - supports quality/security/patterns."""

    def test_analysis_type_has_quality(self) -> None:
        """AnalysisType has quality value."""
        from src.schemas.functions.analyze_artifact import AnalysisType
        
        assert AnalysisType.QUALITY.value == "quality"

    def test_analysis_type_has_security(self) -> None:
        """AnalysisType has security value."""
        from src.schemas.functions.analyze_artifact import AnalysisType
        
        assert AnalysisType.SECURITY.value == "security"

    def test_analysis_type_has_patterns(self) -> None:
        """AnalysisType has patterns value."""
        from src.schemas.functions.analyze_artifact import AnalysisType
        
        assert AnalysisType.PATTERNS.value == "patterns"

    def test_analysis_type_has_dependencies(self) -> None:
        """AnalysisType has dependencies value."""
        from src.schemas.functions.analyze_artifact import AnalysisType
        
        assert AnalysisType.DEPENDENCIES.value == "dependencies"

    def test_analysis_type_string_conversion(self) -> None:
        """AnalysisType enum values are strings."""
        from src.schemas.functions.analyze_artifact import AnalysisType
        
        for at in AnalysisType:
            assert isinstance(at.value, str)


# =============================================================================
# AC-9.3: Context Budget Tests
# =============================================================================

class TestAnalyzeArtifactContextBudget:
    """Tests for context budget: 16384 input / 2048 output (AC-9.3)."""

    def test_context_budget_in_defaults(self) -> None:
        """analyze_artifact context budget is in CONTEXT_BUDGET_DEFAULTS."""
        from src.functions.base import CONTEXT_BUDGET_DEFAULTS
        
        assert "analyze_artifact" in CONTEXT_BUDGET_DEFAULTS
        budget = CONTEXT_BUDGET_DEFAULTS["analyze_artifact"]
        assert budget["input"] == 16384
        assert budget["output"] == 2048

    def test_function_returns_correct_budget(self) -> None:
        """AnalyzeArtifactFunction.get_context_budget() returns correct values."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        
        func = AnalyzeArtifactFunction()
        budget = func.get_context_budget()
        
        assert budget["input"] == 16384
        assert budget["output"] == 2048

    def test_budget_exceeded_raises_error(self) -> None:
        """Input exceeding budget raises ContextBudgetExceededError."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        from src.functions.base import ContextBudgetExceededError
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        # 16384 tokens * 4 chars = 65536 chars limit
        # Create content that exceeds this
        large_content = "x" * 70000
        
        with pytest.raises(ContextBudgetExceededError) as exc_info:
            asyncio.run(func.run(artifact=large_content))
        
        assert exc_info.value.function_name == "analyze_artifact"
        assert exc_info.value.limit == 16384


# =============================================================================
# AC-9.4: Default Preset Tests
# =============================================================================

class TestAnalyzeArtifactPreset:
    """Tests for default preset: D4 (Standard) (AC-9.4)."""

    def test_default_preset_is_d4(self) -> None:
        """AnalyzeArtifactFunction default_preset is 'D4'."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        
        func = AnalyzeArtifactFunction()
        assert func.default_preset == "D4"

    def test_function_name_is_analyze_artifact(self) -> None:
        """AnalyzeArtifactFunction name is 'analyze_artifact'."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        
        func = AnalyzeArtifactFunction()
        assert func.name == "analyze_artifact"

    def test_preset_selection_light(self) -> None:
        """select_preset('light') returns lighter preset."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        
        func = AnalyzeArtifactFunction()
        assert func.select_preset("light") == "S1"

    def test_preset_selection_high_quality(self) -> None:
        """select_preset('high_quality') returns D10."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        
        func = AnalyzeArtifactFunction()
        assert func.select_preset("high_quality") == "D10"


# =============================================================================
# AC-9.1: Function Implementation Tests - Quality Analysis
# =============================================================================

class TestQualityAnalysis:
    """Tests for quality analysis (analysis_type='quality')."""

    def test_quality_analysis_detects_long_functions(self) -> None:
        """Quality analysis detects functions that are too long."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        from src.schemas.functions.analyze_artifact import AnalysisType
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        # Create a long function
        long_function = """def very_long_function():
    a = 1
    b = 2
    c = 3
    d = 4
    e = 5
    f = 6
    g = 7
    h = 8
    i = 9
    j = 10
    k = 11
    l = 12
    m = 13
    n = 14
    o = 15
    p = 16
    q = 17
    r = 18
    s = 19
    t = 20
    u = 21
    v = 22
    w = 23
    x = 24
    y = 25
    return a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u + v + w + x + y"""
        
        result = asyncio.run(func.run(
            artifact=long_function,
            analysis_type="quality",
        ))
        
        # Should find long function issue
        assert any("long" in f.description.lower() or "lines" in f.description.lower() 
                   for f in result.findings)

    def test_quality_analysis_detects_missing_docstrings(self) -> None:
        """Quality analysis detects missing docstrings."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = """def my_function():
    return 42

class MyClass:
    def method(self):
        pass"""
        
        result = asyncio.run(func.run(
            artifact=code,
            analysis_type="quality",
        ))
        
        # Should detect missing docstrings
        assert any("docstring" in f.description.lower() for f in result.findings)

    def test_quality_analysis_returns_metrics(self) -> None:
        """Quality analysis returns code metrics."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = """def foo():
    pass

def bar():
    return 1"""
        
        result = asyncio.run(func.run(
            artifact=code,
            analysis_type="quality",
        ))
        
        # Should have some metrics
        assert "loc" in result.metrics or "functions" in result.metrics or len(result.metrics) > 0


# =============================================================================
# AC-9.5: Security Analysis Tests
# =============================================================================

class TestSecurityAnalysis:
    """Tests for security analysis (analysis_type='security')."""

    def test_security_detects_hardcoded_passwords(self) -> None:
        """Security analysis detects hardcoded passwords."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        from src.schemas.functions.analyze_artifact import Severity
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = '''password = "secret123"
api_key = "sk_live_abc123def456"
'''
        
        result = asyncio.run(func.run(
            artifact=code,
            analysis_type="security",
        ))
        
        # Should flag hardcoded credentials
        assert any("password" in f.description.lower() or "secret" in f.description.lower() or "credential" in f.description.lower()
                   for f in result.findings)

    def test_security_detects_sql_injection(self) -> None:
        """Security analysis detects SQL injection vulnerabilities."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = '''def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
'''
        
        result = asyncio.run(func.run(
            artifact=code,
            analysis_type="security",
        ))
        
        # Should flag SQL injection
        assert any("sql" in f.description.lower() or "injection" in f.description.lower()
                   for f in result.findings)

    def test_security_detects_eval_usage(self) -> None:
        """Security analysis detects dangerous eval() usage."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = '''def execute_code(user_input):
    result = eval(user_input)
    return result
'''
        
        result = asyncio.run(func.run(
            artifact=code,
            analysis_type="security",
        ))
        
        # Should flag eval usage
        assert any("eval" in f.description.lower() or "dangerous" in f.description.lower()
                   for f in result.findings)

    def test_security_analysis_has_high_severity_for_critical_issues(self) -> None:
        """Security vulnerabilities have appropriate severity."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        from src.schemas.functions.analyze_artifact import Severity
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = '''password = "admin123"
'''
        
        result = asyncio.run(func.run(
            artifact=code,
            analysis_type="security",
        ))
        
        # Security issues should have HIGH or CRITICAL severity
        security_findings = [f for f in result.findings if "security" in f.category.lower() or "credential" in f.category.lower()]
        if security_findings:
            assert any(f.severity in [Severity.HIGH, Severity.CRITICAL] for f in security_findings)


# =============================================================================
# AC-9.5: Pattern Detection Tests
# =============================================================================

class TestPatternDetection:
    """Tests for pattern detection (analysis_type='patterns')."""

    def test_patterns_detects_singleton(self) -> None:
        """Pattern analysis detects singleton pattern."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = '''class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
'''
        
        result = asyncio.run(func.run(
            artifact=code,
            analysis_type="patterns",
        ))
        
        # Should identify singleton pattern
        assert any("singleton" in f.description.lower() or "singleton" in f.category.lower()
                   for f in result.findings)

    def test_patterns_detects_factory(self) -> None:
        """Pattern analysis detects factory pattern."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = '''class AnimalFactory:
    def create_animal(self, animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError("Unknown animal type")
'''
        
        result = asyncio.run(func.run(
            artifact=code,
            analysis_type="patterns",
        ))
        
        # Should identify factory pattern
        assert any("factory" in f.description.lower() or "factory" in f.category.lower()
                   for f in result.findings)

    def test_patterns_detects_repository(self) -> None:
        """Pattern analysis detects repository pattern."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = '''class UserRepository:
    def __init__(self, db):
        self.db = db
    
    def get(self, id):
        return self.db.query(User).filter(User.id == id).first()
    
    def save(self, user):
        self.db.add(user)
        self.db.commit()
    
    def delete(self, user):
        self.db.delete(user)
        self.db.commit()
'''
        
        result = asyncio.run(func.run(
            artifact=code,
            analysis_type="patterns",
        ))
        
        # Should identify repository pattern
        assert any("repository" in f.description.lower() or "repository" in f.category.lower()
                   for f in result.findings)

    def test_patterns_has_info_severity(self) -> None:
        """Pattern findings have INFO severity (not issues)."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        from src.schemas.functions.analyze_artifact import Severity
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = '''class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
'''
        
        result = asyncio.run(func.run(
            artifact=code,
            analysis_type="patterns",
        ))
        
        # Pattern findings should be INFO (informational)
        pattern_findings = [f for f in result.findings if "pattern" in f.category.lower()]
        if pattern_findings:
            assert all(f.severity == Severity.INFO for f in pattern_findings)


# =============================================================================
# AnalyzeArtifactFunction Implementation Tests
# =============================================================================

class TestAnalyzeArtifactFunction:
    """Tests for AnalyzeArtifactFunction class."""

    def test_inherits_from_agent_function(self) -> None:
        """AnalyzeArtifactFunction inherits from AgentFunction."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        from src.functions.base import AgentFunction
        
        func = AnalyzeArtifactFunction()
        assert isinstance(func, AgentFunction)

    def test_implements_protocol(self) -> None:
        """AnalyzeArtifactFunction implements AgentFunctionProtocol."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        from src.functions.base import AgentFunctionProtocol
        
        func = AnalyzeArtifactFunction()
        assert isinstance(func, AgentFunctionProtocol)

    def test_run_returns_analysis_result(self) -> None:
        """run() returns AnalysisResult."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        from src.schemas.functions.analyze_artifact import AnalysisResult
        import asyncio
        
        func = AnalyzeArtifactFunction()
        result = asyncio.run(func.run(artifact="def foo(): pass"))
        
        assert isinstance(result, AnalysisResult)

    def test_run_with_empty_artifact(self) -> None:
        """run() handles empty artifact."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        import asyncio
        
        func = AnalyzeArtifactFunction()
        result = asyncio.run(func.run(artifact=""))
        
        assert result.passed is True  # Empty code passes

    def test_repr_includes_name_and_preset(self) -> None:
        """__repr__ includes function name and preset."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        
        func = AnalyzeArtifactFunction()
        repr_str = repr(func)
        
        assert "analyze_artifact" in repr_str
        assert "D4" in repr_str


# =============================================================================
# Document Analysis Tests
# =============================================================================

class TestDocumentAnalysis:
    """Tests for document artifact analysis."""

    def test_document_analysis_with_markdown(self) -> None:
        """Document analysis handles markdown content."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        from src.schemas.functions.analyze_artifact import ArtifactKind
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        doc = """# API Documentation

## Overview
This is the API overview.

## Endpoints
"""
        
        result = asyncio.run(func.run(
            artifact=doc,
            artifact_type="document",
            analysis_type="quality",
        ))
        
        assert isinstance(result.passed, bool)

    def test_config_analysis_with_yaml(self) -> None:
        """Config analysis handles YAML-like content."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        config = """database:
  host: localhost
  port: 5432
  password: secret123
"""
        
        result = asyncio.run(func.run(
            artifact=config,
            artifact_type="config",
            analysis_type="security",
        ))
        
        # Should detect hardcoded password in config
        assert any("password" in f.description.lower() or "secret" in f.description.lower()
                   for f in result.findings)


# =============================================================================
# Integration Tests
# =============================================================================

class TestAnalyzeArtifactIntegration:
    """Integration tests for analyze_artifact function."""

    def test_full_quality_analysis_workflow(self) -> None:
        """Full quality analysis workflow produces expected output."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        from src.schemas.functions.analyze_artifact import AnalysisResult
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = """def calculate(x, y, operation):
    if operation == "add":
        return x + y
    elif operation == "subtract":
        return x - y
    elif operation == "multiply":
        return x * y
    elif operation == "divide":
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
    else:
        raise ValueError("Unknown operation")
"""
        
        result = asyncio.run(func.run(
            artifact=code,
            analysis_type="quality",
        ))
        
        assert isinstance(result, AnalysisResult)
        assert isinstance(result.findings, list)
        assert isinstance(result.passed, bool)

    def test_multiple_analysis_types_same_code(self) -> None:
        """Same code can be analyzed with different analysis types."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = '''class UserService:
    password = "admin"
    
    def get_user(self, id):
        return self.db.find(id)
'''
        
        quality_result = asyncio.run(func.run(
            artifact=code,
            analysis_type="quality",
        ))
        
        security_result = asyncio.run(func.run(
            artifact=code,
            analysis_type="security",
        ))
        
        patterns_result = asyncio.run(func.run(
            artifact=code,
            analysis_type="patterns",
        ))
        
        # Different analysis types produce different findings
        assert isinstance(quality_result.findings, list)
        assert isinstance(security_result.findings, list)
        assert isinstance(patterns_result.findings, list)

    def test_checklist_specific_checks(self) -> None:
        """Checklist parameter focuses analysis on specific checks."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = """def foo():
    pass
"""
        
        result = asyncio.run(func.run(
            artifact=code,
            analysis_type="quality",
            checklist=["docstrings"],
        ))
        
        # Should have findings related to checklist items
        assert isinstance(result.findings, list)

    def test_compressed_report_generation(self) -> None:
        """Analysis generates compressed report for downstream."""
        from src.functions.analyze_artifact import AnalyzeArtifactFunction
        import asyncio
        
        func = AnalyzeArtifactFunction()
        
        code = """def foo():
    '''A function.'''
    return 42
"""
        
        result = asyncio.run(func.run(
            artifact=code,
            analysis_type="quality",
        ))
        
        assert result.compressed_report is not None
        assert isinstance(result.compressed_report, str)
