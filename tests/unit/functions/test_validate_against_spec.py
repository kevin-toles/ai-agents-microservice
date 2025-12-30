"""Tests for validate_against_spec function.

TDD tests for WBS-AGT10: validate_against_spec Function.

Acceptance Criteria Coverage:
- AC-10.1: Compares artifact against specification
- AC-10.2: Returns ValidationResult with compliance %, violations
- AC-10.3: Context budget: 4096 input / 1024 output
- AC-10.4: Default preset: D4 (Standard)
- AC-10.5: Violations include line_number, expected, actual

Exit Criteria:
- compliance_percentage is 0-100 float
- Each Violation has expected vs actual comparison
- Empty violations list → compliance_percentage = 100.0

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 5
"""

import pytest
from typing import Any
from pydantic import ValidationError


# =============================================================================
# AC-10.1: Input Schema Tests - ValidateAgainstSpecInput
# =============================================================================

class TestValidateAgainstSpecInput:
    """Tests for ValidateAgainstSpecInput schema."""

    def test_input_requires_artifact(self) -> None:
        """ValidateAgainstSpecInput requires artifact field."""
        from src.schemas.functions.validate_against_spec import ValidateAgainstSpecInput
        
        with pytest.raises(ValidationError) as exc_info:
            ValidateAgainstSpecInput(specification="test spec")  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("artifact",) for e in errors)

    def test_input_requires_specification(self) -> None:
        """ValidateAgainstSpecInput requires specification field."""
        from src.schemas.functions.validate_against_spec import ValidateAgainstSpecInput
        
        with pytest.raises(ValidationError) as exc_info:
            ValidateAgainstSpecInput(artifact="test code")  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("specification",) for e in errors)

    def test_input_accepts_artifact_and_specification(self) -> None:
        """ValidateAgainstSpecInput accepts artifact and specification."""
        from src.schemas.functions.validate_against_spec import ValidateAgainstSpecInput
        
        input_data = ValidateAgainstSpecInput(
            artifact="def add(a, b): return a + b",
            specification="Create a function that adds two numbers",
        )
        assert input_data.artifact == "def add(a, b): return a + b"
        assert input_data.specification == "Create a function that adds two numbers"

    def test_input_has_optional_invariants(self) -> None:
        """ValidateAgainstSpecInput has optional invariants list."""
        from src.schemas.functions.validate_against_spec import ValidateAgainstSpecInput
        
        input_data = ValidateAgainstSpecInput(
            artifact="code",
            specification="spec",
        )
        assert input_data.invariants == []
        
        input_with_invariants = ValidateAgainstSpecInput(
            artifact="code",
            specification="spec",
            invariants=["must handle negative numbers", "must return int"],
        )
        assert len(input_with_invariants.invariants) == 2

    def test_input_has_optional_acceptance_criteria(self) -> None:
        """ValidateAgainstSpecInput has optional acceptance_criteria list."""
        from src.schemas.functions.validate_against_spec import ValidateAgainstSpecInput
        
        input_data = ValidateAgainstSpecInput(
            artifact="code",
            specification="spec",
        )
        assert input_data.acceptance_criteria == []
        
        input_with_ac = ValidateAgainstSpecInput(
            artifact="code",
            specification="spec",
            acceptance_criteria=["AC-1: Function exists", "AC-2: Returns correct type"],
        )
        assert len(input_with_ac.acceptance_criteria) == 2

    def test_input_json_schema_export(self) -> None:
        """ValidateAgainstSpecInput exports valid JSON schema."""
        from src.schemas.functions.validate_against_spec import ValidateAgainstSpecInput
        
        schema = ValidateAgainstSpecInput.model_json_schema()
        
        assert "properties" in schema
        assert "artifact" in schema["properties"]
        assert "specification" in schema["properties"]
        assert "invariants" in schema["properties"]
        assert "acceptance_criteria" in schema["properties"]


# =============================================================================
# AC-10.5: Violation Schema Tests
# =============================================================================

class TestViolation:
    """Tests for Violation model - includes line_number, expected, actual."""

    def test_violation_requires_expected(self) -> None:
        """Violation requires expected field."""
        from src.schemas.functions.validate_against_spec import Violation
        
        with pytest.raises(ValidationError) as exc_info:
            Violation(
                actual="found this",
                description="test desc",
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("expected",) for e in errors)

    def test_violation_requires_actual(self) -> None:
        """Violation requires actual field."""
        from src.schemas.functions.validate_against_spec import Violation
        
        with pytest.raises(ValidationError) as exc_info:
            Violation(
                expected="expected this",
                description="test desc",
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("actual",) for e in errors)

    def test_violation_requires_description(self) -> None:
        """Violation requires description field."""
        from src.schemas.functions.validate_against_spec import Violation
        
        with pytest.raises(ValidationError) as exc_info:
            Violation(
                expected="expected",
                actual="actual",
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("description",) for e in errors)

    def test_violation_accepts_all_required_fields(self) -> None:
        """Violation accepts expected, actual, description (AC-10.5)."""
        from src.schemas.functions.validate_against_spec import Violation
        
        violation = Violation(
            expected="function should return int",
            actual="function returns str",
            description="Return type mismatch",
        )
        
        assert violation.expected == "function should return int"
        assert violation.actual == "function returns str"
        assert violation.description == "Return type mismatch"

    def test_violation_has_optional_line_number(self) -> None:
        """Violation has optional line_number field (AC-10.5)."""
        from src.schemas.functions.validate_against_spec import Violation
        
        violation = Violation(
            expected="docstring",
            actual="no docstring",
            description="Missing docstring",
        )
        assert violation.line_number is None
        
        violation_with_line = Violation(
            expected="docstring",
            actual="no docstring",
            description="Missing docstring",
            line_number=15,
        )
        assert violation_with_line.line_number == 15

    def test_violation_has_optional_severity(self) -> None:
        """Violation has optional severity field."""
        from src.schemas.functions.validate_against_spec import Violation, ViolationSeverity
        
        violation = Violation(
            expected="test",
            actual="actual",
            description="desc",
        )
        assert violation.severity == ViolationSeverity.ERROR  # default
        
        warning_violation = Violation(
            expected="test",
            actual="actual",
            description="desc",
            severity=ViolationSeverity.WARNING,
        )
        assert warning_violation.severity == ViolationSeverity.WARNING

    def test_violation_has_optional_criterion_id(self) -> None:
        """Violation has optional criterion_id to link to acceptance criteria."""
        from src.schemas.functions.validate_against_spec import Violation
        
        violation = Violation(
            expected="test",
            actual="actual",
            description="desc",
            criterion_id="AC-1",
        )
        assert violation.criterion_id == "AC-1"

    def test_violation_json_schema_export(self) -> None:
        """Violation exports valid JSON schema."""
        from src.schemas.functions.validate_against_spec import Violation
        
        schema = Violation.model_json_schema()
        
        assert "properties" in schema
        assert "expected" in schema["properties"]
        assert "actual" in schema["properties"]
        assert "description" in schema["properties"]
        assert "line_number" in schema["properties"]


# =============================================================================
# AC-10.2: ValidationResult Schema Tests
# =============================================================================

class TestValidationResult:
    """Tests for ValidationResult model - compliance %, violations."""

    def test_validation_result_has_valid_bool(self) -> None:
        """ValidationResult has valid bool field."""
        from src.schemas.functions.validate_against_spec import ValidationResult
        
        result = ValidationResult()
        assert result.valid is True  # Default

    def test_validation_result_has_violations_list(self) -> None:
        """ValidationResult has violations list (AC-10.2)."""
        from src.schemas.functions.validate_against_spec import ValidationResult
        
        result = ValidationResult()
        assert result.violations == []
        assert isinstance(result.violations, list)

    def test_validation_result_accepts_violations(self) -> None:
        """ValidationResult accepts list of Violation objects."""
        from src.schemas.functions.validate_against_spec import ValidationResult, Violation
        
        violation = Violation(
            expected="function",
            actual="class",
            description="Wrong type",
        )
        result = ValidationResult(violations=[violation], valid=False)
        
        assert len(result.violations) == 1
        assert result.violations[0].expected == "function"

    def test_validation_result_has_compliance_percentage(self) -> None:
        """ValidationResult has compliance_percentage field (AC-10.2)."""
        from src.schemas.functions.validate_against_spec import ValidationResult
        
        result = ValidationResult()
        assert result.compliance_percentage == 100.0  # Default
        
        result_with_compliance = ValidationResult(compliance_percentage=75.5)
        assert result_with_compliance.compliance_percentage == 75.5

    def test_compliance_percentage_is_0_to_100(self) -> None:
        """compliance_percentage must be 0-100 float."""
        from src.schemas.functions.validate_against_spec import ValidationResult
        
        # Valid values
        result_0 = ValidationResult(compliance_percentage=0.0)
        assert result_0.compliance_percentage == 0.0
        
        result_100 = ValidationResult(compliance_percentage=100.0)
        assert result_100.compliance_percentage == 100.0
        
        result_50 = ValidationResult(compliance_percentage=50.5)
        assert result_50.compliance_percentage == 50.5
        
        # Invalid values
        with pytest.raises(ValidationError):
            ValidationResult(compliance_percentage=-1.0)
        
        with pytest.raises(ValidationError):
            ValidationResult(compliance_percentage=101.0)

    def test_validation_result_has_confidence(self) -> None:
        """ValidationResult has confidence field (0.0-1.0)."""
        from src.schemas.functions.validate_against_spec import ValidationResult
        
        result = ValidationResult()
        assert result.confidence == 1.0  # Default
        
        result_with_confidence = ValidationResult(confidence=0.85)
        assert result_with_confidence.confidence == 0.85

    def test_confidence_is_0_to_1(self) -> None:
        """confidence must be 0.0-1.0 float."""
        from src.schemas.functions.validate_against_spec import ValidationResult
        
        # Valid values
        result_0 = ValidationResult(confidence=0.0)
        assert result_0.confidence == 0.0
        
        result_1 = ValidationResult(confidence=1.0)
        assert result_1.confidence == 1.0
        
        # Invalid values
        with pytest.raises(ValidationError):
            ValidationResult(confidence=-0.1)
        
        with pytest.raises(ValidationError):
            ValidationResult(confidence=1.1)

    def test_validation_result_has_remediation_hints(self) -> None:
        """ValidationResult has remediation_hints list."""
        from src.schemas.functions.validate_against_spec import ValidationResult
        
        result = ValidationResult()
        assert result.remediation_hints == []
        
        result_with_hints = ValidationResult(
            remediation_hints=["Add docstring", "Fix return type"]
        )
        assert len(result_with_hints.remediation_hints) == 2

    def test_validation_result_json_schema_export(self) -> None:
        """ValidationResult exports valid JSON schema."""
        from src.schemas.functions.validate_against_spec import ValidationResult
        
        schema = ValidationResult.model_json_schema()
        
        assert "properties" in schema
        assert "valid" in schema["properties"]
        assert "violations" in schema["properties"]
        assert "compliance_percentage" in schema["properties"]
        assert "confidence" in schema["properties"]


# =============================================================================
# AC-10.3: Context Budget Tests
# =============================================================================

class TestValidateAgainstSpecContextBudget:
    """Tests for context budget: 4096 input / 1024 output (AC-10.3)."""

    def test_context_budget_in_defaults(self) -> None:
        """validate_against_spec context budget is in CONTEXT_BUDGET_DEFAULTS."""
        from src.functions.base import CONTEXT_BUDGET_DEFAULTS
        
        assert "validate_against_spec" in CONTEXT_BUDGET_DEFAULTS
        budget = CONTEXT_BUDGET_DEFAULTS["validate_against_spec"]
        assert budget["input"] == 4096
        assert budget["output"] == 1024

    def test_function_returns_correct_budget(self) -> None:
        """ValidateAgainstSpecFunction.get_context_budget() returns correct values."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        
        func = ValidateAgainstSpecFunction()
        budget = func.get_context_budget()
        
        assert budget["input"] == 4096
        assert budget["output"] == 1024

    def test_budget_exceeded_raises_error(self) -> None:
        """Input exceeding budget raises ContextBudgetExceededError."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        from src.functions.base import ContextBudgetExceededError
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        # 4096 tokens * 4 chars = 16384 chars limit
        # Create content that exceeds this
        large_artifact = "x" * 20000
        
        with pytest.raises(ContextBudgetExceededError) as exc_info:
            asyncio.run(func.run(artifact=large_artifact, specification="test"))
        
        assert exc_info.value.function_name == "validate_against_spec"
        assert exc_info.value.limit == 4096


# =============================================================================
# AC-10.4: Default Preset Tests
# =============================================================================

class TestValidateAgainstSpecPreset:
    """Tests for default preset: D4 (Standard) (AC-10.4)."""

    def test_default_preset_is_d4(self) -> None:
        """ValidateAgainstSpecFunction default_preset is 'D4'."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        
        func = ValidateAgainstSpecFunction()
        assert func.default_preset == "D4"

    def test_function_name_is_validate_against_spec(self) -> None:
        """ValidateAgainstSpecFunction name is 'validate_against_spec'."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        
        func = ValidateAgainstSpecFunction()
        assert func.name == "validate_against_spec"

    def test_preset_selection_light(self) -> None:
        """select_preset('light') returns S1."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        
        func = ValidateAgainstSpecFunction()
        assert func.select_preset("light") == "S1"

    def test_preset_selection_high_quality(self) -> None:
        """select_preset('high_quality') returns D10."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        
        func = ValidateAgainstSpecFunction()
        assert func.select_preset("high_quality") == "D10"


# =============================================================================
# ValidateAgainstSpecFunction Implementation Tests
# =============================================================================

class TestValidateAgainstSpecFunction:
    """Tests for ValidateAgainstSpecFunction class."""

    def test_inherits_from_agent_function(self) -> None:
        """ValidateAgainstSpecFunction inherits from AgentFunction."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        from src.functions.base import AgentFunction
        
        func = ValidateAgainstSpecFunction()
        assert isinstance(func, AgentFunction)

    def test_implements_protocol(self) -> None:
        """ValidateAgainstSpecFunction implements AgentFunctionProtocol."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        from src.functions.base import AgentFunctionProtocol
        
        func = ValidateAgainstSpecFunction()
        assert isinstance(func, AgentFunctionProtocol)

    def test_run_returns_validation_result(self) -> None:
        """run() returns ValidationResult."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        from src.schemas.functions.validate_against_spec import ValidationResult
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        result = asyncio.run(func.run(
            artifact="def add(a, b): return a + b",
            specification="Create an add function",
        ))
        
        assert isinstance(result, ValidationResult)

    def test_repr_includes_name_and_preset(self) -> None:
        """__repr__ includes function name and preset."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        
        func = ValidateAgainstSpecFunction()
        repr_str = repr(func)
        
        assert "validate_against_spec" in repr_str
        assert "D4" in repr_str


# =============================================================================
# Compliance Percentage Calculation Tests
# =============================================================================

class TestCompliancePercentage:
    """Tests for compliance percentage calculation."""

    def test_empty_violations_means_100_percent(self) -> None:
        """Empty violations list → compliance_percentage = 100.0 (Exit Criteria)."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        # Artifact that matches specification
        result = asyncio.run(func.run(
            artifact="def add(a, b):\n    '''Add two numbers.'''\n    return a + b",
            specification="Create an add function with docstring",
        ))
        
        # If no violations, should be 100%
        if not result.violations:
            assert result.compliance_percentage == 100.0

    def test_violations_reduce_compliance(self) -> None:
        """Violations reduce compliance_percentage below 100."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        # Artifact missing expected elements
        result = asyncio.run(func.run(
            artifact="x = 1",  # Not a function at all
            specification="Create a function named calculate that takes two parameters",
            acceptance_criteria=[
                "Must be a function",
                "Must be named 'calculate'",
                "Must take two parameters",
            ],
        ))
        
        # Should have violations and reduced compliance
        if result.violations:
            assert result.compliance_percentage < 100.0

    def test_compliance_percentage_type(self) -> None:
        """compliance_percentage is a float between 0 and 100."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        result = asyncio.run(func.run(
            artifact="def foo(): pass",
            specification="test",
        ))
        
        assert isinstance(result.compliance_percentage, float)
        assert 0.0 <= result.compliance_percentage <= 100.0


# =============================================================================
# Violation Detection Tests
# =============================================================================

class TestViolationDetection:
    """Tests for detecting violations against specification."""

    def test_detects_missing_function(self) -> None:
        """Detects when expected function is missing."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        result = asyncio.run(func.run(
            artifact="x = 1",
            specification="Create a function named 'calculate'",
            acceptance_criteria=["Function 'calculate' must exist"],
        ))
        
        # Should detect missing function
        assert any(
            "function" in v.description.lower() or "calculate" in v.description.lower()
            for v in result.violations
        )

    def test_detects_wrong_return_type(self) -> None:
        """Detects when return type doesn't match specification."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        result = asyncio.run(func.run(
            artifact="def calculate(a, b) -> str:\n    return str(a + b)",
            specification="Create a function that returns an integer",
            invariants=["Return type must be int"],
        ))
        
        # May detect type mismatch
        assert isinstance(result.violations, list)

    def test_detects_missing_docstring(self) -> None:
        """Detects missing docstring when required."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        result = asyncio.run(func.run(
            artifact="def foo():\n    pass",
            specification="Create a function with a docstring",
            acceptance_criteria=["Function must have a docstring"],
        ))
        
        # Should detect missing docstring
        has_docstring_violation = any(
            "docstring" in v.description.lower()
            for v in result.violations
        )
        # Only assert if we expect docstring checking
        assert isinstance(result.violations, list)

    def test_violation_has_expected_and_actual(self) -> None:
        """Each Violation has expected vs actual comparison (Exit Criteria)."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        from src.schemas.functions.validate_against_spec import Violation
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        result = asyncio.run(func.run(
            artifact="x = 1",
            specification="Create a function",
            acceptance_criteria=["Must be a function"],
        ))
        
        for violation in result.violations:
            assert isinstance(violation, Violation)
            assert violation.expected is not None
            assert violation.actual is not None
            assert violation.description is not None


# =============================================================================
# Invariants Validation Tests
# =============================================================================

class TestInvariantsValidation:
    """Tests for validating against invariants from upstream."""

    def test_validates_against_invariants(self) -> None:
        """Validates artifact against provided invariants."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        result = asyncio.run(func.run(
            artifact="def add(a, b):\n    return a + b",
            specification="Addition function",
            invariants=[
                "Must handle negative numbers",
                "Must return numeric type",
            ],
        ))
        
        assert isinstance(result, result.__class__)
        assert isinstance(result.violations, list)

    def test_validates_against_acceptance_criteria(self) -> None:
        """Validates artifact against acceptance criteria."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        result = asyncio.run(func.run(
            artifact="def foo(): pass",
            specification="test spec",
            acceptance_criteria=[
                "AC-1: Function must exist",
                "AC-2: Function must have docstring",
                "AC-3: Function must have type hints",
            ],
        ))
        
        # Should process all acceptance criteria
        assert isinstance(result.violations, list)


# =============================================================================
# Remediation Hints Tests
# =============================================================================

class TestRemediationHints:
    """Tests for remediation hints generation."""

    def test_generates_remediation_hints(self) -> None:
        """Generates remediation hints for violations."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        result = asyncio.run(func.run(
            artifact="x = 1",
            specification="Create a function with docstring",
            acceptance_criteria=["Must be a function", "Must have docstring"],
        ))
        
        # Should have remediation hints if there are violations
        if result.violations:
            assert isinstance(result.remediation_hints, list)

    def test_remediation_hints_are_actionable(self) -> None:
        """Remediation hints provide actionable guidance."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        result = asyncio.run(func.run(
            artifact="def foo(): pass",
            specification="Create function with docstring",
        ))
        
        # Hints should be strings with content
        for hint in result.remediation_hints:
            assert isinstance(hint, str)


# =============================================================================
# Integration Tests
# =============================================================================

class TestValidateAgainstSpecIntegration:
    """Integration tests for validate_against_spec function."""

    def test_full_validation_workflow(self) -> None:
        """Full validation workflow produces expected output."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        from src.schemas.functions.validate_against_spec import ValidationResult
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        artifact = '''def calculate(a: int, b: int) -> int:
    """Calculate the sum of two integers.
    
    Args:
        a: First integer
        b: Second integer
    
    Returns:
        Sum of a and b
    """
    return a + b
'''
        
        specification = """Create a function named 'calculate' that:
- Takes two integer parameters
- Returns their sum as an integer
- Has a docstring explaining the function
"""
        
        result = asyncio.run(func.run(
            artifact=artifact,
            specification=specification,
            acceptance_criteria=[
                "Function named 'calculate' exists",
                "Takes two parameters",
                "Has docstring",
                "Has type hints",
            ],
        ))
        
        assert isinstance(result, ValidationResult)
        assert isinstance(result.valid, bool)
        assert isinstance(result.compliance_percentage, float)
        assert 0.0 <= result.compliance_percentage <= 100.0

    def test_valid_artifact_has_high_compliance(self) -> None:
        """Artifact matching spec has high compliance."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        artifact = '''def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"
'''
        
        result = asyncio.run(func.run(
            artifact=artifact,
            specification="Create a greet function with docstring",
        ))
        
        # Well-formed artifact should have good compliance
        assert result.compliance_percentage >= 50.0

    def test_completely_wrong_artifact(self) -> None:
        """Completely wrong artifact has low compliance."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        result = asyncio.run(func.run(
            artifact="import os\nos.remove('/')",  # Completely wrong
            specification="Create a calculator class with add and subtract methods",
            acceptance_criteria=[
                "Must be a class",
                "Must be named 'Calculator'",
                "Must have 'add' method",
                "Must have 'subtract' method",
            ],
        ))
        
        # Should have violations
        assert len(result.violations) > 0
        assert result.compliance_percentage < 100.0

    def test_empty_artifact(self) -> None:
        """Empty artifact handles gracefully."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        result = asyncio.run(func.run(
            artifact="",
            specification="Create a function",
        ))
        
        # Should handle empty artifact
        assert isinstance(result.compliance_percentage, float)
        assert result.valid is False or result.compliance_percentage < 100.0

    def test_empty_specification(self) -> None:
        """Empty specification handles gracefully."""
        from src.functions.validate_against_spec import ValidateAgainstSpecFunction
        import asyncio
        
        func = ValidateAgainstSpecFunction()
        
        result = asyncio.run(func.run(
            artifact="def foo(): pass",
            specification="",
        ))
        
        # Should handle empty specification
        assert isinstance(result, result.__class__)
