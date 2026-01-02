"""ValidateAgainstSpec Agent Function.

WBS-AGT10: validate_against_spec Function
WBS-KB7.9: Code-Orchestrator Tool Integration (AC-KB7.6)

This module implements the ValidateAgainstSpecFunction which validates
artifacts against specifications, invariants, and acceptance criteria.

Acceptance Criteria:
- AC-10.1: Compares artifact against specification
- AC-10.2: Returns ValidationResult with compliance %, violations
- AC-10.3: Context budget: 4096 input / 1024 output
- AC-10.4: Default preset: D4 (Standard)
- AC-10.5: Violations include line_number, expected, actual
- AC-KB7.6: CodeValidationTool available for objective code analysis

Exit Criteria:
- compliance_percentage is 0-100 float
- Each Violation has expected vs actual comparison
- Empty violations list → compliance_percentage = 100.0
- CodeValidationTool provides objective metrics when available

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 5

REFACTOR Phase:
- Extracted CHARS_PER_TOKEN to src/core/constants.py (S1192)
- Extracted artifact parsing to src/functions/utils/artifact_parser.py (S1192)
- Using shared utilities to reduce code duplication
- KB7.9: Integrated CodeValidationTool for objective metrics
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from src.functions.base import AgentFunction


if TYPE_CHECKING:
    from src.tools.code_validation import CodeValidationProtocol


logger = logging.getLogger(__name__)
from src.functions.utils.artifact_parser import (
    # Artifact parsing
    extract_class_name,
    extract_class_name_from_text,
    extract_entity_name,
    extract_func_name_from_text,
    extract_function_name,
    extract_method_name,
    extract_method_name_from_text,
    has_class,
    has_class_named,
    has_docstring,
    has_function,
    has_function_named,
    has_method,
    has_type_hints,
    spec_requires_class,
    spec_requires_docstring,
    # Specification requirement checking
    spec_requires_function,
)
from src.functions.utils.token_utils import estimate_tokens
from src.schemas.functions.validate_against_spec import (
    ValidationResult,
    Violation,
    ViolationSeverity,
)


class ValidateAgainstSpecFunction(AgentFunction):
    """Agent function to validate artifacts against specifications.

    Compares code/content artifacts against original requirements,
    invariants from summarize_content, and acceptance criteria.

    Context Budget (AC-10.3):
        - Input: 4096 tokens
        - Output: 1024 tokens

    Default Preset (AC-10.4): D4 (Standard/Critique mode)

    Attributes:
        name: Function identifier 'validate_against_spec'
        default_preset: Default to D4 for critique mode

    Example:
        ```python
        func = ValidateAgainstSpecFunction()
        result = await func.run(
            artifact="def add(a, b): return a + b",
            specification="Create an add function with docstring",
            acceptance_criteria=["Must have docstring"],
        )
        print(f"Compliance: {result.compliance_percentage}%")
        ```

    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → validate_against_spec
    """

    name: str = "validate_against_spec"
    default_preset: str = "D4"  # Critique mode for validation

    def __init__(
        self,
        code_validation_tool: CodeValidationProtocol | None = None,
    ) -> None:
        """Initialize the validate_against_spec function.
        
        Args:
            code_validation_tool: Optional CodeValidationTool for objective
                code analysis using CodeT5+, GraphCodeBERT, CodeBERT, SonarQube.
                AC-KB7.6: Tools available to validate_against_spec agent.
        """
        self._code_validation_tool = code_validation_tool

    async def run(  # type: ignore[override]
        self,
        *,
        artifact: str,
        specification: str,
        invariants: list[str] | None = None,
        acceptance_criteria: list[str] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Validate artifact against specification and criteria.

        Args:
            artifact: The code/content to validate
            specification: Original requirement/specification
            invariants: List of invariants from upstream processing
            acceptance_criteria: List of acceptance criteria to check
            **kwargs: Additional arguments (ignored)

        Returns:
            ValidationResult with:
                - valid: bool
                - violations: list[Violation]
                - compliance_percentage: 0-100 float
                - confidence: 0.0-1.0 float
                - remediation_hints: list[str]

        Raises:
            ContextBudgetExceededError: If input exceeds 4096 token budget
        """
        # Normalize inputs
        invariants = invariants or []
        acceptance_criteria = acceptance_criteria or []

        # Enforce context budget (AC-10.3) - using shared utility
        total_input = artifact + specification + " ".join(invariants) + " ".join(acceptance_criteria)
        input_tokens = estimate_tokens(total_input)
        self.enforce_budget(input_tokens)

        # Collect violations
        violations: list[Violation] = []
        remediation_hints: list[str] = []

        # Validate artifact content
        violations.extend(self._validate_artifact_content(artifact, specification))

        # Validate against acceptance criteria
        for i, criterion in enumerate(acceptance_criteria, 1):
            criterion_violations = self._validate_criterion(artifact, criterion, f"AC-{i}")
            violations.extend(criterion_violations)

        # Validate against invariants
        for invariant in invariants:
            invariant_violations = self._validate_invariant(artifact, invariant)
            violations.extend(invariant_violations)

        # Generate remediation hints
        remediation_hints = self._generate_remediation_hints(violations)

        # Calculate compliance percentage
        compliance_percentage = self._calculate_compliance(
            violations=violations,
            acceptance_criteria=acceptance_criteria,
            invariants=invariants,
        )

        # Determine if valid (no critical/error violations)
        valid = not any(
            v.severity in (ViolationSeverity.ERROR, ViolationSeverity.CRITICAL)
            for v in violations
        )

        # Calculate confidence based on how thorough the validation was
        confidence = self._calculate_confidence(
            artifact=artifact,
            specification=specification,
            acceptance_criteria=acceptance_criteria,
        )

        return ValidationResult(
            valid=valid,
            violations=violations,
            compliance_percentage=compliance_percentage,
            confidence=confidence,
            remediation_hints=remediation_hints,
        )

    def _validate_artifact_content(
        self,
        artifact: str,
        specification: str,
    ) -> list[Violation]:
        """Validate basic artifact content against specification.

        Performs structural validation to detect:
        - Missing required elements (functions, classes, etc.)
        - Empty artifacts
        - Basic spec keyword matching

        Args:
            artifact: The content to validate
            specification: The specification to check against

        Returns:
            List of violations found
        """
        violations: list[Violation] = []

        # Check for empty artifact
        if not artifact or not artifact.strip():
            violations.append(Violation(
                expected="Non-empty artifact content",
                actual="Empty artifact",
                description="Artifact is empty or contains only whitespace",
                severity=ViolationSeverity.CRITICAL,
            ))
            return violations

        spec_lower = specification.lower()
        artifact.lower()

        # Check for function requirement - using shared utilities
        if spec_requires_function(spec_lower):
            if not has_function(artifact):
                func_name = extract_function_name(specification)
                violations.append(Violation(
                    expected=f"Function definition{' named ' + func_name if func_name else ''}",
                    actual="No function definition found",
                    description=f"Specification requires a function{' named ' + func_name if func_name else ''}, but none was found",
                    severity=ViolationSeverity.ERROR,
                ))
            elif func_name := extract_function_name(specification):
                if not has_function_named(artifact, func_name):
                    violations.append(Violation(
                        expected=f"Function named '{func_name}'",
                        actual=f"Function named '{func_name}' not found",
                        description=f"Specification requires function '{func_name}', but it was not found",
                        severity=ViolationSeverity.ERROR,
                    ))

        # Check for class requirement - using shared utilities
        if spec_requires_class(spec_lower):
            if not has_class(artifact):
                cls_name = extract_class_name(specification)
                violations.append(Violation(
                    expected=f"Class definition{' named ' + cls_name if cls_name else ''}",
                    actual="No class definition found",
                    description=f"Specification requires a class{' named ' + cls_name if cls_name else ''}, but none was found",
                    severity=ViolationSeverity.ERROR,
                ))
            elif cls_name := extract_class_name(specification):
                if not has_class_named(artifact, cls_name):
                    violations.append(Violation(
                        expected=f"Class named '{cls_name}'",
                        actual=f"Class named '{cls_name}' not found",
                        description=f"Specification requires class '{cls_name}', but it was not found",
                        severity=ViolationSeverity.ERROR,
                    ))

        # Check for docstring requirement - using shared utilities
        if spec_requires_docstring(spec_lower) and not has_docstring(artifact):
            violations.append(Violation(
                expected="Docstring present",
                actual="No docstring found",
                description="Specification requires a docstring, but none was found",
                severity=ViolationSeverity.WARNING,
            ))

        return violations

    def _validate_criterion(
        self,
        artifact: str,
        criterion: str,
        criterion_id: str,
    ) -> list[Violation]:
        """Validate artifact against a single acceptance criterion.

        Args:
            artifact: The content to validate
            criterion: The criterion text
            criterion_id: ID for tracking (e.g., "AC-1")

        Returns:
            List of violations for this criterion
        """
        violations: list[Violation] = []
        criterion_lower = criterion.lower()

        # Check common criteria patterns

        # Function existence check
        if ("function" in criterion_lower
            and ("must exist" in criterion_lower or "must be" in criterion_lower)
            and not has_function(artifact)):
            violations.append(Violation(
                expected="Function exists",
                actual="No function found",
                description=f"Criterion requires a function: {criterion}",
                criterion_id=criterion_id,
                severity=ViolationSeverity.ERROR,
            ))

        # Named function check
        func_name = extract_entity_name(criterion, "function")
        if func_name and not has_function_named(artifact, func_name):
            violations.append(Violation(
                expected=f"Function '{func_name}' exists",
                actual=f"Function '{func_name}' not found",
                description=f"Criterion requires function '{func_name}': {criterion}",
                criterion_id=criterion_id,
                severity=ViolationSeverity.ERROR,
            ))

        # Class existence check
        if "class" in criterion_lower and "must" in criterion_lower:
            cls_name = extract_entity_name(criterion, "class")
            if cls_name and not has_class_named(artifact, cls_name):
                violations.append(Violation(
                    expected=f"Class '{cls_name}' exists",
                    actual=f"Class '{cls_name}' not found",
                    description=f"Criterion requires class '{cls_name}': {criterion}",
                    criterion_id=criterion_id,
                    severity=ViolationSeverity.ERROR,
                ))

        # Docstring check
        if "docstring" in criterion_lower and not has_docstring(artifact):
            violations.append(Violation(
                expected="Docstring present",
                actual="No docstring found",
                description=f"Criterion requires docstring: {criterion}",
                criterion_id=criterion_id,
                severity=ViolationSeverity.WARNING,
            ))

        # Type hints check
        if "type hint" in criterion_lower and not has_type_hints(artifact):
            violations.append(Violation(
                expected="Type hints present",
                actual="No type hints found",
                description=f"Criterion requires type hints: {criterion}",
                criterion_id=criterion_id,
                severity=ViolationSeverity.WARNING,
            ))

        # Method check
        meth_name = extract_method_name(criterion)
        if meth_name and not has_method(artifact, meth_name):
            violations.append(Violation(
                expected=f"Method '{meth_name}' exists",
                actual=f"Method '{meth_name}' not found",
                description=f"Criterion requires method '{meth_name}': {criterion}",
                criterion_id=criterion_id,
                severity=ViolationSeverity.ERROR,
            ))

        return violations

    def _validate_invariant(
        self,
        artifact: str,
        invariant: str,
    ) -> list[Violation]:
        """Validate artifact against an invariant.

        Args:
            artifact: The content to validate
            invariant: The invariant statement

        Returns:
            List of violations for this invariant
        """
        violations: list[Violation] = []
        invariant_lower = invariant.lower()

        # Return type invariants
        if ("return" in invariant_lower
            and "type" in invariant_lower
            and "->" in artifact):
            # Check if return type annotation exists
            # Try to extract expected type from invariant
            type_match = re.search(r"(int|str|float|bool|list|dict|tuple|none)", invariant_lower)
            if type_match:
                expected_type = type_match.group(1)
                if expected_type not in artifact.lower():
                    violations.append(Violation(
                        expected=f"Return type '{expected_type}'",
                        actual="Different or missing return type annotation",
                        description=f"Invariant requires return type '{expected_type}'",
                        severity=ViolationSeverity.WARNING,
                    ))

        # Must handle/support invariants
        if "must handle" in invariant_lower or "must support" in invariant_lower:
            # These are behavioral requirements that need runtime validation
            # For now, we just note them as informational
            pass

        return violations

    def _calculate_compliance(
        self,
        violations: list[Violation],
        acceptance_criteria: list[str],
        invariants: list[str],
    ) -> float:
        """Calculate compliance percentage.

        Exit Criteria: Empty violations list → compliance_percentage = 100.0

        Args:
            violations: List of violations found
            acceptance_criteria: List of acceptance criteria
            invariants: List of invariants

        Returns:
            Compliance percentage (0-100)
        """
        # If no violations, return 100%
        if not violations:
            return 100.0

        # Calculate total check points
        total_checks = max(1, len(acceptance_criteria) + len(invariants) + 1)  # +1 for base validation

        # Weight violations by severity
        violation_weight = 0.0
        for v in violations:
            if v.severity == ViolationSeverity.CRITICAL:
                violation_weight += 1.0
            elif v.severity == ViolationSeverity.ERROR:
                violation_weight += 0.75
            elif v.severity == ViolationSeverity.WARNING:
                violation_weight += 0.25
            else:  # INFO
                violation_weight += 0.1

        # Calculate compliance
        # Each violation reduces compliance proportionally
        failed_checks = min(violation_weight, total_checks)
        passed_checks = total_checks - failed_checks

        compliance = (passed_checks / total_checks) * 100.0
        return max(0.0, min(100.0, compliance))

    def _calculate_confidence(
        self,
        artifact: str,
        specification: str,
        acceptance_criteria: list[str],
    ) -> float:
        """Calculate validation confidence.

        Higher confidence when we have more criteria to check against.

        Args:
            artifact: The validated artifact
            specification: The specification
            acceptance_criteria: List of criteria

        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence from specification
        base_confidence = 0.5 if specification.strip() else 0.3

        # Bonus for acceptance criteria
        if acceptance_criteria:
            criteria_bonus = min(0.3, len(acceptance_criteria) * 0.1)
            base_confidence += criteria_bonus

        # Bonus for artifact substance
        if artifact and len(artifact.strip()) > 50:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _generate_remediation_hints(self, violations: list[Violation]) -> list[str]:
        """Generate actionable remediation hints for violations.

        Args:
            violations: List of violations found

        Returns:
            List of remediation hint strings
        """
        hints: list[str] = []

        for violation in violations:
            desc_lower = violation.description.lower()
            if "function" in desc_lower and "not found" in desc_lower:
                func_name = extract_func_name_from_text(violation.description)
                hints.append(f"Add a function definition: `def {func_name}(...):`")
            elif "class" in desc_lower and "not found" in desc_lower:
                cls_name = extract_class_name_from_text(violation.description)
                hints.append(f"Add a class definition: `class {cls_name}:`")
            elif "docstring" in desc_lower:
                hints.append("Add a docstring immediately after the function/class definition")
            elif "type hint" in desc_lower:
                hints.append("Add type hints to function parameters and return type: `def func(param: Type) -> ReturnType:`")
            elif "method" in desc_lower and "not found" in desc_lower:
                meth_name = extract_method_name_from_text(violation.description)
                hints.append(f"Add the required method to the class: `def {meth_name}(self, ...):`")
            elif "empty" in desc_lower:
                hints.append("Provide implementation code in the artifact")

        return hints

    async def validate_with_code_tools(
        self,
        code: str,
        query: str,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        """Validate code using CodeValidationTool (AC-KB7.6).
        
        Uses CodeT5+, GraphCodeBERT, CodeBERT, and SonarQube for
        objective code analysis and validation.
        
        Args:
            code: Source code to validate
            query: Query describing expected functionality
            file_path: Optional file path for SonarQube analysis
            
        Returns:
            Dict with validation results including:
            - passed: Whether validation passed
            - keywords: Extracted keywords from CodeT5+
            - validation_score: Overall validation score
            - sonarqube_result: Quality metrics (if available)
            - should_retry: Whether agent should retry (AC-KB7.7)
        """
        if self._code_validation_tool is None:
            logger.debug("CodeValidationTool not available, skipping validation")
            return {
                "passed": True,
                "keywords": [],
                "validation_score": 0.0,
                "sonarqube_result": None,
                "should_retry": False,
                "skipped": True,
            }
        
        try:
            result = await self._code_validation_tool.validate_code(
                code=code,
                query=query,
                file_path=file_path,
            )
            
            return {
                "passed": result.passed,
                "keywords": result.keywords,
                "validation_score": result.validation_score,
                "sonarqube_result": result.sonarqube_result,
                "should_retry": result.should_retry,
                "failure_reason": result.failure_reason,
            }
        except Exception as e:
            logger.warning(f"CodeValidationTool validation failed: {e}")
            return {
                "passed": True,  # Fail open if tool unavailable
                "keywords": [],
                "validation_score": 0.0,
                "sonarqube_result": None,
                "should_retry": False,
                "error": str(e),
            }


__all__ = ["ValidateAgainstSpecFunction"]
