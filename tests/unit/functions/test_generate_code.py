"""Tests for generate_code agent function.

WBS-AGT8: generate_code Function tests.

TDD RED Phase: Tests written before implementation.

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 3
Reference: WBS.md → WBS-AGT8

Acceptance Criteria:
- AC-8.1: Generates code from natural language spec
- AC-8.2: Returns CodeOutput with language, code, explanation
- AC-8.3: Context budget: 4096 input / 8192 output
- AC-8.4: Default preset: D4 (Standard)
- AC-8.5: Supports target_language parameter
- AC-8.6: Includes test stubs when include_tests=True
"""

import ast
import pytest


# =============================================================================
# AC-8.1: Input Schema Tests
# =============================================================================

class TestGenerateCodeInput:
    """Tests for GenerateCodeInput schema."""

    def test_input_requires_specification(self) -> None:
        """GenerateCodeInput requires specification field."""
        from src.schemas.functions.generate_code import GenerateCodeInput
        
        with pytest.raises(Exception):  # ValidationError
            GenerateCodeInput()

    def test_input_accepts_valid_specification(self) -> None:
        """GenerateCodeInput accepts valid specification."""
        from src.schemas.functions.generate_code import GenerateCodeInput
        
        input_data = GenerateCodeInput(
            specification="Create a function that adds two numbers"
        )
        assert input_data.specification == "Create a function that adds two numbers"

    def test_input_has_target_language_field(self) -> None:
        """GenerateCodeInput has target_language field (AC-8.5)."""
        from src.schemas.functions.generate_code import GenerateCodeInput
        
        input_data = GenerateCodeInput(
            specification="Create a class",
            target_language="python"
        )
        assert input_data.target_language == "python"

    def test_input_target_language_default(self) -> None:
        """GenerateCodeInput defaults target_language to 'python'."""
        from src.schemas.functions.generate_code import GenerateCodeInput
        
        input_data = GenerateCodeInput(specification="Test spec")
        assert input_data.target_language == "python"

    def test_input_has_include_tests_field(self) -> None:
        """GenerateCodeInput has include_tests field (AC-8.6)."""
        from src.schemas.functions.generate_code import GenerateCodeInput
        
        input_data = GenerateCodeInput(
            specification="Create a class",
            include_tests=True
        )
        assert input_data.include_tests is True

    def test_input_include_tests_default_false(self) -> None:
        """GenerateCodeInput defaults include_tests to False."""
        from src.schemas.functions.generate_code import GenerateCodeInput
        
        input_data = GenerateCodeInput(specification="Test spec")
        assert input_data.include_tests is False

    def test_input_has_context_artifacts_field(self) -> None:
        """GenerateCodeInput has context_artifacts for related code."""
        from src.schemas.functions.generate_code import GenerateCodeInput
        
        input_data = GenerateCodeInput(
            specification="Create a class",
            context_artifacts=["def existing_func(): pass"]
        )
        assert len(input_data.context_artifacts) == 1

    def test_input_has_patterns_to_follow_field(self) -> None:
        """GenerateCodeInput has patterns_to_follow from CODING_PATTERNS_ANALYSIS."""
        from src.schemas.functions.generate_code import GenerateCodeInput
        
        input_data = GenerateCodeInput(
            specification="Create a repository",
            patterns_to_follow=["repository-pattern", "factory-pattern"]
        )
        assert len(input_data.patterns_to_follow) == 2

    def test_input_has_constraints_field(self) -> None:
        """GenerateCodeInput has constraints for must-haves."""
        from src.schemas.functions.generate_code import GenerateCodeInput
        
        input_data = GenerateCodeInput(
            specification="Create a class",
            constraints=["Must use async/await", "Must have type hints"]
        )
        assert len(input_data.constraints) == 2

    def test_input_json_schema_export(self) -> None:
        """GenerateCodeInput exports valid JSON schema."""
        from src.schemas.functions.generate_code import GenerateCodeInput
        
        schema = GenerateCodeInput.model_json_schema()
        
        assert "properties" in schema
        assert "specification" in schema["properties"]
        assert "target_language" in schema["properties"]
        assert "include_tests" in schema["properties"]


# =============================================================================
# AC-8.2: Output Schema Tests
# =============================================================================

class TestCodeOutput:
    """Tests for CodeOutput schema."""

    def test_output_requires_code(self) -> None:
        """CodeOutput requires code field."""
        from src.schemas.functions.generate_code import CodeOutput
        
        with pytest.raises(Exception):  # ValidationError
            CodeOutput()

    def test_output_has_language_field(self) -> None:
        """CodeOutput has language field (AC-8.2)."""
        from src.schemas.functions.generate_code import CodeOutput
        
        output = CodeOutput(
            code="def add(a, b): return a + b",
            language="python"
        )
        assert output.language == "python"

    def test_output_has_code_field(self) -> None:
        """CodeOutput has code field (AC-8.2)."""
        from src.schemas.functions.generate_code import CodeOutput
        
        output = CodeOutput(
            code="def add(a, b): return a + b",
            language="python"
        )
        assert "def add" in output.code

    def test_output_has_explanation_field(self) -> None:
        """CodeOutput has explanation field (AC-8.2)."""
        from src.schemas.functions.generate_code import CodeOutput
        
        output = CodeOutput(
            code="def add(a, b): return a + b",
            language="python",
            explanation="Simple addition function"
        )
        assert output.explanation == "Simple addition function"

    def test_output_has_test_hints_field(self) -> None:
        """CodeOutput has test_hints for suggested test cases."""
        from src.schemas.functions.generate_code import CodeOutput
        
        output = CodeOutput(
            code="def add(a, b): return a + b",
            language="python",
            test_hints=["test positive numbers", "test negative numbers"]
        )
        assert len(output.test_hints) == 2

    def test_output_has_test_code_field(self) -> None:
        """CodeOutput has test_code for generated tests (AC-8.6)."""
        from src.schemas.functions.generate_code import CodeOutput
        
        output = CodeOutput(
            code="def add(a, b): return a + b",
            language="python",
            test_code="def test_add(): assert add(1, 2) == 3"
        )
        assert "test_add" in output.test_code

    def test_output_has_compressed_intent_field(self) -> None:
        """CodeOutput has compressed_intent for downstream validation."""
        from src.schemas.functions.generate_code import CodeOutput
        
        output = CodeOutput(
            code="def add(a, b): return a + b",
            language="python",
            compressed_intent="Function to add two numbers"
        )
        assert output.compressed_intent is not None

    def test_output_has_citations_field(self) -> None:
        """CodeOutput has citations list for sources used."""
        from src.schemas.functions.generate_code import CodeOutput
        from src.schemas.citations import Citation, SourceMetadata
        
        source = SourceMetadata(
            source_type="code",
            repo="code-reference-engine",
            file_path="backend/utils/math.py",
        )
        citation = Citation(marker=1, source=source)
        
        output = CodeOutput(
            code="def add(a, b): return a + b",
            language="python",
            citations=[citation]
        )
        assert len(output.citations) == 1

    def test_output_json_schema_export(self) -> None:
        """CodeOutput exports valid JSON schema."""
        from src.schemas.functions.generate_code import CodeOutput
        
        schema = CodeOutput.model_json_schema()
        
        assert "properties" in schema
        assert "code" in schema["properties"]
        assert "language" in schema["properties"]
        assert "explanation" in schema["properties"]


# =============================================================================
# AC-8.5: Target Language Tests
# =============================================================================

class TestTargetLanguage:
    """Tests for target_language parameter (AC-8.5)."""

    def test_language_enum_has_python(self) -> None:
        """TargetLanguage enum includes Python."""
        from src.schemas.functions.generate_code import TargetLanguage
        
        assert TargetLanguage.PYTHON.value == "python"

    def test_language_enum_has_javascript(self) -> None:
        """TargetLanguage enum includes JavaScript."""
        from src.schemas.functions.generate_code import TargetLanguage
        
        assert TargetLanguage.JAVASCRIPT.value == "javascript"

    def test_language_enum_has_typescript(self) -> None:
        """TargetLanguage enum includes TypeScript."""
        from src.schemas.functions.generate_code import TargetLanguage
        
        assert TargetLanguage.TYPESCRIPT.value == "typescript"

    def test_language_enum_has_java(self) -> None:
        """TargetLanguage enum includes Java."""
        from src.schemas.functions.generate_code import TargetLanguage
        
        assert TargetLanguage.JAVA.value == "java"

    def test_language_enum_has_sql(self) -> None:
        """TargetLanguage enum includes SQL."""
        from src.schemas.functions.generate_code import TargetLanguage
        
        assert TargetLanguage.SQL.value == "sql"


# =============================================================================
# AC-8.3: Context Budget Tests
# =============================================================================

class TestGenerateCodeContextBudget:
    """Tests for context budget enforcement (AC-8.3)."""

    @pytest.mark.asyncio
    async def test_budget_exceeds_input_limit(self) -> None:
        """GenerateCodeFunction raises error when input exceeds 4096 tokens."""
        from src.functions.generate_code import GenerateCodeFunction
        from src.functions.base import ContextBudgetExceededError
        
        func = GenerateCodeFunction()
        # 4096 tokens * 4 chars/token = ~16384 chars
        huge_spec = "x " * 20000  # Way over limit
        
        with pytest.raises(ContextBudgetExceededError):
            await func.run(specification=huge_spec)

    def test_context_budget_values(self) -> None:
        """GenerateCodeFunction has correct budget values (AC-8.3)."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        budget = func.get_context_budget()
        
        # AC-8.3: 4096 input / 8192 output
        assert budget["input"] == 4096
        assert budget["output"] == 8192


# =============================================================================
# AC-8.4: Default Preset Tests
# =============================================================================

class TestGenerateCodePreset:
    """Tests for preset configuration (AC-8.4)."""

    def test_default_preset_is_d4(self) -> None:
        """GenerateCodeFunction has default preset D4 (AC-8.4)."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        assert func.default_preset == "D4"

    def test_available_presets_include_simple(self) -> None:
        """GenerateCodeFunction has simple preset S3."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        assert "simple" in func.available_presets

    def test_available_presets_include_quality(self) -> None:
        """GenerateCodeFunction has quality preset D4."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        assert "quality" in func.available_presets

    def test_available_presets_include_long_file(self) -> None:
        """GenerateCodeFunction has long_file preset S6."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        assert "long_file" in func.available_presets


# =============================================================================
# AC-8.1: Code Generation Tests
# =============================================================================

class TestGenerateCodeFunction:
    """Tests for GenerateCodeFunction core functionality."""

    def test_function_name_is_generate_code(self) -> None:
        """GenerateCodeFunction has correct name."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        assert func.name == "generate_code"

    @pytest.mark.asyncio
    async def test_generates_python_code(self) -> None:
        """GenerateCodeFunction generates Python code (AC-8.1)."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a function that adds two numbers",
            target_language="python"
        )
        
        assert result.language == "python"
        assert "def" in result.code or "lambda" in result.code

    @pytest.mark.asyncio
    async def test_generates_valid_python_syntax(self) -> None:
        """Generated Python code passes syntax validation."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a function that multiplies two numbers",
            target_language="python"
        )
        
        # Should parse without syntax error
        try:
            ast.parse(result.code)
            valid = True
        except SyntaxError:
            valid = False
        
        assert valid, f"Generated code has syntax error: {result.code}"

    @pytest.mark.asyncio
    async def test_includes_explanation(self) -> None:
        """GenerateCodeFunction includes explanation in output (AC-8.2)."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a class representing a user",
            target_language="python"
        )
        
        assert result.explanation is not None
        assert len(result.explanation) > 0

    @pytest.mark.asyncio
    async def test_returns_code_output_type(self) -> None:
        """GenerateCodeFunction returns CodeOutput type."""
        from src.functions.generate_code import GenerateCodeFunction
        from src.schemas.functions.generate_code import CodeOutput
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a simple function",
            target_language="python"
        )
        
        assert isinstance(result, CodeOutput)


# =============================================================================
# AC-8.6: Test Stub Generation Tests
# =============================================================================

class TestTestStubGeneration:
    """Tests for test stub generation (AC-8.6)."""

    @pytest.mark.asyncio
    async def test_include_tests_generates_test_code(self) -> None:
        """include_tests=True generates test code (AC-8.6)."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a function that divides two numbers",
            target_language="python",
            include_tests=True
        )
        
        assert result.test_code is not None
        assert len(result.test_code) > 0

    @pytest.mark.asyncio
    async def test_include_tests_generates_pytest_style(self) -> None:
        """include_tests=True generates pytest-style tests (AC-8.6)."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a function that checks if a number is even",
            target_language="python",
            include_tests=True
        )
        
        # Should have test function
        assert "def test_" in result.test_code or "test_" in result.test_code

    @pytest.mark.asyncio
    async def test_include_tests_false_no_test_code(self) -> None:
        """include_tests=False does not generate test code."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a function that returns hello world",
            target_language="python",
            include_tests=False
        )
        
        # test_code should be None or empty
        assert not result.test_code

    @pytest.mark.asyncio
    async def test_generated_tests_are_valid_python(self) -> None:
        """Generated test code passes syntax validation."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a function that squares a number",
            target_language="python",
            include_tests=True
        )
        
        if result.test_code:
            try:
                ast.parse(result.test_code)
                valid = True
            except SyntaxError:
                valid = False
            
            assert valid, f"Generated test code has syntax error: {result.test_code}"


# =============================================================================
# Language-Specific Generation Tests
# =============================================================================

class TestLanguageSpecificGeneration:
    """Tests for language-specific code generation."""

    @pytest.mark.asyncio
    async def test_generates_javascript(self) -> None:
        """GenerateCodeFunction generates JavaScript code."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a function that concatenates strings",
            target_language="javascript"
        )
        
        assert result.language == "javascript"
        # JavaScript patterns
        assert "function" in result.code or "const" in result.code or "=>" in result.code

    @pytest.mark.asyncio
    async def test_generates_typescript(self) -> None:
        """GenerateCodeFunction generates TypeScript code."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a function that takes a string and returns its length",
            target_language="typescript"
        )
        
        assert result.language == "typescript"
        # TypeScript should have type annotations
        assert ":" in result.code  # Type annotation

    @pytest.mark.asyncio
    async def test_generates_sql(self) -> None:
        """GenerateCodeFunction generates SQL code."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a query to select all users",
            target_language="sql"
        )
        
        assert result.language == "sql"
        assert "SELECT" in result.code.upper()


# =============================================================================
# Context and Pattern Tests
# =============================================================================

class TestContextAndPatterns:
    """Tests for context artifacts and patterns."""

    @pytest.mark.asyncio
    async def test_uses_context_artifacts(self) -> None:
        """GenerateCodeFunction considers context_artifacts."""
        from src.functions.generate_code import GenerateCodeFunction
        
        existing_code = '''
class BaseRepository:
    """Base class for repositories."""
    def get(self, id: str):
        raise NotImplementedError
'''
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a UserRepository that extends BaseRepository",
            target_language="python",
            context_artifacts=[existing_code]
        )
        
        # Should reference the base class pattern
        assert "BaseRepository" in result.code or "Repository" in result.code

    @pytest.mark.asyncio
    async def test_follows_patterns(self) -> None:
        """GenerateCodeFunction follows patterns_to_follow."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a factory for creating database connections",
            target_language="python",
            patterns_to_follow=["factory-pattern"]
        )
        
        # Should use factory pattern naming/structure
        assert "factory" in result.code.lower() or "create" in result.code.lower()

    @pytest.mark.asyncio
    async def test_respects_constraints(self) -> None:
        """GenerateCodeFunction respects constraints."""
        from src.functions.generate_code import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a function that fetches data",
            target_language="python",
            constraints=["Must use async/await"]
        )
        
        # Should have async keyword
        assert "async" in result.code


# =============================================================================
# Integration Tests
# =============================================================================

class TestGenerateCodeIntegration:
    """Integration tests for generate_code function."""

    @pytest.mark.asyncio
    async def test_full_workflow(self) -> None:
        """Full workflow: spec → code with tests and explanation."""
        from src.functions.generate_code import GenerateCodeFunction
        from src.schemas.functions.generate_code import CodeOutput
        
        func = GenerateCodeFunction()
        result = await func.run(
            specification="Create a Calculator class with add and subtract methods",
            target_language="python",
            include_tests=True,
            constraints=["Must have type hints"]
        )
        
        # Check all output fields
        assert isinstance(result, CodeOutput)
        assert result.language == "python"
        assert len(result.code) > 0
        assert result.explanation is not None
        assert result.test_code is not None

    @pytest.mark.asyncio
    async def test_import_from_functions_package(self) -> None:
        """GenerateCodeFunction can be imported from functions package."""
        from src.functions import GenerateCodeFunction
        
        func = GenerateCodeFunction()
        assert func.name == "generate_code"

    @pytest.mark.asyncio
    async def test_import_schemas_from_package(self) -> None:
        """Code schemas can be imported from schemas package."""
        from src.schemas.functions import (
            GenerateCodeInput,
            CodeOutput,
            TargetLanguage,
        )
        
        # Should not raise
        assert GenerateCodeInput is not None
        assert CodeOutput is not None
        assert TargetLanguage is not None
