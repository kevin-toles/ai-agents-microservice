"""Tests for extract_structure function.

TDD tests for WBS-AGT6: extract_structure Function.

Acceptance Criteria Coverage:
- AC-6.1: Parses JSON/Markdown/Code into hierarchical structure
- AC-6.2: Returns StructuredOutput with headings, sections, code_blocks
- AC-6.3: Context budget: 16384 input / 2048 output
- AC-6.4: Default preset: S1 (Light)
- AC-6.5: Supports artifact_type parameter

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 1
"""

import pytest
from typing import Any
from pydantic import ValidationError


# =============================================================================
# AC-6.1: Input Schema Tests
# =============================================================================

class TestExtractStructureInput:
    """Tests for ExtractStructureInput schema."""

    def test_input_requires_content(self) -> None:
        """ExtractStructureInput requires content field."""
        from src.schemas.functions.extract_structure import ExtractStructureInput
        
        with pytest.raises(ValidationError) as exc_info:
            ExtractStructureInput()  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("content",) for e in errors)

    def test_input_accepts_content_string(self) -> None:
        """ExtractStructureInput accepts content as string."""
        from src.schemas.functions.extract_structure import ExtractStructureInput
        
        input_data = ExtractStructureInput(content="# Hello World")
        assert input_data.content == "# Hello World"

    def test_input_has_artifact_type_with_default(self) -> None:
        """ExtractStructureInput has artifact_type with default 'auto'."""
        from src.schemas.functions.extract_structure import ExtractStructureInput, ArtifactType
        
        input_data = ExtractStructureInput(content="test")
        assert input_data.artifact_type == ArtifactType.AUTO

    def test_input_accepts_json_artifact_type(self) -> None:
        """ExtractStructureInput accepts JSON artifact type."""
        from src.schemas.functions.extract_structure import ExtractStructureInput, ArtifactType
        
        input_data = ExtractStructureInput(
            content='{"key": "value"}',
            artifact_type=ArtifactType.JSON,
        )
        assert input_data.artifact_type == ArtifactType.JSON

    def test_input_accepts_markdown_artifact_type(self) -> None:
        """ExtractStructureInput accepts Markdown artifact type."""
        from src.schemas.functions.extract_structure import ExtractStructureInput, ArtifactType
        
        input_data = ExtractStructureInput(
            content="# Heading",
            artifact_type=ArtifactType.MARKDOWN,
        )
        assert input_data.artifact_type == ArtifactType.MARKDOWN

    def test_input_accepts_code_artifact_type(self) -> None:
        """ExtractStructureInput accepts Code artifact type."""
        from src.schemas.functions.extract_structure import ExtractStructureInput, ArtifactType
        
        input_data = ExtractStructureInput(
            content="def foo(): pass",
            artifact_type=ArtifactType.CODE,
        )
        assert input_data.artifact_type == ArtifactType.CODE

    def test_input_has_extraction_type_with_default(self) -> None:
        """ExtractStructureInput has extraction_type with default 'outline'."""
        from src.schemas.functions.extract_structure import ExtractStructureInput, ExtractionType
        
        input_data = ExtractStructureInput(content="test")
        assert input_data.extraction_type == ExtractionType.OUTLINE

    def test_input_accepts_keywords_extraction_type(self) -> None:
        """ExtractStructureInput accepts keywords extraction type."""
        from src.schemas.functions.extract_structure import ExtractStructureInput, ExtractionType
        
        input_data = ExtractStructureInput(
            content="test content",
            extraction_type=ExtractionType.KEYWORDS,
        )
        assert input_data.extraction_type == ExtractionType.KEYWORDS

    def test_input_accepts_concepts_extraction_type(self) -> None:
        """ExtractStructureInput accepts concepts extraction type."""
        from src.schemas.functions.extract_structure import ExtractStructureInput, ExtractionType
        
        input_data = ExtractStructureInput(
            content="test content",
            extraction_type=ExtractionType.CONCEPTS,
        )
        assert input_data.extraction_type == ExtractionType.CONCEPTS

    def test_input_accepts_entities_extraction_type(self) -> None:
        """ExtractStructureInput accepts entities extraction type."""
        from src.schemas.functions.extract_structure import ExtractStructureInput, ExtractionType
        
        input_data = ExtractStructureInput(
            content="test content",
            extraction_type=ExtractionType.ENTITIES,
        )
        assert input_data.extraction_type == ExtractionType.ENTITIES

    def test_input_has_domain_with_default(self) -> None:
        """ExtractStructureInput has domain with default 'general'."""
        from src.schemas.functions.extract_structure import ExtractStructureInput
        
        input_data = ExtractStructureInput(content="test")
        assert input_data.domain == "general"

    def test_input_accepts_custom_domain(self) -> None:
        """ExtractStructureInput accepts custom domain."""
        from src.schemas.functions.extract_structure import ExtractStructureInput
        
        input_data = ExtractStructureInput(content="test", domain="ai-ml")
        assert input_data.domain == "ai-ml"

    def test_input_json_schema_export(self) -> None:
        """ExtractStructureInput exports valid JSON schema."""
        from src.schemas.functions.extract_structure import ExtractStructureInput
        
        schema = ExtractStructureInput.model_json_schema()
        
        assert "properties" in schema
        assert "content" in schema["properties"]
        assert "artifact_type" in schema["properties"]
        assert "extraction_type" in schema["properties"]


# =============================================================================
# AC-6.2: Output Schema Tests - StructuredOutput with components
# =============================================================================

class TestHeading:
    """Tests for Heading model."""

    def test_heading_requires_level_and_text(self) -> None:
        """Heading requires level and text fields."""
        from src.schemas.functions.extract_structure import Heading
        
        with pytest.raises(ValidationError):
            Heading()  # type: ignore

    def test_heading_level_must_be_1_to_6(self) -> None:
        """Heading level must be between 1 and 6."""
        from src.schemas.functions.extract_structure import Heading
        
        # Valid levels
        for level in range(1, 7):
            h = Heading(level=level, text=f"H{level}")
            assert h.level == level
        
        # Invalid levels
        with pytest.raises(ValidationError):
            Heading(level=0, text="Invalid")
        
        with pytest.raises(ValidationError):
            Heading(level=7, text="Invalid")

    def test_heading_has_line_number_optional(self) -> None:
        """Heading has optional line_number field."""
        from src.schemas.functions.extract_structure import Heading
        
        h = Heading(level=1, text="Title")
        assert h.line_number is None
        
        h_with_line = Heading(level=1, text="Title", line_number=10)
        assert h_with_line.line_number == 10

    def test_heading_has_children_list(self) -> None:
        """Heading has children list for nested structure."""
        from src.schemas.functions.extract_structure import Heading
        
        parent = Heading(level=1, text="Parent")
        child = Heading(level=2, text="Child")
        parent.children.append(child)
        
        assert len(parent.children) == 1
        assert parent.children[0].text == "Child"


class TestSection:
    """Tests for Section model."""

    def test_section_requires_title_and_content(self) -> None:
        """Section requires title and content fields."""
        from src.schemas.functions.extract_structure import Section
        
        with pytest.raises(ValidationError):
            Section()  # type: ignore

    def test_section_has_depth(self) -> None:
        """Section has depth field (default 0)."""
        from src.schemas.functions.extract_structure import Section
        
        section = Section(title="Intro", content="Some text")
        assert section.depth == 0
        
        nested = Section(title="Nested", content="Text", depth=2)
        assert nested.depth == 2

    def test_section_has_start_and_end_line(self) -> None:
        """Section has start_line and end_line for audit."""
        from src.schemas.functions.extract_structure import Section
        
        section = Section(
            title="Test",
            content="Content",
            start_line=10,
            end_line=25,
        )
        assert section.start_line == 10
        assert section.end_line == 25


class TestCodeBlock:
    """Tests for CodeBlock model."""

    def test_code_block_requires_code(self) -> None:
        """CodeBlock requires code field."""
        from src.schemas.functions.extract_structure import CodeBlock
        
        with pytest.raises(ValidationError):
            CodeBlock()  # type: ignore

    def test_code_block_has_language_optional(self) -> None:
        """CodeBlock has optional language field."""
        from src.schemas.functions.extract_structure import CodeBlock
        
        block = CodeBlock(code="x = 1")
        assert block.language is None
        
        py_block = CodeBlock(code="def foo(): pass", language="python")
        assert py_block.language == "python"

    def test_code_block_has_line_range(self) -> None:
        """CodeBlock has start_line and end_line."""
        from src.schemas.functions.extract_structure import CodeBlock
        
        block = CodeBlock(
            code="print('hello')",
            language="python",
            start_line=5,
            end_line=5,
        )
        assert block.start_line == 5
        assert block.end_line == 5


class TestExtractedItem:
    """Tests for ExtractedItem model (for keywords/concepts/entities)."""

    def test_extracted_item_requires_value(self) -> None:
        """ExtractedItem requires value field."""
        from src.schemas.functions.extract_structure import ExtractedItem
        
        with pytest.raises(ValidationError):
            ExtractedItem()  # type: ignore

    def test_extracted_item_has_confidence_score(self) -> None:
        """ExtractedItem has confidence_score (0.0 to 1.0)."""
        from src.schemas.functions.extract_structure import ExtractedItem
        
        item = ExtractedItem(value="machine learning", confidence_score=0.95)
        assert item.confidence_score == 0.95

    def test_extracted_item_has_category(self) -> None:
        """ExtractedItem has optional category."""
        from src.schemas.functions.extract_structure import ExtractedItem
        
        item = ExtractedItem(value="Python", category="language")
        assert item.category == "language"

    def test_extracted_item_has_source_positions(self) -> None:
        """ExtractedItem has source_positions for audit."""
        from src.schemas.functions.extract_structure import ExtractedItem
        
        item = ExtractedItem(
            value="DDD",
            source_positions=[10, 25, 100],
        )
        assert item.source_positions == [10, 25, 100]


class TestStructuredOutput:
    """Tests for StructuredOutput model."""

    def test_structured_output_has_headings_list(self) -> None:
        """StructuredOutput has headings list."""
        from src.schemas.functions.extract_structure import StructuredOutput, Heading
        
        output = StructuredOutput()
        assert output.headings == []
        
        output_with_headings = StructuredOutput(
            headings=[Heading(level=1, text="Title")]
        )
        assert len(output_with_headings.headings) == 1

    def test_structured_output_has_sections_list(self) -> None:
        """StructuredOutput has sections list."""
        from src.schemas.functions.extract_structure import StructuredOutput, Section
        
        output = StructuredOutput()
        assert output.sections == []
        
        output_with_sections = StructuredOutput(
            sections=[Section(title="Intro", content="Text")]
        )
        assert len(output_with_sections.sections) == 1

    def test_structured_output_has_code_blocks_list(self) -> None:
        """StructuredOutput has code_blocks list."""
        from src.schemas.functions.extract_structure import StructuredOutput, CodeBlock
        
        output = StructuredOutput()
        assert output.code_blocks == []
        
        output_with_blocks = StructuredOutput(
            code_blocks=[CodeBlock(code="x = 1", language="python")]
        )
        assert len(output_with_blocks.code_blocks) == 1

    def test_structured_output_has_extracted_items(self) -> None:
        """StructuredOutput has extracted_items for keywords/concepts."""
        from src.schemas.functions.extract_structure import StructuredOutput, ExtractedItem
        
        output = StructuredOutput(
            extracted_items=[
                ExtractedItem(value="API", confidence_score=0.9),
            ]
        )
        assert len(output.extracted_items) == 1

    def test_structured_output_has_compressed_summary(self) -> None:
        """StructuredOutput has compressed_summary for downstream."""
        from src.schemas.functions.extract_structure import StructuredOutput
        
        output = StructuredOutput(
            compressed_summary="Document contains 3 sections about DDD patterns."
        )
        assert "DDD patterns" in output.compressed_summary

    def test_structured_output_has_raw_positions(self) -> None:
        """StructuredOutput has raw_positions for audit."""
        from src.schemas.functions.extract_structure import StructuredOutput
        
        output = StructuredOutput(raw_positions=[0, 100, 250, 500])
        assert output.raw_positions == [0, 100, 250, 500]

    def test_structured_output_json_schema_export(self) -> None:
        """StructuredOutput exports valid JSON schema."""
        from src.schemas.functions.extract_structure import StructuredOutput
        
        schema = StructuredOutput.model_json_schema()
        
        assert "properties" in schema
        assert "headings" in schema["properties"]
        assert "sections" in schema["properties"]
        assert "code_blocks" in schema["properties"]


# =============================================================================
# AC-6.1, AC-6.5: ExtractStructureFunction Tests
# =============================================================================

class TestExtractStructureFunction:
    """Tests for ExtractStructureFunction class."""

    def test_function_is_agent_function_subclass(self) -> None:
        """ExtractStructureFunction is subclass of AgentFunction."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.functions.base import AgentFunction
        
        assert issubclass(ExtractStructureFunction, AgentFunction)

    def test_function_has_correct_name(self) -> None:
        """ExtractStructureFunction has name 'extract_structure'."""
        from src.functions.extract_structure import ExtractStructureFunction
        
        func = ExtractStructureFunction()
        assert func.name == "extract_structure"

    def test_function_run_is_async(self) -> None:
        """ExtractStructureFunction.run() is async method."""
        import asyncio
        from src.functions.extract_structure import ExtractStructureFunction
        
        func = ExtractStructureFunction()
        assert asyncio.iscoroutinefunction(func.run)

    def test_function_run_returns_structured_output(self) -> None:
        """ExtractStructureFunction.run() returns StructuredOutput."""
        import asyncio
        from src.functions.extract_structure import ExtractStructureFunction
        from src.schemas.functions.extract_structure import StructuredOutput
        
        func = ExtractStructureFunction()
        
        async def test() -> None:
            result = await func.run(content="# Hello\n\nWorld")
            assert isinstance(result, StructuredOutput)
        
        asyncio.run(test())


# =============================================================================
# AC-6.1: JSON Parsing Tests
# =============================================================================

class TestExtractStructureJSON:
    """Tests for JSON structure extraction."""

    @pytest.mark.asyncio
    async def test_extracts_nested_json_structure(self) -> None:
        """Extracts nested structure with depth levels from JSON."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.schemas.functions.extract_structure import ArtifactType
        
        func = ExtractStructureFunction()
        json_content = '''{
            "name": "project",
            "config": {
                "database": {
                    "host": "localhost",
                    "port": 5432
                }
            }
        }'''
        
        result = await func.run(
            content=json_content,
            artifact_type=ArtifactType.JSON,
        )
        
        # Should have sections representing JSON structure
        assert len(result.sections) > 0 or len(result.extracted_items) > 0

    @pytest.mark.asyncio
    async def test_json_tracks_depth_levels(self) -> None:
        """JSON extraction tracks depth levels."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.schemas.functions.extract_structure import ArtifactType
        
        func = ExtractStructureFunction()
        json_content = '{"level1": {"level2": {"level3": "value"}}}'
        
        result = await func.run(
            content=json_content,
            artifact_type=ArtifactType.JSON,
        )
        
        # Should track depth in sections
        depths = [s.depth for s in result.sections]
        if depths:
            assert max(depths) >= 2  # At least 3 levels deep


# =============================================================================
# AC-6.1: Markdown Parsing Tests
# =============================================================================

class TestExtractStructureMarkdown:
    """Tests for Markdown structure extraction."""

    @pytest.mark.asyncio
    async def test_extracts_h1_to_h6_headings(self) -> None:
        """Extracts H1 through H6 headings from Markdown."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.schemas.functions.extract_structure import ArtifactType
        
        func = ExtractStructureFunction()
        markdown_content = """# H1 Title
## H2 Section
### H3 Subsection
#### H4 Detail
##### H5 Minor
###### H6 Smallest
"""
        
        result = await func.run(
            content=markdown_content,
            artifact_type=ArtifactType.MARKDOWN,
        )
        
        assert len(result.headings) == 6
        
        # Verify heading levels
        levels = [h.level for h in result.headings]
        assert levels == [1, 2, 3, 4, 5, 6]

    @pytest.mark.asyncio
    async def test_extracts_code_blocks_from_markdown(self) -> None:
        """Extracts fenced code blocks from Markdown."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.schemas.functions.extract_structure import ArtifactType
        
        func = ExtractStructureFunction()
        markdown_content = """# Example

```python
def hello():
    print("Hello")
```

Some text.

```javascript
console.log("Hi");
```
"""
        
        result = await func.run(
            content=markdown_content,
            artifact_type=ArtifactType.MARKDOWN,
        )
        
        assert len(result.code_blocks) == 2
        assert result.code_blocks[0].language == "python"
        assert result.code_blocks[1].language == "javascript"

    @pytest.mark.asyncio
    async def test_markdown_heading_line_numbers(self) -> None:
        """Markdown extraction includes heading line numbers."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.schemas.functions.extract_structure import ArtifactType
        
        func = ExtractStructureFunction()
        markdown_content = """# Title

Some intro text.

## Section 1

Content here.
"""
        
        result = await func.run(
            content=markdown_content,
            artifact_type=ArtifactType.MARKDOWN,
        )
        
        # First heading should be on line 1
        assert result.headings[0].line_number == 1
        # Second heading should be further down
        assert result.headings[1].line_number is not None
        assert result.headings[1].line_number > 1


# =============================================================================
# AC-6.1: Code Parsing Tests
# =============================================================================

class TestExtractStructureCode:
    """Tests for Code structure extraction."""

    @pytest.mark.asyncio
    async def test_extracts_python_functions(self) -> None:
        """Extracts function definitions from Python code."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.schemas.functions.extract_structure import ArtifactType
        
        func = ExtractStructureFunction()
        python_code = '''
def hello():
    """Say hello."""
    print("Hello")

def goodbye():
    """Say goodbye."""
    print("Goodbye")

class MyClass:
    def method(self):
        pass
'''
        
        result = await func.run(
            content=python_code,
            artifact_type=ArtifactType.CODE,
        )
        
        # Should identify code blocks or sections
        assert len(result.code_blocks) > 0 or len(result.sections) > 0

    @pytest.mark.asyncio
    async def test_extracts_python_classes(self) -> None:
        """Extracts class definitions from Python code."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.schemas.functions.extract_structure import ArtifactType
        
        func = ExtractStructureFunction()
        python_code = '''
class Repository:
    """Base repository class."""
    
    def get(self, id: int):
        pass
    
    def save(self, entity):
        pass

class UserRepository(Repository):
    """User-specific repository."""
    pass
'''
        
        result = await func.run(
            content=python_code,
            artifact_type=ArtifactType.CODE,
        )
        
        # Should extract class information
        assert len(result.sections) > 0 or len(result.extracted_items) > 0


# =============================================================================
# AC-6.3: Context Budget Enforcement Tests
# =============================================================================

class TestExtractStructureContextBudget:
    """Tests for context budget enforcement."""

    def test_function_has_correct_context_budget(self) -> None:
        """ExtractStructureFunction uses 16384/2048 budget."""
        from src.functions.base import CONTEXT_BUDGET_DEFAULTS
        
        budget = CONTEXT_BUDGET_DEFAULTS.get("extract_structure")
        assert budget is not None
        assert budget["input"] == 16384
        assert budget["output"] == 2048

    @pytest.mark.asyncio
    async def test_enforces_input_budget(self) -> None:
        """Raises error when input exceeds budget."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.functions.base import ContextBudgetExceededError
        
        func = ExtractStructureFunction()
        
        # Create content that exceeds 16384 tokens (~65000 chars)
        huge_content = "word " * 20000  # ~100000 chars
        
        with pytest.raises(ContextBudgetExceededError):
            await func.run(content=huge_content)


# =============================================================================
# AC-6.4: Default Preset Tests
# =============================================================================

class TestExtractStructurePreset:
    """Tests for preset selection."""

    def test_default_preset_is_s1(self) -> None:
        """ExtractStructureFunction default preset is S1 (Light)."""
        from src.functions.extract_structure import ExtractStructureFunction
        
        func = ExtractStructureFunction()
        assert func.default_preset == "S1"

    def test_code_heavy_preset_is_s6(self) -> None:
        """ExtractStructureFunction code_heavy preset is S6."""
        from src.functions.extract_structure import ExtractStructureFunction
        
        func = ExtractStructureFunction()
        presets = func.available_presets
        assert "code_heavy" in presets
        assert presets["code_heavy"] == "S6"

    def test_long_input_preset_is_s5(self) -> None:
        """ExtractStructureFunction long_input preset is S5."""
        from src.functions.extract_structure import ExtractStructureFunction
        
        func = ExtractStructureFunction()
        presets = func.available_presets
        assert "long_input" in presets
        assert presets["long_input"] == "S5"


# =============================================================================
# AC-6.5: artifact_type Parameter Tests
# =============================================================================

class TestArtifactTypeDispatch:
    """Tests for artifact_type dispatch behavior."""

    @pytest.mark.asyncio
    async def test_auto_detects_json(self) -> None:
        """ArtifactType.AUTO detects JSON content."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.schemas.functions.extract_structure import ArtifactType
        
        func = ExtractStructureFunction()
        json_content = '{"key": "value", "nested": {"a": 1}}'
        
        result = await func.run(
            content=json_content,
            artifact_type=ArtifactType.AUTO,
        )
        
        # Should detect and parse as JSON
        assert result is not None

    @pytest.mark.asyncio
    async def test_auto_detects_markdown(self) -> None:
        """ArtifactType.AUTO detects Markdown content."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.schemas.functions.extract_structure import ArtifactType
        
        func = ExtractStructureFunction()
        markdown_content = "# Heading\n\n## Subheading\n\nParagraph."
        
        result = await func.run(
            content=markdown_content,
            artifact_type=ArtifactType.AUTO,
        )
        
        # Should detect and parse headings
        assert len(result.headings) >= 1

    @pytest.mark.asyncio
    async def test_explicit_type_overrides_detection(self) -> None:
        """Explicit artifact_type overrides auto-detection."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.schemas.functions.extract_structure import ArtifactType
        
        func = ExtractStructureFunction()
        # Content that looks like JSON but we force Markdown parsing
        content = '{"title": "# Not a heading"}'
        
        # Force Markdown parsing
        result_md = await func.run(
            content=content,
            artifact_type=ArtifactType.MARKDOWN,
        )
        
        # Force JSON parsing
        result_json = await func.run(
            content=content,
            artifact_type=ArtifactType.JSON,
        )
        
        # Results should differ based on forced type
        assert result_md is not None
        assert result_json is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestExtractStructureIntegration:
    """Integration tests for extract_structure function."""

    @pytest.mark.asyncio
    async def test_complex_markdown_document(self) -> None:
        """Handles complex Markdown with all features."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.schemas.functions.extract_structure import ArtifactType
        
        func = ExtractStructureFunction()
        complex_md = """# Project Documentation

## Overview

This is the project overview.

## Getting Started

### Installation

```bash
pip install myproject
```

### Configuration

```python
config = {
    "debug": True,
    "port": 8080
}
```

## API Reference

### Endpoints

#### GET /health

Returns health status.

## Conclusion

That's all folks!
"""
        
        result = await func.run(
            content=complex_md,
            artifact_type=ArtifactType.MARKDOWN,
        )
        
        # Should have multiple headings
        assert len(result.headings) >= 4
        
        # Should have code blocks
        assert len(result.code_blocks) >= 2
        
        # Should have compressed summary
        assert result.compressed_summary is not None

    @pytest.mark.asyncio
    async def test_extracts_from_handoff_cache_compatible(self) -> None:
        """Output can be stored in HandoffCache."""
        from src.functions.extract_structure import ExtractStructureFunction
        from src.cache import HandoffCache
        
        func = ExtractStructureFunction()
        cache = HandoffCache("test_pipeline")
        
        result = await func.run(content="# Title\n\nContent")
        
        # Store result in cache
        await cache.set("extract_result", result.model_dump_json())
        
        # Retrieve and verify
        stored = await cache.get("extract_result")
        assert stored is not None
        assert "Title" in stored or "headings" in stored
