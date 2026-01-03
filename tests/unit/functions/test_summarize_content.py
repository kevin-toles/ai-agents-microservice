"""Tests for summarize_content function.

TDD tests for WBS-AGT7: summarize_content Function.

Acceptance Criteria Coverage:
- AC-7.1: Generates summaries with citation markers [^N]
- AC-7.2: Returns CitedContent with footnotes list
- AC-7.3: Context budget: 8192 input / 4096 output
- AC-7.4: Default preset: D4 (Standard)
- AC-7.5: Supports detail_level parameter (brief/standard/comprehensive)

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 2
"""

import pytest
from typing import Any
from pydantic import ValidationError


# =============================================================================
# AC-7.5: Input Schema Tests - detail_level parameter
# =============================================================================

class TestSummarizeContentInput:
    """Tests for SummarizeContentInput schema."""

    def test_input_requires_content(self) -> None:
        """SummarizeContentInput requires content field."""
        from src.schemas.functions.summarize_content import SummarizeContentInput
        
        with pytest.raises(ValidationError) as exc_info:
            SummarizeContentInput()  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("content",) for e in errors)

    def test_input_accepts_content_string(self) -> None:
        """SummarizeContentInput accepts content as string."""
        from src.schemas.functions.summarize_content import SummarizeContentInput
        
        input_data = SummarizeContentInput(content="This is a long text to summarize.")
        assert input_data.content == "This is a long text to summarize."

    def test_input_has_detail_level_with_default(self) -> None:
        """SummarizeContentInput has detail_level with default 'standard'."""
        from src.schemas.functions.summarize_content import SummarizeContentInput, DetailLevel
        
        input_data = SummarizeContentInput(content="test")
        assert input_data.detail_level == DetailLevel.STANDARD

    def test_input_accepts_brief_detail_level(self) -> None:
        """SummarizeContentInput accepts brief detail level."""
        from src.schemas.functions.summarize_content import SummarizeContentInput, DetailLevel
        
        input_data = SummarizeContentInput(
            content="test",
            detail_level=DetailLevel.BRIEF,
        )
        assert input_data.detail_level == DetailLevel.BRIEF

    def test_input_accepts_comprehensive_detail_level(self) -> None:
        """SummarizeContentInput accepts comprehensive detail level."""
        from src.schemas.functions.summarize_content import SummarizeContentInput, DetailLevel
        
        input_data = SummarizeContentInput(
            content="test",
            detail_level=DetailLevel.COMPREHENSIVE,
        )
        assert input_data.detail_level == DetailLevel.COMPREHENSIVE

    def test_input_has_target_tokens_optional(self) -> None:
        """SummarizeContentInput has optional target_tokens field."""
        from src.schemas.functions.summarize_content import SummarizeContentInput
        
        input_data = SummarizeContentInput(content="test")
        assert input_data.target_tokens is None
        
        input_with_target = SummarizeContentInput(content="test", target_tokens=500)
        assert input_with_target.target_tokens == 500

    def test_input_has_preserve_list(self) -> None:
        """SummarizeContentInput has preserve list for must-include concepts."""
        from src.schemas.functions.summarize_content import SummarizeContentInput
        
        input_data = SummarizeContentInput(
            content="test",
            preserve=["DDD", "Repository Pattern"],
        )
        assert input_data.preserve == ["DDD", "Repository Pattern"]

    def test_input_has_style_with_default(self) -> None:
        """SummarizeContentInput has style with default 'technical'."""
        from src.schemas.functions.summarize_content import SummarizeContentInput, SummaryStyle
        
        input_data = SummarizeContentInput(content="test")
        assert input_data.style == SummaryStyle.TECHNICAL

    def test_input_accepts_executive_style(self) -> None:
        """SummarizeContentInput accepts executive style."""
        from src.schemas.functions.summarize_content import SummarizeContentInput, SummaryStyle
        
        input_data = SummarizeContentInput(
            content="test",
            style=SummaryStyle.EXECUTIVE,
        )
        assert input_data.style == SummaryStyle.EXECUTIVE

    def test_input_accepts_bullets_style(self) -> None:
        """SummarizeContentInput accepts bullets style."""
        from src.schemas.functions.summarize_content import SummarizeContentInput, SummaryStyle
        
        input_data = SummarizeContentInput(
            content="test",
            style=SummaryStyle.BULLETS,
        )
        assert input_data.style == SummaryStyle.BULLETS

    def test_input_has_sources_list(self) -> None:
        """SummarizeContentInput has sources list for citations."""
        from src.schemas.functions.summarize_content import SummarizeContentInput
        from src.schemas.citations import SourceMetadata
        
        source = SourceMetadata(
            source_type="book",
            title="Domain-Driven Design",
            author="Evans, Eric",
        )
        input_data = SummarizeContentInput(
            content="test",
            sources=[source],
        )
        assert len(input_data.sources) == 1

    def test_input_json_schema_export(self) -> None:
        """SummarizeContentInput exports valid JSON schema."""
        from src.schemas.functions.summarize_content import SummarizeContentInput
        
        schema = SummarizeContentInput.model_json_schema()
        
        assert "properties" in schema
        assert "content" in schema["properties"]
        assert "detail_level" in schema["properties"]
        assert "style" in schema["properties"]


# =============================================================================
# AC-7.2: Output Schema Tests - CitedContent with footnotes
# =============================================================================

class TestSummarizeContentOutput:
    """Tests for SummarizeContentOutput schema."""

    def test_output_has_summary(self) -> None:
        """SummarizeContentOutput has summary field."""
        from src.schemas.functions.summarize_content import SummarizeContentOutput
        
        output = SummarizeContentOutput(summary="This is a summary.")
        assert output.summary == "This is a summary."

    def test_output_has_footnotes_list(self) -> None:
        """SummarizeContentOutput has footnotes list (AC-7.2)."""
        from src.schemas.functions.summarize_content import SummarizeContentOutput
        from src.schemas.citations import Citation, SourceMetadata
        
        source = SourceMetadata(
            source_type="book",
            title="Test Book",
        )
        citation = Citation(
            marker=1,
            source=source,
        )
        output = SummarizeContentOutput(
            summary="Summary with citation[^1]",
            footnotes=[citation],
        )
        assert len(output.footnotes) == 1
        assert output.footnotes[0].marker == 1

    def test_output_has_invariants(self) -> None:
        """SummarizeContentOutput has invariants list for validation."""
        from src.schemas.functions.summarize_content import SummarizeContentOutput
        
        output = SummarizeContentOutput(
            summary="Summary",
            invariants=["Fact 1: X > Y", "Fact 2: API uses REST"],
        )
        assert len(output.invariants) == 2

    def test_output_has_compression_ratio(self) -> None:
        """SummarizeContentOutput has compression_ratio field."""
        from src.schemas.functions.summarize_content import SummarizeContentOutput
        
        output = SummarizeContentOutput(
            summary="Short",
            compression_ratio=0.25,
        )
        assert output.compression_ratio == pytest.approx(0.25)

    def test_output_has_token_count(self) -> None:
        """SummarizeContentOutput has token_count field."""
        from src.schemas.functions.summarize_content import SummarizeContentOutput
        
        output = SummarizeContentOutput(
            summary="Summary text",
            token_count=150,
        )
        assert output.token_count == 150

    def test_output_as_cited_content(self) -> None:
        """SummarizeContentOutput can convert to CitedContent."""
        from src.schemas.functions.summarize_content import SummarizeContentOutput
        from src.schemas.citations import Citation, CitedContent, SourceMetadata
        
        source = SourceMetadata(
            source_type="book",
            title="Test Book",
            author="Author",
        )
        citation = Citation(
            marker=1,
            source=source,
        )
        output = SummarizeContentOutput(
            summary="Summary with citation[^1]",
            footnotes=[citation],
        )
        
        cited = output.to_cited_content()
        assert isinstance(cited, CitedContent)
        assert cited.text == output.summary
        assert len(cited.citations) == 1

    def test_output_json_schema_export(self) -> None:
        """SummarizeContentOutput exports valid JSON schema."""
        from src.schemas.functions.summarize_content import SummarizeContentOutput
        
        schema = SummarizeContentOutput.model_json_schema()
        
        assert "properties" in schema
        assert "summary" in schema["properties"]
        assert "footnotes" in schema["properties"]


# =============================================================================
# AC-7.1: Citation Marker Tests
# =============================================================================

class TestCitationMarkers:
    """Tests for citation marker injection."""

    @pytest.mark.asyncio
    async def test_output_contains_citation_markers(self) -> None:
        """Summary output contains [^N] citation markers (AC-7.1)."""
        from src.functions.summarize_content import SummarizeContentFunction
        from src.schemas.citations import SourceMetadata
        
        func = SummarizeContentFunction()
        
        source = SourceMetadata(
            source_type="book",
            title="Domain-Driven Design",
            author="Evans, Eric",
            publisher="Addison-Wesley",
            year=2003,
        )
        
        result = await func.run(
            content="Domain-Driven Design is a software design approach.",
            sources=[source],
        )
        
        # Should contain at least one citation marker
        import re
        markers = re.findall(r"\[\^(\d+)\]", result.summary)
        assert len(markers) >= 1

    @pytest.mark.asyncio
    async def test_markers_match_footnotes(self) -> None:
        """Citation markers [^N] match footnotes in output (AC-7.1, AC-7.2)."""
        from src.functions.summarize_content import SummarizeContentFunction
        from src.schemas.citations import SourceMetadata
        
        func = SummarizeContentFunction()
        
        sources = [
            SourceMetadata(
                source_type="book",
                title="Book One",
                author="Author One",
            ),
            SourceMetadata(
                source_type="book",
                title="Book Two",
                author="Author Two",
            ),
        ]
        
        result = await func.run(
            content="Text from multiple sources.",
            sources=sources,
        )
        
        # Extract markers from summary
        import re
        markers = re.findall(r"\[\^(\d+)\]", result.summary)
        marker_nums = {int(m) for m in markers}
        
        # All markers should have corresponding footnotes
        footnote_nums = {f.marker for f in result.footnotes}
        assert marker_nums.issubset(footnote_nums)

    @pytest.mark.asyncio
    async def test_footnotes_have_chicago_format(self) -> None:
        """Footnotes use Chicago-style citations (AC-7.2)."""
        from src.functions.summarize_content import SummarizeContentFunction
        from src.schemas.citations import SourceMetadata
        
        func = SummarizeContentFunction()
        
        source = SourceMetadata(
            source_type="book",
            title="Domain-Driven Design",
            author="Evans, Eric",
            publisher="Addison-Wesley",
            publication_city="Boston",
            year=2003,
        )
        
        result = await func.run(
            content="DDD principles are important.",
            sources=[source],
        )
        
        # Should have at least one footnote
        assert len(result.footnotes) >= 1
        
        # Footnote should have chicago_format() method
        footnote = result.footnotes[0]
        chicago = footnote.chicago_format()
        
        # Chicago format should include author and title
        assert "Evans" in chicago
        assert "Domain-Driven Design" in chicago


# =============================================================================
# AC-7.3: Context Budget Tests
# =============================================================================

class TestSummarizeContentContextBudget:
    """Tests for context budget enforcement."""

    def test_function_has_correct_context_budget(self) -> None:
        """SummarizeContentFunction uses 8192/4096 budget (AC-7.3)."""
        from src.functions.base import CONTEXT_BUDGET_DEFAULTS
        
        budget = CONTEXT_BUDGET_DEFAULTS.get("summarize_content")
        assert budget is not None
        assert budget["input"] == 8192
        assert budget["output"] == 4096

    @pytest.mark.asyncio
    async def test_enforces_input_budget(self) -> None:
        """Raises error when input exceeds budget."""
        from src.functions.summarize_content import SummarizeContentFunction
        from src.functions.base import ContextBudgetExceededError
        
        func = SummarizeContentFunction()
        
        # Create content that exceeds 8192 tokens (~32000 chars)
        huge_content = "word " * 10000  # ~50000 chars
        
        with pytest.raises(ContextBudgetExceededError):
            await func.run(content=huge_content)


# =============================================================================
# AC-7.4: Default Preset Tests
# =============================================================================

class TestSummarizeContentPreset:
    """Tests for preset selection."""

    def test_default_preset_is_d4(self) -> None:
        """SummarizeContentFunction default preset is D4 (Standard) (AC-7.4)."""
        from src.functions.summarize_content import SummarizeContentFunction
        
        func = SummarizeContentFunction()
        assert func.default_preset == "D4"

    def test_short_input_preset_is_s4(self) -> None:
        """SummarizeContentFunction short_input preset is S4."""
        from src.functions.summarize_content import SummarizeContentFunction
        
        func = SummarizeContentFunction()
        presets = func.available_presets
        assert "short_input" in presets
        assert presets["short_input"] == "S4"

    def test_long_input_preset_is_s5(self) -> None:
        """SummarizeContentFunction long_input preset is S5."""
        from src.functions.summarize_content import SummarizeContentFunction
        
        func = SummarizeContentFunction()
        presets = func.available_presets
        assert "long_input" in presets
        assert presets["long_input"] == "S5"


# =============================================================================
# AC-7.5: detail_level Parameter Tests
# =============================================================================

class TestDetailLevel:
    """Tests for detail_level parameter behavior."""

    @pytest.mark.asyncio
    async def test_brief_produces_short_output(self) -> None:
        """detail_level='brief' produces <500 token output."""
        from src.functions.summarize_content import SummarizeContentFunction
        from src.schemas.functions.summarize_content import DetailLevel
        
        func = SummarizeContentFunction()
        
        # Long input content
        long_content = """
        This is a comprehensive document about software architecture patterns.
        It covers domain-driven design, microservices, event sourcing, and CQRS.
        Each pattern is explained in detail with examples and use cases.
        The repository pattern helps abstract data access logic.
        The unit of work pattern manages transaction boundaries.
        These patterns work together to create maintainable systems.
        """ * 10  # Make it longer
        
        result = await func.run(
            content=long_content,
            detail_level=DetailLevel.BRIEF,
        )
        
        # Brief should be <500 tokens (~2000 chars)
        assert result.token_count is not None
        assert result.token_count < 500

    @pytest.mark.asyncio
    async def test_standard_produces_medium_output(self) -> None:
        """detail_level='standard' produces balanced output."""
        from src.functions.summarize_content import SummarizeContentFunction
        from src.schemas.functions.summarize_content import DetailLevel
        
        func = SummarizeContentFunction()
        
        content = "Test content for summarization." * 50
        
        result = await func.run(
            content=content,
            detail_level=DetailLevel.STANDARD,
        )
        
        # Standard should be reasonable length
        assert result.token_count is not None
        assert result.token_count >= 50

    @pytest.mark.asyncio
    async def test_comprehensive_produces_detailed_output(self) -> None:
        """detail_level='comprehensive' produces detailed output."""
        from src.functions.summarize_content import SummarizeContentFunction
        from src.schemas.functions.summarize_content import DetailLevel
        
        func = SummarizeContentFunction()
        
        content = "Detailed content about patterns." * 50
        
        result = await func.run(
            content=content,
            detail_level=DetailLevel.COMPREHENSIVE,
        )
        
        # Comprehensive should be more detailed
        assert result.token_count is not None
        assert result.token_count >= 100


# =============================================================================
# SummarizeContentFunction Tests
# =============================================================================

class TestSummarizeContentFunction:
    """Tests for SummarizeContentFunction class."""

    def test_function_is_agent_function_subclass(self) -> None:
        """SummarizeContentFunction is subclass of AgentFunction."""
        from src.functions.summarize_content import SummarizeContentFunction
        from src.functions.base import AgentFunction
        
        assert issubclass(SummarizeContentFunction, AgentFunction)

    def test_function_has_correct_name(self) -> None:
        """SummarizeContentFunction has name 'summarize_content'."""
        from src.functions.summarize_content import SummarizeContentFunction
        
        func = SummarizeContentFunction()
        assert func.name == "summarize_content"

    def test_function_run_is_async(self) -> None:
        """SummarizeContentFunction.run() is async method."""
        import asyncio
        from src.functions.summarize_content import SummarizeContentFunction
        
        func = SummarizeContentFunction()
        assert asyncio.iscoroutinefunction(func.run)

    @pytest.mark.asyncio
    async def test_function_run_returns_output(self) -> None:
        """SummarizeContentFunction.run() returns SummarizeContentOutput."""
        from src.functions.summarize_content import SummarizeContentFunction
        from src.schemas.functions.summarize_content import SummarizeContentOutput
        
        func = SummarizeContentFunction()
        
        result = await func.run(content="This is test content to summarize.")
        assert isinstance(result, SummarizeContentOutput)

    @pytest.mark.asyncio
    async def test_function_preserves_concepts(self) -> None:
        """SummarizeContentFunction preserves specified concepts."""
        from src.functions.summarize_content import SummarizeContentFunction
        
        func = SummarizeContentFunction()
        
        content = """
        Domain-Driven Design is an approach to software development.
        It emphasizes the Repository Pattern for data access.
        The Aggregate pattern helps maintain consistency.
        """
        
        result = await func.run(
            content=content,
            preserve=["Repository Pattern"],
        )
        
        # Summary should mention preserved concept or it should be in invariants
        found = "Repository" in result.summary or any(
            "Repository" in inv for inv in result.invariants
        )
        assert found

    @pytest.mark.asyncio
    async def test_function_tracks_compression_ratio(self) -> None:
        """SummarizeContentFunction tracks compression ratio."""
        from src.functions.summarize_content import SummarizeContentFunction
        
        func = SummarizeContentFunction()
        
        long_content = "This is a longer piece of content. " * 100
        
        result = await func.run(content=long_content)
        
        # Should have compression ratio < 1.0
        assert result.compression_ratio is not None
        assert 0.0 < result.compression_ratio < 1.0


# =============================================================================
# Style Tests
# =============================================================================

class TestSummaryStyle:
    """Tests for summary style variations."""

    @pytest.mark.asyncio
    async def test_technical_style(self) -> None:
        """Technical style produces technical summary."""
        from src.functions.summarize_content import SummarizeContentFunction
        from src.schemas.functions.summarize_content import SummaryStyle
        
        func = SummarizeContentFunction()
        
        result = await func.run(
            content="Technical content about APIs and architecture.",
            style=SummaryStyle.TECHNICAL,
        )
        
        # Should produce prose summary
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    @pytest.mark.asyncio
    async def test_bullets_style(self) -> None:
        """Bullets style produces bullet-point summary."""
        from src.functions.summarize_content import SummarizeContentFunction
        from src.schemas.functions.summarize_content import SummaryStyle
        
        func = SummarizeContentFunction()
        
        result = await func.run(
            content="Content with multiple points. First point. Second point. Third point.",
            style=SummaryStyle.BULLETS,
        )
        
        # Should contain bullet points
        assert "-" in result.summary or "•" in result.summary or "* " in result.summary


# =============================================================================
# Integration Tests
# =============================================================================

class TestSummarizeContentIntegration:
    """Integration tests for summarize_content function."""

    @pytest.mark.asyncio
    async def test_full_workflow(self) -> None:
        """Complete summarization workflow with citations."""
        from src.functions.summarize_content import SummarizeContentFunction
        from src.schemas.citations import SourceMetadata
        from src.schemas.functions.summarize_content import DetailLevel, SummaryStyle
        
        func = SummarizeContentFunction()
        
        content = """
        Domain-Driven Design (DDD) is a software design approach that focuses on 
        modeling software to match a domain according to input from domain experts.
        
        Key concepts include:
        - Ubiquitous Language: A common vocabulary shared by developers and domain experts
        - Bounded Contexts: Explicit boundaries where a domain model applies
        - Aggregates: Clusters of entities treated as a single unit
        - Repositories: Abstractions for data persistence
        
        These patterns help create maintainable and scalable software systems.
        """
        
        sources = [
            SourceMetadata(
                source_type="book",
                title="Domain-Driven Design",
                author="Evans, Eric",
                publisher="Addison-Wesley",
                publication_city="Boston",
                year=2003,
            ),
        ]
        
        result = await func.run(
            content=content,
            sources=sources,
            detail_level=DetailLevel.STANDARD,
            style=SummaryStyle.TECHNICAL,
        )
        
        # Verify output structure
        assert result.summary is not None
        assert len(result.summary) > 0
        assert result.compression_ratio is not None
        assert result.token_count is not None
        
        # Verify citations if sources were provided
        if sources:
            assert isinstance(result.footnotes, list)  # Footnotes is a list (may be empty)

    @pytest.mark.asyncio
    async def test_output_compatible_with_handoff_cache(self) -> None:
        """Output can be stored in HandoffCache."""
        from src.functions.summarize_content import SummarizeContentFunction
        from src.cache import HandoffCache
        
        func = SummarizeContentFunction()
        cache = HandoffCache("test_pipeline")
        
        result = await func.run(content="Content to summarize.")
        
        # Store result in cache
        await cache.set("summary_result", result.model_dump_json())
        
        # Retrieve and verify
        stored = await cache.get("summary_result")
        assert stored is not None
        assert "summary" in stored
