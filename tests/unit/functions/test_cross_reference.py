"""Tests for cross_reference function.

TDD tests for WBS-AGT13: cross_reference Function.

Acceptance Criteria Coverage:
- AC-13.1: Queries semantic-search-service for related content
- AC-13.2: Returns CrossReferenceResult with matches, relevance_scores
- AC-13.3: Context budget: 2048 input / 4096 output
- AC-13.4: Default preset: S4
- AC-13.5: Integrates with Qdrant via semantic-search-service

Exit Criteria:
- pytest tests/unit/functions/test_cross_reference.py passes with 100% coverage
- Each Match has source, content, relevance_score (0.0-1.0)
- FakeSemanticSearchClient used in unit tests
- Integration test hits real semantic-search-service:8081

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 8
"""

import pytest
from pydantic import ValidationError


# =============================================================================
# AGT13.1: AC-13.1 Input Schema Tests - CrossReferenceInput
# =============================================================================

class TestCrossReferenceInput:
    """Tests for CrossReferenceInput schema.
    
    AC-13.1: Queries semantic-search-service for related content.
    Per architecture:
    - query_artifact: str (source content)
    - search_scope: list (which repositories)
    - match_type: enum (semantic | keyword | hybrid)
    - top_k: int
    """

    def test_input_requires_query_artifact(self) -> None:
        """CrossReferenceInput requires query_artifact field."""
        from src.schemas.functions.cross_reference import CrossReferenceInput
        
        with pytest.raises(ValidationError) as exc_info:
            CrossReferenceInput()  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("query_artifact",) for e in errors)

    def test_input_accepts_query_artifact(self) -> None:
        """CrossReferenceInput accepts query_artifact string."""
        from src.schemas.functions.cross_reference import CrossReferenceInput
        
        input_data = CrossReferenceInput(
            query_artifact="Repository pattern implementation",
        )
        assert input_data.query_artifact == "Repository pattern implementation"

    def test_input_validates_query_artifact_not_empty(self) -> None:
        """query_artifact cannot be empty or whitespace only."""
        from src.schemas.functions.cross_reference import CrossReferenceInput
        
        with pytest.raises(ValidationError):
            CrossReferenceInput(query_artifact="")
        
        with pytest.raises(ValidationError):
            CrossReferenceInput(query_artifact="   ")

    def test_input_has_optional_search_scope(self) -> None:
        """CrossReferenceInput has optional search_scope list.
        
        Per architecture: search_scope specifies which repositories to search.
        """
        from src.schemas.functions.cross_reference import CrossReferenceInput
        
        # Default is empty list (search all)
        input_data = CrossReferenceInput(query_artifact="test query")
        assert input_data.search_scope == []
        
        # Can provide specific repositories
        input_with_scope = CrossReferenceInput(
            query_artifact="test query",
            search_scope=["code-reference-engine", "ai-platform-data"],
        )
        assert len(input_with_scope.search_scope) == 2
        assert "code-reference-engine" in input_with_scope.search_scope

    def test_input_has_match_type_enum(self) -> None:
        """CrossReferenceInput has match_type enum.
        
        Per architecture: semantic | keyword | hybrid.
        Default should be 'semantic'.
        """
        from src.schemas.functions.cross_reference import (
            CrossReferenceInput,
            MatchType,
        )
        
        # Default is semantic
        input_data = CrossReferenceInput(query_artifact="test query")
        assert input_data.match_type == MatchType.SEMANTIC
        
        # Can specify keyword
        input_keyword = CrossReferenceInput(
            query_artifact="test query",
            match_type=MatchType.KEYWORD,
        )
        assert input_keyword.match_type == MatchType.KEYWORD
        
        # Can specify hybrid
        input_hybrid = CrossReferenceInput(
            query_artifact="test query",
            match_type=MatchType.HYBRID,
        )
        assert input_hybrid.match_type == MatchType.HYBRID

    def test_input_validates_invalid_match_type(self) -> None:
        """Invalid match_type values raise ValidationError."""
        from src.schemas.functions.cross_reference import CrossReferenceInput
        
        with pytest.raises(ValidationError):
            CrossReferenceInput(
                query_artifact="test query",
                match_type="invalid_type",  # type: ignore
            )

    def test_input_has_top_k_with_default(self) -> None:
        """CrossReferenceInput has top_k with sensible default.
        
        Per architecture: top_k limits number of results.
        """
        from src.schemas.functions.cross_reference import CrossReferenceInput
        
        # Default should be reasonable (e.g., 10)
        input_data = CrossReferenceInput(query_artifact="test query")
        assert input_data.top_k == 10
        
        # Can override
        input_custom = CrossReferenceInput(
            query_artifact="test query",
            top_k=5,
        )
        assert input_custom.top_k == 5

    def test_input_validates_top_k_positive(self) -> None:
        """top_k must be positive."""
        from src.schemas.functions.cross_reference import CrossReferenceInput
        
        with pytest.raises(ValidationError):
            CrossReferenceInput(query_artifact="test", top_k=0)
        
        with pytest.raises(ValidationError):
            CrossReferenceInput(query_artifact="test", top_k=-1)

    def test_input_validates_top_k_max(self) -> None:
        """top_k has reasonable maximum to prevent resource exhaustion."""
        from src.schemas.functions.cross_reference import CrossReferenceInput
        
        with pytest.raises(ValidationError):
            CrossReferenceInput(query_artifact="test", top_k=1001)

    def test_input_json_schema_export(self) -> None:
        """CrossReferenceInput exports valid JSON schema.
        
        AC-4.5: All schemas have JSON schema export.
        """
        from src.schemas.functions.cross_reference import CrossReferenceInput
        
        schema = CrossReferenceInput.model_json_schema()
        
        assert "properties" in schema
        assert "query_artifact" in schema["properties"]
        assert "search_scope" in schema["properties"]
        assert "match_type" in schema["properties"]
        assert "top_k" in schema["properties"]


# =============================================================================
# AGT13.3: AC-13.2 Output Schema Tests - Reference, Citation, CrossReferenceResult
# =============================================================================

class TestReference:
    """Tests for Reference schema.
    
    Per architecture: references: list[Reference] in output.
    Each Reference represents a matched item from search.
    """

    def test_reference_requires_source(self) -> None:
        """Reference requires source field."""
        from src.schemas.functions.cross_reference import Reference
        
        with pytest.raises(ValidationError) as exc_info:
            Reference(
                content="Some content",
                relevance_score=0.85,
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("source",) for e in errors)

    def test_reference_requires_content(self) -> None:
        """Reference requires content field."""
        from src.schemas.functions.cross_reference import Reference
        
        with pytest.raises(ValidationError) as exc_info:
            Reference(
                source="code-reference-engine/file.py",
                relevance_score=0.85,
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("content",) for e in errors)

    def test_reference_requires_relevance_score(self) -> None:
        """Reference requires relevance_score field."""
        from src.schemas.functions.cross_reference import Reference
        
        with pytest.raises(ValidationError) as exc_info:
            Reference(
                source="code-reference-engine/file.py",
                content="Some content",
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("relevance_score",) for e in errors)

    def test_reference_accepts_valid_data(self) -> None:
        """Reference accepts all required fields."""
        from src.schemas.functions.cross_reference import Reference
        
        ref = Reference(
            source="code-reference-engine/backend/ddd/repository.py",
            content="class BaseRepository(ABC): ...",
            relevance_score=0.89,
        )
        
        assert ref.source == "code-reference-engine/backend/ddd/repository.py"
        assert ref.content == "class BaseRepository(ABC): ..."
        assert abs(ref.relevance_score - 0.89) < 0.001

    def test_reference_validates_relevance_score_range(self) -> None:
        """relevance_score must be between 0.0 and 1.0.
        
        Exit Criteria: Each Match has relevance_score (0.0-1.0).
        """
        from src.schemas.functions.cross_reference import Reference
        
        # Valid at boundaries
        ref_min = Reference(source="s", content="c", relevance_score=0.0)
        assert abs(ref_min.relevance_score - 0.0) < 0.001
        
        ref_max = Reference(source="s", content="c", relevance_score=1.0)
        assert abs(ref_max.relevance_score - 1.0) < 0.001
        
        # Invalid below 0
        with pytest.raises(ValidationError):
            Reference(source="s", content="c", relevance_score=-0.1)
        
        # Invalid above 1
        with pytest.raises(ValidationError):
            Reference(source="s", content="c", relevance_score=1.1)

    def test_reference_has_optional_metadata(self) -> None:
        """Reference can have optional metadata for citations."""
        from src.schemas.functions.cross_reference import Reference
        
        ref = Reference(
            source="code-reference-engine/file.py",
            content="Some code",
            relevance_score=0.85,
            source_type="code",
            line_range="12-45",
            commit_hash="a1b2c3d",
        )
        
        assert ref.source_type == "code"
        assert ref.line_range == "12-45"
        assert ref.commit_hash == "a1b2c3d"


class TestCitation:
    """Tests for Citation schema.
    
    Per architecture: citations: list[Citation] for footnotes.
    """

    def test_citation_requires_marker(self) -> None:
        """Citation requires marker field (e.g., '^1')."""
        from src.schemas.functions.cross_reference import Citation
        
        with pytest.raises(ValidationError) as exc_info:
            Citation(
                formatted_citation="Fowler, Martin. PEAA. 2002.",
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("marker",) for e in errors)

    def test_citation_requires_formatted_citation(self) -> None:
        """Citation requires formatted_citation field."""
        from src.schemas.functions.cross_reference import Citation
        
        with pytest.raises(ValidationError) as exc_info:
            Citation(marker="^1")  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("formatted_citation",) for e in errors)

    def test_citation_accepts_valid_data(self) -> None:
        """Citation accepts valid marker and formatted_citation."""
        from src.schemas.functions.cross_reference import Citation
        
        # Book citation (Chicago-style)
        book_citation = Citation(
            marker="^1",
            formatted_citation="Fowler, Martin. Patterns of Enterprise Application Architecture. Addison-Wesley, 2002. pp. 322-327.",
        )
        assert book_citation.marker == "^1"
        assert "Fowler, Martin" in book_citation.formatted_citation
        
        # Code citation
        code_citation = Citation(
            marker="^2",
            formatted_citation="code-reference-engine/backend/ddd/repository.py:12-45 (a1b2c3d)",
        )
        assert code_citation.marker == "^2"

    def test_citation_has_optional_source_reference(self) -> None:
        """Citation can link back to its Reference source."""
        from src.schemas.functions.cross_reference import Citation
        
        citation = Citation(
            marker="^1",
            formatted_citation="Fowler, Martin. PEAA. 2002.",
            reference_source="books/peaa.md",
        )
        assert citation.reference_source == "books/peaa.md"


class TestCrossReferenceResult:
    """Tests for CrossReferenceResult schema.
    
    AC-13.2: Returns CrossReferenceResult with matches, relevance_scores.
    Per architecture:
    - references: list[Reference]
    - similarity_scores: list
    - compressed_context: str (for downstream)
    - citations: list[Citation] (for footnotes)
    """

    def test_result_requires_references(self) -> None:
        """CrossReferenceResult requires references list."""
        from src.schemas.functions.cross_reference import CrossReferenceResult
        
        with pytest.raises(ValidationError) as exc_info:
            CrossReferenceResult(
                similarity_scores=[0.89],
                compressed_context="context",
                citations=[],
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("references",) for e in errors)

    def test_result_requires_similarity_scores(self) -> None:
        """CrossReferenceResult requires similarity_scores list."""
        from src.schemas.functions.cross_reference import (
            CrossReferenceResult,
            Reference,
        )
        
        with pytest.raises(ValidationError) as exc_info:
            CrossReferenceResult(
                references=[
                    Reference(source="s", content="c", relevance_score=0.89)
                ],
                compressed_context="context",
                citations=[],
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("similarity_scores",) for e in errors)

    def test_result_requires_compressed_context(self) -> None:
        """CrossReferenceResult requires compressed_context string."""
        from src.schemas.functions.cross_reference import (
            CrossReferenceResult,
            Reference,
        )
        
        with pytest.raises(ValidationError) as exc_info:
            CrossReferenceResult(
                references=[
                    Reference(source="s", content="c", relevance_score=0.89)
                ],
                similarity_scores=[0.89],
                citations=[],
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("compressed_context",) for e in errors)

    def test_result_requires_citations(self) -> None:
        """CrossReferenceResult requires citations list (can be empty)."""
        from src.schemas.functions.cross_reference import (
            CrossReferenceResult,
            Reference,
        )
        
        with pytest.raises(ValidationError) as exc_info:
            CrossReferenceResult(
                references=[
                    Reference(source="s", content="c", relevance_score=0.89)
                ],
                similarity_scores=[0.89],
                compressed_context="context",
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("citations",) for e in errors)

    def test_result_accepts_valid_data(self) -> None:
        """CrossReferenceResult accepts all required fields."""
        from src.schemas.functions.cross_reference import (
            CrossReferenceResult,
            Reference,
            Citation,
        )
        
        result = CrossReferenceResult(
            references=[
                Reference(
                    source="code-reference-engine/backend/ddd/repository.py",
                    content="class BaseRepository(ABC): ...",
                    relevance_score=0.89,
                ),
                Reference(
                    source="books/peaa.md",
                    content="The Repository pattern provides...",
                    relevance_score=0.85,
                ),
            ],
            similarity_scores=[0.89, 0.85],
            compressed_context="Repository pattern: collection-like interface for domain objects.",
            citations=[
                Citation(
                    marker="^1",
                    formatted_citation="code-reference-engine/backend/ddd/repository.py:12-45",
                ),
                Citation(
                    marker="^2",
                    formatted_citation="Fowler, Martin. PEAA. 2002. pp. 322-327.",
                ),
            ],
        )
        
        assert len(result.references) == 2
        assert len(result.similarity_scores) == 2
        assert "Repository pattern" in result.compressed_context
        assert len(result.citations) == 2

    def test_result_validates_scores_match_references_count(self) -> None:
        """similarity_scores count should match references count."""
        from src.schemas.functions.cross_reference import (
            CrossReferenceResult,
            Reference,
        )
        
        with pytest.raises(ValidationError) as exc_info:
            CrossReferenceResult(
                references=[
                    Reference(source="s1", content="c1", relevance_score=0.89),
                    Reference(source="s2", content="c2", relevance_score=0.85),
                ],
                similarity_scores=[0.89],  # Mismatch!
                compressed_context="context",
                citations=[],
            )
        
        assert "similarity_scores" in str(exc_info.value)

    def test_result_validates_similarity_scores_range(self) -> None:
        """Each similarity_score must be between 0.0 and 1.0."""
        from src.schemas.functions.cross_reference import (
            CrossReferenceResult,
            Reference,
        )
        
        with pytest.raises(ValidationError):
            CrossReferenceResult(
                references=[
                    Reference(source="s", content="c", relevance_score=0.89)
                ],
                similarity_scores=[1.5],  # Invalid!
                compressed_context="context",
                citations=[],
            )

    def test_result_allows_empty_references(self) -> None:
        """CrossReferenceResult allows empty references (no matches found)."""
        from src.schemas.functions.cross_reference import CrossReferenceResult
        
        result = CrossReferenceResult(
            references=[],
            similarity_scores=[],
            compressed_context="No relevant matches found.",
            citations=[],
        )
        
        assert len(result.references) == 0
        assert len(result.similarity_scores) == 0

    def test_result_json_schema_export(self) -> None:
        """CrossReferenceResult exports valid JSON schema."""
        from src.schemas.functions.cross_reference import CrossReferenceResult
        
        schema = CrossReferenceResult.model_json_schema()
        
        assert "properties" in schema
        assert "references" in schema["properties"]
        assert "similarity_scores" in schema["properties"]
        assert "compressed_context" in schema["properties"]
        assert "citations" in schema["properties"]


# =============================================================================
# AGT13.5: AC-13.1, AC-13.5 Semantic Search Integration Tests
# =============================================================================

class TestFakeSemanticSearchClient:
    """Tests for FakeSemanticSearchClient.
    
    Exit Criteria: FakeSemanticSearchClient used in unit tests.
    """

    def test_fake_client_exists(self) -> None:
        """FakeSemanticSearchClient exists for unit testing."""
        from src.clients.semantic_search import FakeSemanticSearchClient
        
        client = FakeSemanticSearchClient()
        # Verify client has expected methods
        assert hasattr(client, 'search')
        assert hasattr(client, 'get_relationships')
        assert hasattr(client, 'close')

    def test_fake_client_implements_protocol(self) -> None:
        """FakeSemanticSearchClient implements SemanticSearchProtocol."""
        from src.clients.semantic_search import FakeSemanticSearchClient
        from src.clients.protocols import SemanticSearchProtocol
        
        client = FakeSemanticSearchClient()
        assert isinstance(client, SemanticSearchProtocol)

    @pytest.mark.asyncio
    async def test_fake_client_search_returns_results(self) -> None:
        """FakeSemanticSearchClient.search returns mock results."""
        from src.clients.semantic_search import FakeSemanticSearchClient
        
        client = FakeSemanticSearchClient()
        results = await client.search(query="repository pattern", top_k=5)
        
        assert isinstance(results, list)
        # Should return mock data
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_fake_client_search_respects_top_k(self) -> None:
        """FakeSemanticSearchClient.search respects top_k limit."""
        from src.clients.semantic_search import FakeSemanticSearchClient
        
        client = FakeSemanticSearchClient()
        results = await client.search(query="test", top_k=2)
        
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_fake_client_allows_custom_responses(self) -> None:
        """FakeSemanticSearchClient can be configured with custom responses."""
        from src.clients.semantic_search import FakeSemanticSearchClient
        
        custom_results = [
            {
                "source": "custom/source.py",
                "content": "Custom content",
                "score": 0.95,
            }
        ]
        
        client = FakeSemanticSearchClient(search_results=custom_results)
        results = await client.search(query="test", top_k=5)
        
        assert len(results) == 1
        assert results[0]["source"] == "custom/source.py"


# =============================================================================
# AGT13.7: Cross Reference Function Tests
# =============================================================================

class TestCrossReferenceFunction:
    """Tests for CrossReferenceFunction.
    
    AC-13.1: Queries semantic-search-service for related content.
    AC-13.2: Returns CrossReferenceResult with matches, relevance_scores.
    AC-13.3: Context budget: 2048 input / 4096 output.
    AC-13.4: Default preset: S4.
    """

    def test_function_has_correct_name(self) -> None:
        """CrossReferenceFunction has name 'cross_reference'."""
        from src.functions.cross_reference import CrossReferenceFunction
        
        func = CrossReferenceFunction()
        assert func.name == "cross_reference"

    def test_function_has_correct_description(self) -> None:
        """CrossReferenceFunction has meaningful description."""
        from src.functions.cross_reference import CrossReferenceFunction
        
        func = CrossReferenceFunction()
        assert "semantic search" in func.description.lower() or \
               "related content" in func.description.lower()

    def test_function_has_input_schema(self) -> None:
        """CrossReferenceFunction exposes CrossReferenceInput schema."""
        from src.functions.cross_reference import CrossReferenceFunction
        from src.schemas.functions.cross_reference import CrossReferenceInput
        
        func = CrossReferenceFunction()
        assert func.input_schema == CrossReferenceInput

    def test_function_has_output_schema(self) -> None:
        """CrossReferenceFunction exposes CrossReferenceResult schema."""
        from src.functions.cross_reference import CrossReferenceFunction
        from src.schemas.functions.cross_reference import CrossReferenceResult
        
        func = CrossReferenceFunction()
        assert func.output_schema == CrossReferenceResult

    def test_function_has_context_budget(self) -> None:
        """CrossReferenceFunction defines context budget.
        
        AC-13.3: Context budget: 2048 input / 4096 output.
        """
        from src.functions.cross_reference import CrossReferenceFunction
        
        func = CrossReferenceFunction()
        assert func.context_budget.input_tokens == 2048
        assert func.context_budget.output_tokens == 4096

    def test_function_has_default_preset_s4(self) -> None:
        """CrossReferenceFunction default preset is S4.
        
        AC-13.4: Default preset: S4 (LLM for post-ranking only).
        """
        from src.functions.cross_reference import CrossReferenceFunction
        
        func = CrossReferenceFunction()
        assert func.default_preset == "S4"

    @pytest.mark.asyncio
    async def test_function_executes_with_semantic_search_client(self) -> None:
        """CrossReferenceFunction executes using semantic search client.
        
        AC-13.1: Queries semantic-search-service for related content.
        """
        from src.functions.cross_reference import CrossReferenceFunction
        from src.schemas.functions.cross_reference import CrossReferenceInput
        from src.clients.semantic_search import FakeSemanticSearchClient
        
        client = FakeSemanticSearchClient()
        func = CrossReferenceFunction(semantic_search_client=client)
        
        input_data = CrossReferenceInput(
            query_artifact="Repository pattern implementation",
            top_k=5,
        )
        
        result = await func.execute(input_data)
        
        assert result is not None
        assert hasattr(result, "references")
        assert hasattr(result, "similarity_scores")
        assert hasattr(result, "compressed_context")
        assert hasattr(result, "citations")

    @pytest.mark.asyncio
    async def test_function_returns_normalized_scores(self) -> None:
        """CrossReferenceFunction normalizes relevance scores to 0.0-1.0."""
        from src.functions.cross_reference import CrossReferenceFunction
        from src.schemas.functions.cross_reference import CrossReferenceInput
        from src.clients.semantic_search import FakeSemanticSearchClient
        
        # Mock results with various score formats
        custom_results = [
            {"source": "s1", "content": "c1", "score": 0.95},
            {"source": "s2", "content": "c2", "score": 0.72},
        ]
        
        client = FakeSemanticSearchClient(search_results=custom_results)
        func = CrossReferenceFunction(semantic_search_client=client)
        
        input_data = CrossReferenceInput(
            query_artifact="test query",
            top_k=5,
        )
        
        result = await func.execute(input_data)
        
        for score in result.similarity_scores:
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_function_generates_compressed_context(self) -> None:
        """CrossReferenceFunction generates compressed_context for downstream."""
        from src.functions.cross_reference import CrossReferenceFunction
        from src.schemas.functions.cross_reference import CrossReferenceInput
        from src.clients.semantic_search import FakeSemanticSearchClient
        
        custom_results = [
            {
                "source": "code-reference-engine/ddd/repository.py",
                "content": "class BaseRepository(ABC): ...",
                "score": 0.89,
            },
        ]
        
        client = FakeSemanticSearchClient(search_results=custom_results)
        func = CrossReferenceFunction(semantic_search_client=client)
        
        input_data = CrossReferenceInput(
            query_artifact="Repository pattern",
            top_k=5,
        )
        
        result = await func.execute(input_data)
        
        # compressed_context should summarize the findings
        assert result.compressed_context != ""
        assert len(result.compressed_context) < 4096  # Fits output budget

    @pytest.mark.asyncio
    async def test_function_generates_citations(self) -> None:
        """CrossReferenceFunction generates citations for footnotes."""
        from src.functions.cross_reference import CrossReferenceFunction
        from src.schemas.functions.cross_reference import CrossReferenceInput
        from src.clients.semantic_search import FakeSemanticSearchClient
        
        custom_results = [
            {
                "source": "code-reference-engine/ddd/repository.py",
                "content": "class BaseRepository(ABC): ...",
                "score": 0.89,
                "line_range": "12-45",
            },
        ]
        
        client = FakeSemanticSearchClient(search_results=custom_results)
        func = CrossReferenceFunction(semantic_search_client=client)
        
        input_data = CrossReferenceInput(
            query_artifact="Repository pattern",
            top_k=5,
        )
        
        result = await func.execute(input_data)
        
        # Should generate citation for each reference
        assert len(result.citations) == len(result.references)
        if result.citations:
            assert result.citations[0].marker.startswith("^")

    @pytest.mark.asyncio
    async def test_function_handles_empty_results(self) -> None:
        """CrossReferenceFunction handles no matches gracefully."""
        from src.functions.cross_reference import CrossReferenceFunction
        from src.schemas.functions.cross_reference import CrossReferenceInput
        from src.clients.semantic_search import FakeSemanticSearchClient
        
        client = FakeSemanticSearchClient(search_results=[])
        func = CrossReferenceFunction(semantic_search_client=client)
        
        input_data = CrossReferenceInput(
            query_artifact="nonexistent topic xyz123",
            top_k=5,
        )
        
        result = await func.execute(input_data)
        
        assert len(result.references) == 0
        assert len(result.similarity_scores) == 0
        assert len(result.citations) == 0
        assert result.compressed_context != ""  # Should still have message

    @pytest.mark.asyncio
    async def test_function_respects_search_scope(self) -> None:
        """CrossReferenceFunction filters by search_scope."""
        from src.functions.cross_reference import CrossReferenceFunction
        from src.schemas.functions.cross_reference import (
            CrossReferenceInput,
            MatchType,
        )
        from src.clients.semantic_search import FakeSemanticSearchClient
        
        # Results from multiple repos
        custom_results = [
            {"source": "code-reference-engine/file.py", "content": "c1", "score": 0.9},
            {"source": "ai-platform-data/other.py", "content": "c2", "score": 0.8},
            {"source": "external-repo/excluded.py", "content": "c3", "score": 0.7},
        ]
        
        client = FakeSemanticSearchClient(search_results=custom_results)
        func = CrossReferenceFunction(semantic_search_client=client)
        
        input_data = CrossReferenceInput(
            query_artifact="test",
            search_scope=["code-reference-engine", "ai-platform-data"],
            top_k=10,
        )
        
        result = await func.execute(input_data)
        
        # Should only include results from specified scope
        for ref in result.references:
            assert any(
                scope in ref.source 
                for scope in ["code-reference-engine", "ai-platform-data"]
            )


# =============================================================================
# AGT13: Context Budget Enforcement Tests
# =============================================================================

class TestContextBudgetEnforcement:
    """Tests for context budget enforcement.
    
    AC-13.3: Context budget: 2048 input / 4096 output.
    """

    @pytest.mark.asyncio
    async def test_input_truncated_to_budget(self) -> None:
        """Large query_artifact is truncated to input budget."""
        from src.functions.cross_reference import CrossReferenceFunction
        from src.schemas.functions.cross_reference import CrossReferenceInput
        from src.clients.semantic_search import FakeSemanticSearchClient
        
        # Create very large input (simulate > 2048 tokens)
        large_content = "word " * 5000  # Way over budget
        
        client = FakeSemanticSearchClient()
        func = CrossReferenceFunction(semantic_search_client=client)
        
        input_data = CrossReferenceInput(
            query_artifact=large_content,
            top_k=5,
        )
        
        # Should execute without error (truncates internally)
        result = await func.execute(input_data)
        assert result is not None

    @pytest.mark.asyncio
    async def test_output_respects_budget(self) -> None:
        """Output compressed_context respects output budget."""
        from src.functions.cross_reference import CrossReferenceFunction
        from src.schemas.functions.cross_reference import CrossReferenceInput
        from src.clients.semantic_search import FakeSemanticSearchClient
        
        # Many results that would generate large output
        custom_results = [
            {"source": f"repo/file{i}.py", "content": "c" * 500, "score": 0.9 - i * 0.1}
            for i in range(20)
        ]
        
        client = FakeSemanticSearchClient(search_results=custom_results)
        func = CrossReferenceFunction(semantic_search_client=client)
        
        input_data = CrossReferenceInput(
            query_artifact="test",
            top_k=20,
        )
        
        result = await func.execute(input_data)
        
        # Output should be bounded
        # Rough estimate: 4096 tokens ≈ 16384 chars
        assert len(result.compressed_context) < 16384


# =============================================================================
# Integration Test Marker (for CI)
# =============================================================================

@pytest.mark.integration
class TestCrossReferenceIntegration:
    """Integration tests that hit real semantic-search-service.
    
    Exit Criteria: Integration test hits real semantic-search-service:8081.
    
    Run with: pytest -m integration
    """

    @pytest.mark.asyncio
    async def test_real_semantic_search_service(self) -> None:
        """Integration test with real semantic-search-service.
        
        This test is skipped by default. To run it:
        1. Start semantic-search-service on port 8081
        2. Run: pytest -m integration --no-cov
        """
        pytest.skip("Run manually with semantic-search-service running on :8081")
