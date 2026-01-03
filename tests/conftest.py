"""Test configuration and shared fixtures.

Pattern: Pytest fixtures, conftest.py
Anti-Pattern Avoided: Fixture reuse without explicit scope
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.core.config import Settings
from src.agents.cross_reference.state import (
    CrossReferenceState,
    SourceChapter,
    TraversalConfig,
    RelationshipType,
)


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with safe defaults."""
    return Settings(
        neo4j_uri="bolt://localhost:7688",
        neo4j_username="test",
        neo4j_password="test",
        qdrant_host="localhost",
        qdrant_port=6334,
        llm_gateway_url="http://localhost:8001",
        debug=True,
        log_level="DEBUG",
    )


# ============================================================================
# State Fixtures
# ============================================================================

@pytest.fixture
def sample_source_chapter() -> SourceChapter:
    """Create a sample source chapter for testing."""
    return SourceChapter(
        book="A Philosophy of Software Design",
        chapter=1,
        title="Introduction",
        tier=1,
        content="Software design is about managing complexity...",
        keywords=["complexity", "abstraction", "modules"],
        concepts=["complexity management", "information hiding"],
    )


@pytest.fixture
def sample_traversal_config() -> TraversalConfig:
    """Create a sample traversal configuration."""
    return TraversalConfig(
        max_hops=3,
        relationship_types=[
            RelationshipType.PARALLEL,
            RelationshipType.PERPENDICULAR,
            RelationshipType.SKIP_TIER,
        ],
        allow_cycles=True,
        min_similarity=0.7,
        max_results_per_tier=10,
    )


@pytest.fixture
def sample_state(
    sample_source_chapter: SourceChapter,
    sample_traversal_config: TraversalConfig,
) -> CrossReferenceState:
    """Create a sample cross-reference state."""
    return CrossReferenceState(
        source=sample_source_chapter,
        config=sample_traversal_config,
        taxonomy_id="ai-ml",
    )


# ============================================================================
# Mock Service Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Create a mock LLM Gateway client."""
    mock = AsyncMock()
    mock.generate.return_value = {
        "response": "Mock LLM response",
        "model": "test-model",
        "tokens_used": 100,
    }
    return mock


@pytest.fixture
def mock_qdrant_client() -> AsyncMock:
    """Create a mock Qdrant client."""
    mock = AsyncMock()
    mock.search.return_value = []
    return mock


@pytest.fixture
def mock_neo4j_client() -> AsyncMock:
    """Create a mock Neo4j client."""
    mock = AsyncMock()
    mock.execute_query.return_value = []
    return mock


# ============================================================================
# FakeClient Protocol Pattern (WBS-AGT1 AC-1.5)
# ============================================================================
# 
# FakeClient pattern provides test doubles that implement service protocols
# without requiring network calls. Use these in unit tests instead of mocking.
#
# Pattern: Protocol Duck Typing
# Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Testing Patterns
# ============================================================================

from typing import Protocol, Any, runtime_checkable
from pydantic import BaseModel


@runtime_checkable
class InferenceClientProtocol(Protocol):
    """Protocol for inference service clients.
    
    Enables duck typing for test doubles without inheritance.
    """
    
    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        preset: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate completion from LLM."""
        ...


@runtime_checkable
class SemanticSearchClientProtocol(Protocol):
    """Protocol for semantic search service clients."""
    
    async def search(
        self,
        query: str,
        collection: str | None = None,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for semantically similar content."""
        ...


class FakeInferenceClient:
    """Fake inference client for testing.
    
    Implements InferenceClientProtocol via duck typing.
    Configure expected responses in constructor.
    
    Example:
        ```python
        client = FakeInferenceClient(
            response="Generated summary of the content."
        )
        result = await client.complete("Summarize this")
        assert result["response"] == "Generated summary of the content."
        ```
    """
    
    def __init__(
        self,
        response: str = "Fake LLM response",
        model: str = "fake-model",
        tokens_used: int = 100,
        should_raise: Exception | None = None,
    ) -> None:
        self._response = response
        self._model = model
        self._tokens_used = tokens_used
        self._should_raise = should_raise
        self._call_count = 0
        self._call_args: list[dict[str, Any]] = []
    
    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        preset: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return canned completion response."""
        # Yield control to event loop (maintains async interface)
        await asyncio.sleep(0)
        
        self._call_count += 1
        self._call_args.append({
            "prompt": prompt,
            "model": model,
            "preset": preset,
            "max_tokens": max_tokens,
        })
        
        if self._should_raise:
            raise self._should_raise
        
        return {
            "response": self._response,
            "model": model or self._model,
            "tokens_used": self._tokens_used,
        }
    
    @property
    def call_count(self) -> int:
        return self._call_count
    
    @property
    def call_args(self) -> list[dict[str, Any]]:
        return self._call_args


class FakeSemanticSearchClient:
    """Fake semantic search client for testing.
    
    Implements SemanticSearchClientProtocol via duck typing.
    Configure expected search results in constructor.
    """
    
    def __init__(
        self,
        results: list[dict[str, Any]] | None = None,
        should_raise: Exception | None = None,
    ) -> None:
        self._results = results or []
        self._should_raise = should_raise
        self._call_count = 0
        self._call_args: list[dict[str, Any]] = []
    
    async def search(
        self,
        query: str,
        collection: str | None = None,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return canned search results."""
        # Yield control to event loop (maintains async interface)
        await asyncio.sleep(0)
        
        self._call_count += 1
        self._call_args.append({
            "query": query,
            "collection": collection,
            "limit": limit,
            "filters": filters,
        })
        
        if self._should_raise:
            raise self._should_raise
        
        return self._results[:limit]
    
    @property
    def call_count(self) -> int:
        return self._call_count
    
    @property
    def call_args(self) -> list[dict[str, Any]]:
        return self._call_args


@pytest.fixture
def fake_inference_client() -> FakeInferenceClient:
    """Create a FakeInferenceClient for testing.
    
    Returns a fake client that passes protocol type checking.
    """
    return FakeInferenceClient()


@pytest.fixture
def fake_semantic_search_client() -> FakeSemanticSearchClient:
    """Create a FakeSemanticSearchClient for testing.
    
    Returns a fake client that passes protocol type checking.
    """
    return FakeSemanticSearchClient(
        results=[
            {
                "id": "doc_001",
                "content": "Related content about software design",
                "score": 0.95,
                "metadata": {"book": "A Philosophy of Software Design", "chapter": 1},
            },
            {
                "id": "doc_002",
                "content": "Information about design patterns",
                "score": 0.88,
                "metadata": {"book": "Design Patterns", "chapter": 3},
            },
        ]
    )


# ============================================================================
# Event Loop Configuration (for async tests)
# ============================================================================

@pytest.fixture(scope="session")
def event_loop_policy():
    """Use the default event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
