"""Test configuration and shared fixtures.

Pattern: Pytest fixtures, conftest.py
Anti-Pattern Avoided: Fixture reuse without explicit scope
"""

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
# Event Loop Configuration (for async tests)
# ============================================================================

@pytest.fixture(scope="session")
def event_loop_policy():
    """Use the default event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
