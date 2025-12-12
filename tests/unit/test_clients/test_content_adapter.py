"""Unit tests for SemanticSearchContentAdapter.

TDD Phase: Tests for the adapter pattern connecting 
SemanticSearchClient to ContentClient protocol.

Pattern: Adapter Pattern verification
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.clients.content_adapter import SemanticSearchContentAdapter


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_semantic_client() -> AsyncMock:
    """Create a mock SemanticSearchClient."""
    return AsyncMock()


@pytest.fixture
def adapter(mock_semantic_client: AsyncMock) -> SemanticSearchContentAdapter:
    """Create an adapter with mocked client."""
    return SemanticSearchContentAdapter(mock_semantic_client)


# ==============================================================================
# SemanticSearchContentAdapter Tests
# ==============================================================================


class TestSemanticSearchContentAdapter:
    """Tests for the content client adapter."""

    @pytest.mark.asyncio
    async def test_adapter_maps_parameters(
        self,
        adapter: SemanticSearchContentAdapter,
        mock_semantic_client: AsyncMock,
    ) -> None:
        """Test that adapter correctly maps book->book_id, chapter->chapter_number."""
        mock_semantic_client.get_chapter_content.return_value = {
            "book_id": "Test_Book",
            "chapter_number": 5,
            "title": "Test Title",
            "summary": "Test content",
            "keywords": ["test"],
            "concepts": ["testing"],
            "page_range": "1-20",
            "found": True,
        }
        
        await adapter.get_chapter_content(book="Test_Book", chapter=5)
        
        # Verify correct parameter mapping
        mock_semantic_client.get_chapter_content.assert_called_once_with(
            book_id="Test_Book",
            chapter_number=5,
        )

    @pytest.mark.asyncio
    async def test_adapter_maps_response(
        self,
        adapter: SemanticSearchContentAdapter,
        mock_semantic_client: AsyncMock,
    ) -> None:
        """Test that adapter maps summary to content."""
        mock_semantic_client.get_chapter_content.return_value = {
            "book_id": "Test_Book",
            "chapter_number": 5,
            "title": "Chapter Title",
            "summary": "This is the chapter summary/content...",
            "keywords": ["a", "b"],
            "concepts": ["x", "y"],
            "page_range": "89-134",
            "found": True,
        }
        
        result = await adapter.get_chapter_content(book="Test_Book", chapter=5)
        
        assert result is not None
        # content field should map from summary
        assert result["content"] == "This is the chapter summary/content..."
        assert result["page_range"] == "89-134"
        assert result["title"] == "Chapter Title"
        assert result["keywords"] == ["a", "b"]
        assert result["concepts"] == ["x", "y"]

    @pytest.mark.asyncio
    async def test_adapter_returns_none_when_not_found(
        self,
        adapter: SemanticSearchContentAdapter,
        mock_semantic_client: AsyncMock,
    ) -> None:
        """Test that adapter returns None when chapter not found."""
        mock_semantic_client.get_chapter_content.return_value = None
        
        result = await adapter.get_chapter_content(book="Missing", chapter=999)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_adapter_handles_empty_fields(
        self,
        adapter: SemanticSearchContentAdapter,
        mock_semantic_client: AsyncMock,
    ) -> None:
        """Test handling of missing/empty fields in response."""
        mock_semantic_client.get_chapter_content.return_value = {
            "book_id": "Test",
            "chapter_number": 1,
            "found": True,
            # Missing summary, title, keywords, concepts, page_range
        }
        
        result = await adapter.get_chapter_content(book="Test", chapter=1)
        
        assert result is not None
        assert result["content"] == ""
        assert result["page_range"] == ""
        assert result["title"] == ""
        assert result["keywords"] == []
        assert result["concepts"] == []


class TestAdapterProtocolCompatibility:
    """Tests verifying adapter meets ContentClient protocol."""

    @pytest.mark.asyncio
    async def test_adapter_signature_matches_protocol(
        self,
        adapter: SemanticSearchContentAdapter,
    ) -> None:
        """Test that adapter has correct method signature."""
        # Verify the method exists and is async
        assert hasattr(adapter, "get_chapter_content")
        import inspect
        assert inspect.iscoroutinefunction(adapter.get_chapter_content)

    @pytest.mark.asyncio
    async def test_adapter_can_be_used_as_content_client(
        self,
        mock_semantic_client: AsyncMock,
    ) -> None:
        """Test adapter can be injected into retrieve_content node."""
        from src.agents.cross_reference.nodes.retrieve_content import set_content_client
        
        adapter = SemanticSearchContentAdapter(mock_semantic_client)
        
        # This should not raise - adapter implements ContentClient protocol
        set_content_client(adapter)
        
        # Reset after test
        set_content_client(None)
