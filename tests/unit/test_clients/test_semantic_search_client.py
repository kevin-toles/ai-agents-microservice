"""Unit tests for SemanticSearchClient.get_chapter_content method.

TDD Phase: Tests for content retrieval via Kitchen Brigade architecture.
ai-agents (Expeditor) → semantic-search (Cookbook) → Neo4j (Pantry)

Reference: CODING_PATTERNS_ANALYSIS.md Anti-Pattern #12 (Connection Pooling)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from src.core.clients.semantic_search import SemanticSearchClient


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_response_found() -> dict:
    """Mock response for found chapter."""
    return {
        "book_id": "Architecture_Patterns_with_Python",
        "chapter_number": 5,
        "title": "TDD, DDD, and Event-Driven Architecture",
        "summary": "This chapter explores test-driven development patterns...",
        "keywords": ["TDD", "DDD", "event-driven", "CQRS", "aggregate"],
        "concepts": ["hexagonal architecture", "ports and adapters"],
        "page_range": "89-134",
        "found": True,
    }


@pytest.fixture
def mock_response_not_found() -> dict:
    """Mock response for not found chapter."""
    return {
        "book_id": "NonExistent_Book",
        "chapter_number": 999,
        "title": "",
        "summary": "",
        "keywords": [],
        "concepts": [],
        "page_range": "",
        "found": False,
    }


@pytest.fixture
def client() -> SemanticSearchClient:
    """Create a client instance for testing."""
    return SemanticSearchClient(
        base_url="http://test-semantic-search:8081",
        timeout=10.0,
    )


# ==============================================================================
# get_chapter_content Tests
# ==============================================================================


class TestGetChapterContent:
    """Tests for SemanticSearchClient.get_chapter_content method."""

    @pytest.mark.asyncio
    async def test_get_chapter_found(
        self,
        client: SemanticSearchClient,
        mock_response_found: dict,
    ) -> None:
        """Test successful retrieval of existing chapter."""
        # Mock httpx client response
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response_found
        mock_http_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(get=AsyncMock(return_value=mock_http_response)),
        ):
            result = await client.get_chapter_content(
                book_id="Architecture_Patterns_with_Python",
                chapter_number=5,
            )

        assert result is not None
        assert result["book_id"] == "Architecture_Patterns_with_Python"
        assert result["chapter_number"] == 5
        assert result["found"] is True
        assert "TDD" in result["keywords"]

    @pytest.mark.asyncio
    async def test_get_chapter_not_found_via_found_flag(
        self,
        client: SemanticSearchClient,
        mock_response_not_found: dict,
    ) -> None:
        """Test response when chapter not found (found=False in response)."""
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response_not_found
        mock_http_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(get=AsyncMock(return_value=mock_http_response)),
        ):
            result = await client.get_chapter_content(
                book_id="NonExistent_Book",
                chapter_number=999,
            )

        assert result is None  # Returns None when found=False

    @pytest.mark.asyncio
    async def test_get_chapter_404_response(
        self,
        client: SemanticSearchClient,
    ) -> None:
        """Test handling of HTTP 404 response."""
        mock_http_response = MagicMock()
        mock_http_response.status_code = 404
        mock_http_response.text = "Not found"
        
        # Create HTTP error
        http_error = httpx.HTTPStatusError(
            message="Not found",
            request=MagicMock(),
            response=mock_http_response,
        )
        mock_http_response.raise_for_status.side_effect = http_error

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(get=AsyncMock(return_value=mock_http_response)),
        ):
            result = await client.get_chapter_content(
                book_id="NonExistent_Book",
                chapter_number=999,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_chapter_500_response(
        self,
        client: SemanticSearchClient,
    ) -> None:
        """Test handling of HTTP 500 response."""
        mock_http_response = MagicMock()
        mock_http_response.status_code = 500
        mock_http_response.text = "Internal server error"
        
        http_error = httpx.HTTPStatusError(
            message="Server error",
            request=MagicMock(),
            response=mock_http_response,
        )
        mock_http_response.raise_for_status.side_effect = http_error

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(get=AsyncMock(return_value=mock_http_response)),
        ):
            result = await client.get_chapter_content(
                book_id="Architecture_Patterns_with_Python",
                chapter_number=5,
            )

        assert result is None  # Returns None on server error

    @pytest.mark.asyncio
    async def test_get_chapter_network_error(
        self,
        client: SemanticSearchClient,
    ) -> None:
        """Test handling of network errors."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("Connection refused")

        with patch.object(client, "_get_client", return_value=mock_client):
            result = await client.get_chapter_content(
                book_id="Architecture_Patterns_with_Python",
                chapter_number=5,
            )

        assert result is None  # Returns None on network error

    @pytest.mark.asyncio
    async def test_get_chapter_correct_url(
        self,
        client: SemanticSearchClient,
        mock_response_found: dict,
    ) -> None:
        """Test that correct URL path is called."""
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response_found
        mock_http_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_http_response)

        with patch.object(client, "_get_client", return_value=mock_http_client):
            await client.get_chapter_content(
                book_id="Test_Book",
                chapter_number=3,
            )

        # Verify the correct URL was called
        mock_http_client.get.assert_called_once_with("/v1/chapters/Test_Book/3")


class TestGetChapterContentIntegration:
    """Integration-style tests for content client behavior."""

    @pytest.mark.asyncio
    async def test_multiple_chapters_retrieval(
        self,
        client: SemanticSearchClient,
    ) -> None:
        """Test retrieving multiple chapters sequentially."""
        responses = [
            {
                "book_id": "Book_A",
                "chapter_number": 1,
                "title": "Chapter 1",
                "summary": "Summary 1",
                "keywords": ["a", "b"],
                "concepts": ["x"],
                "page_range": "1-20",
                "found": True,
            },
            {
                "book_id": "Book_A",
                "chapter_number": 2,
                "title": "Chapter 2",
                "summary": "Summary 2",
                "keywords": ["c", "d"],
                "concepts": ["y"],
                "page_range": "21-40",
                "found": True,
            },
        ]

        call_count = 0

        async def mock_get(url: str) -> MagicMock:
            nonlocal call_count
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = responses[call_count]
            response.raise_for_status = MagicMock()
            call_count += 1
            return response

        mock_http_client = AsyncMock()
        mock_http_client.get = mock_get

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result1 = await client.get_chapter_content("Book_A", 1)
            result2 = await client.get_chapter_content("Book_A", 2)

        assert result1 is not None
        assert result2 is not None
        assert result1["chapter_number"] == 1
        assert result2["chapter_number"] == 2
