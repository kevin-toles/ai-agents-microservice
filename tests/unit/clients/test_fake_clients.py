"""Unit tests for Fake Clients.

TDD Phase: RED - Tests written before implementation.

Acceptance Criteria Verified:
- AC-3.4.1: FakeCodeOrchestratorClient returns configured responses
- AC-3.4.2: FakeSemanticSearchClient returns configured responses
- AC-3.4.3: Fakes implement same protocols as real clients
- AC-3.4.4: Fakes support configurable error injection

Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-3.4
Pattern: FakeClient for testing (CODING_PATTERNS_ANALYSIS.md)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.msep.exceptions import ServiceUnavailableError
from src.clients.protocols import CodeOrchestratorProtocol, SemanticSearchProtocol
from tests.fakes.fake_clients import (
    FakeCodeOrchestratorClient,
    FakeSemanticSearchClient,
)


# ==============================================================================
# FakeCodeOrchestratorClient Tests
# ==============================================================================


class TestFakeCodeOrchestratorClient:
    """Tests for FakeCodeOrchestratorClient (AC-3.4.1, AC-3.4.3)."""

    def test_implements_protocol(self) -> None:
        """FakeCodeOrchestratorClient implements CodeOrchestratorProtocol (AC-3.4.3)."""
        client = FakeCodeOrchestratorClient()
        assert isinstance(client, CodeOrchestratorProtocol)

    @pytest.mark.asyncio
    async def test_get_embeddings_returns_configured_response(self) -> None:
        """get_embeddings returns configured embeddings (AC-3.4.1)."""
        configured_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        client = FakeCodeOrchestratorClient(
            embeddings_response=configured_embeddings
        )

        result = await client.get_embeddings(["text1", "text2"])

        assert np.array_equal(result, configured_embeddings)

    @pytest.mark.asyncio
    async def test_get_embeddings_default_response(self) -> None:
        """get_embeddings returns default empty array if not configured."""
        client = FakeCodeOrchestratorClient()

        result = await client.get_embeddings(["text1"])

        assert isinstance(result, np.ndarray)

    @pytest.mark.asyncio
    async def test_get_similarity_matrix_returns_configured_response(self) -> None:
        """get_similarity_matrix returns configured matrix (AC-3.4.1)."""
        configured_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        client = FakeCodeOrchestratorClient(
            similarity_response=configured_matrix
        )

        result = await client.get_similarity_matrix(["text1", "text2"])

        assert np.array_equal(result, configured_matrix)

    @pytest.mark.asyncio
    async def test_cluster_topics_returns_configured_response(self) -> None:
        """cluster_topics returns configured response (AC-3.4.1)."""
        configured_response = {
            "topic_assignments": [0, 0, 1],
            "topic_count": 2,
            "chapter_topic": 0,
        }
        client = FakeCodeOrchestratorClient(
            cluster_response=configured_response
        )

        result = await client.cluster_topics(["text1", "text2", "text3"], chapter_index=0)

        assert result == configured_response

    @pytest.mark.asyncio
    async def test_extract_keywords_returns_configured_response(self) -> None:
        """extract_keywords returns configured keywords (AC-3.4.1)."""
        configured_keywords = [["TDD", "testing"], ["DDD", "domain"]]
        client = FakeCodeOrchestratorClient(
            keywords_response=configured_keywords
        )

        result = await client.extract_keywords(["text1", "text2"], top_k=2)

        assert result == configured_keywords

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        """close() does nothing (no real resources to release)."""
        client = FakeCodeOrchestratorClient()
        await client.close()  # Should not raise

    def test_records_method_calls(self) -> None:
        """FakeClient records method calls for verification."""
        client = FakeCodeOrchestratorClient()

        assert hasattr(client, "call_history")
        assert isinstance(client.call_history, list)


class TestFakeCodeOrchestratorErrorInjection:
    """Tests for FakeCodeOrchestratorClient error injection (AC-3.4.4)."""

    @pytest.mark.asyncio
    async def test_inject_error_on_get_embeddings(self) -> None:
        """Can inject error on get_embeddings (AC-3.4.4)."""
        client = FakeCodeOrchestratorClient(
            error_on={"get_embeddings": ServiceUnavailableError("Simulated failure")}
        )

        with pytest.raises(ServiceUnavailableError):
            await client.get_embeddings(["text"])

    @pytest.mark.asyncio
    async def test_inject_error_on_get_similarity_matrix(self) -> None:
        """Can inject error on get_similarity_matrix (AC-3.4.4)."""
        client = FakeCodeOrchestratorClient(
            error_on={"get_similarity_matrix": ServiceUnavailableError("Simulated failure")}
        )

        with pytest.raises(ServiceUnavailableError):
            await client.get_similarity_matrix(["text"])

    @pytest.mark.asyncio
    async def test_inject_error_on_cluster_topics(self) -> None:
        """Can inject error on cluster_topics (AC-3.4.4)."""
        client = FakeCodeOrchestratorClient(
            error_on={"cluster_topics": ServiceUnavailableError("Simulated failure")}
        )

        with pytest.raises(ServiceUnavailableError):
            await client.cluster_topics(["text"], chapter_index=0)

    @pytest.mark.asyncio
    async def test_inject_error_on_extract_keywords(self) -> None:
        """Can inject error on extract_keywords (AC-3.4.4)."""
        client = FakeCodeOrchestratorClient(
            error_on={"extract_keywords": ServiceUnavailableError("Simulated failure")}
        )

        with pytest.raises(ServiceUnavailableError):
            await client.extract_keywords(["text"], top_k=5)

    @pytest.mark.asyncio
    async def test_error_only_affects_specified_method(self) -> None:
        """Error injection only affects specified method."""
        configured_embeddings = np.array([[0.1, 0.2]])
        client = FakeCodeOrchestratorClient(
            embeddings_response=configured_embeddings,
            error_on={"extract_keywords": ServiceUnavailableError("Only keywords fails")}
        )

        # get_embeddings should still work
        result = await client.get_embeddings(["text"])
        assert np.array_equal(result, configured_embeddings)

        # extract_keywords should fail
        with pytest.raises(ServiceUnavailableError):
            await client.extract_keywords(["text"], top_k=5)


# ==============================================================================
# FakeSemanticSearchClient Tests
# ==============================================================================


class TestFakeSemanticSearchClient:
    """Tests for FakeSemanticSearchClient (AC-3.4.2, AC-3.4.3)."""

    def test_implements_protocol(self) -> None:
        """FakeSemanticSearchClient implements SemanticSearchProtocol (AC-3.4.3)."""
        client = FakeSemanticSearchClient()
        assert isinstance(client, SemanticSearchProtocol)

    @pytest.mark.asyncio
    async def test_search_returns_configured_response(self) -> None:
        """search returns configured results (AC-3.4.2)."""
        configured_results = {
            "results": [{"chapter_id": "ch1", "score": 0.9}],
            "total": 1,
        }
        client = FakeSemanticSearchClient(
            search_response=configured_results
        )

        result = await client.search("test query")

        assert result == configured_results

    @pytest.mark.asyncio
    async def test_search_default_response(self) -> None:
        """search returns empty results if not configured."""
        client = FakeSemanticSearchClient()

        result = await client.search("test query")

        assert "results" in result
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_get_relationships_returns_configured_response(self) -> None:
        """get_relationships returns configured relationships (AC-3.4.2)."""
        configured_response = {
            "chapter_id": "ch1",
            "relationships": [
                {"target_chapter_id": "ch2", "weight": 0.8}
            ],
        }
        client = FakeSemanticSearchClient(
            relationships_response=configured_response
        )

        result = await client.get_relationships("ch1")

        assert result == configured_response

    @pytest.mark.asyncio
    async def test_get_relationships_batch_returns_configured_response(self) -> None:
        """get_relationships_batch returns configured batch results (AC-3.4.2)."""
        configured_response = {
            "results": {
                "ch1": [{"target_chapter_id": "ch2", "weight": 0.8}],
                "ch2": [{"target_chapter_id": "ch1", "weight": 0.8}],
            },
            "total_chapters": 2,
        }
        client = FakeSemanticSearchClient(
            batch_relationships_response=configured_response
        )

        result = await client.get_relationships_batch(["ch1", "ch2"])

        assert result == configured_response

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        """close() does nothing (no real resources to release)."""
        client = FakeSemanticSearchClient()
        await client.close()  # Should not raise

    def test_records_method_calls(self) -> None:
        """FakeClient records method calls for verification."""
        client = FakeSemanticSearchClient()

        assert hasattr(client, "call_history")
        assert isinstance(client.call_history, list)


class TestFakeSemanticSearchErrorInjection:
    """Tests for FakeSemanticSearchClient error injection (AC-3.4.4)."""

    @pytest.mark.asyncio
    async def test_inject_error_on_search(self) -> None:
        """Can inject error on search (AC-3.4.4)."""
        client = FakeSemanticSearchClient(
            error_on={"search": ServiceUnavailableError("Simulated failure")}
        )

        with pytest.raises(ServiceUnavailableError):
            await client.search("test query")

    @pytest.mark.asyncio
    async def test_inject_error_on_get_relationships(self) -> None:
        """Can inject error on get_relationships (AC-3.4.4)."""
        client = FakeSemanticSearchClient(
            error_on={"get_relationships": ServiceUnavailableError("Simulated failure")}
        )

        with pytest.raises(ServiceUnavailableError):
            await client.get_relationships("ch1")

    @pytest.mark.asyncio
    async def test_inject_error_on_get_relationships_batch(self) -> None:
        """Can inject error on get_relationships_batch (AC-3.4.4)."""
        client = FakeSemanticSearchClient(
            error_on={"get_relationships_batch": ServiceUnavailableError("Simulated failure")}
        )

        with pytest.raises(ServiceUnavailableError):
            await client.get_relationships_batch(["ch1"])

    @pytest.mark.asyncio
    async def test_error_only_affects_specified_method(self) -> None:
        """Error injection only affects specified method."""
        configured_search = {"results": [{"chapter_id": "ch1"}], "total": 1}
        client = FakeSemanticSearchClient(
            search_response=configured_search,
            error_on={"get_relationships": ServiceUnavailableError("Only relationships fails")}
        )

        # search should still work
        result = await client.search("test")
        assert result == configured_search

        # get_relationships should fail
        with pytest.raises(ServiceUnavailableError):
            await client.get_relationships("ch1")


# ==============================================================================
# Call History Tests
# ==============================================================================


class TestCallHistoryTracking:
    """Tests for call history tracking in fake clients."""

    @pytest.mark.asyncio
    async def test_code_orchestrator_tracks_get_embeddings_calls(self) -> None:
        """FakeCodeOrchestratorClient tracks get_embeddings calls."""
        client = FakeCodeOrchestratorClient()

        await client.get_embeddings(["text1", "text2"])

        assert len(client.call_history) == 1
        assert client.call_history[0]["method"] == "get_embeddings"
        assert client.call_history[0]["args"]["texts"] == ["text1", "text2"]

    @pytest.mark.asyncio
    async def test_semantic_search_tracks_search_calls(self) -> None:
        """FakeSemanticSearchClient tracks search calls."""
        client = FakeSemanticSearchClient()

        await client.search("test query", top_k=10)

        assert len(client.call_history) == 1
        assert client.call_history[0]["method"] == "search"
        assert client.call_history[0]["args"]["query"] == "test query"
        assert client.call_history[0]["args"]["top_k"] == 10

    def test_clear_call_history(self) -> None:
        """Can clear call history for test isolation."""
        client = FakeCodeOrchestratorClient()
        client.call_history.append({"method": "test"})

        client.clear_history()

        assert client.call_history == []
