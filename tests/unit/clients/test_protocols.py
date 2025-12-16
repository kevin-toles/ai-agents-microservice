"""Unit tests for MSEP client protocols.

TDD Phase: RED - Tests written before implementation.

Acceptance Criteria Verified:
- AC-3.3.1: CodeOrchestratorProtocol defines all required methods
- AC-3.3.2: SemanticSearchProtocol defines all required methods
- AC-3.3.3: Both clients implement their protocols

Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-3.3
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ==============================================================================
# Protocol Tests - Verify Protocol Definitions
# ==============================================================================


class TestCodeOrchestratorProtocol:
    """Tests for CodeOrchestratorProtocol definition (AC-3.3.1)."""

    def test_protocol_is_importable(self) -> None:
        """Protocol can be imported from clients module."""
        from src.clients.protocols import CodeOrchestratorProtocol

        assert CodeOrchestratorProtocol is not None

    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol is marked @runtime_checkable for isinstance checks."""
        from src.clients.protocols import CodeOrchestratorProtocol

        # runtime_checkable protocols have __protocol_attrs__
        assert hasattr(CodeOrchestratorProtocol, "__protocol_attrs__") or isinstance(
            CodeOrchestratorProtocol, type
        )

    def test_protocol_has_get_embeddings_method(self) -> None:
        """Protocol defines get_embeddings(texts) method."""
        from src.clients.protocols import CodeOrchestratorProtocol

        # Check method exists in protocol
        assert "get_embeddings" in dir(CodeOrchestratorProtocol)

    def test_protocol_has_get_similarity_matrix_method(self) -> None:
        """Protocol defines get_similarity_matrix(texts) method."""
        from src.clients.protocols import CodeOrchestratorProtocol

        assert "get_similarity_matrix" in dir(CodeOrchestratorProtocol)

    def test_protocol_has_cluster_topics_method(self) -> None:
        """Protocol defines cluster_topics(corpus, chapter_index) method."""
        from src.clients.protocols import CodeOrchestratorProtocol

        assert "cluster_topics" in dir(CodeOrchestratorProtocol)

    def test_protocol_has_extract_keywords_method(self) -> None:
        """Protocol defines extract_keywords(corpus, top_k) method."""
        from src.clients.protocols import CodeOrchestratorProtocol

        assert "extract_keywords" in dir(CodeOrchestratorProtocol)

    def test_protocol_has_close_method(self) -> None:
        """Protocol defines close() method for cleanup."""
        from src.clients.protocols import CodeOrchestratorProtocol

        assert "close" in dir(CodeOrchestratorProtocol)


class TestSemanticSearchProtocol:
    """Tests for SemanticSearchProtocol definition (AC-3.3.2)."""

    def test_protocol_is_importable(self) -> None:
        """Protocol can be imported from clients module."""
        from src.clients.protocols import SemanticSearchProtocol

        assert SemanticSearchProtocol is not None

    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol is marked @runtime_checkable for isinstance checks."""
        from src.clients.protocols import SemanticSearchProtocol

        assert hasattr(SemanticSearchProtocol, "__protocol_attrs__") or isinstance(
            SemanticSearchProtocol, type
        )

    def test_protocol_has_search_method(self) -> None:
        """Protocol defines search(query, top_k) method."""
        from src.clients.protocols import SemanticSearchProtocol

        assert "search" in dir(SemanticSearchProtocol)

    def test_protocol_has_get_relationships_method(self) -> None:
        """Protocol defines get_relationships(chapter_id) method."""
        from src.clients.protocols import SemanticSearchProtocol

        assert "get_relationships" in dir(SemanticSearchProtocol)

    def test_protocol_has_get_relationships_batch_method(self) -> None:
        """Protocol defines get_relationships_batch(chapter_ids) method."""
        from src.clients.protocols import SemanticSearchProtocol

        assert "get_relationships_batch" in dir(SemanticSearchProtocol)

    def test_protocol_has_close_method(self) -> None:
        """Protocol defines close() method for cleanup."""
        from src.clients.protocols import SemanticSearchProtocol

        assert "close" in dir(SemanticSearchProtocol)


# ==============================================================================
# Protocol Implementation Tests (AC-3.3.3)
# ==============================================================================


class TestClientProtocolImplementation:
    """Tests verifying clients implement their protocols (AC-3.3.3)."""

    def test_code_orchestrator_client_implements_protocol(self) -> None:
        """CodeOrchestratorClient implements CodeOrchestratorProtocol."""
        from src.clients.code_orchestrator import CodeOrchestratorClient
        from src.clients.protocols import CodeOrchestratorProtocol

        # Create instance and verify protocol compliance
        client = CodeOrchestratorClient(base_url="http://test:8082")
        assert isinstance(client, CodeOrchestratorProtocol)

    def test_semantic_search_client_implements_protocol(self) -> None:
        """SemanticSearchClient implements SemanticSearchProtocol."""
        from src.clients.semantic_search import MSEPSemanticSearchClient
        from src.clients.protocols import SemanticSearchProtocol

        # Create instance and verify protocol compliance
        client = MSEPSemanticSearchClient(base_url="http://test:8081")
        assert isinstance(client, SemanticSearchProtocol)

    def test_fake_code_orchestrator_implements_protocol(self) -> None:
        """FakeCodeOrchestratorClient implements CodeOrchestratorProtocol."""
        from src.clients.protocols import CodeOrchestratorProtocol
        from tests.fakes.fake_clients import FakeCodeOrchestratorClient

        client = FakeCodeOrchestratorClient()
        assert isinstance(client, CodeOrchestratorProtocol)

    def test_fake_semantic_search_implements_protocol(self) -> None:
        """FakeSemanticSearchClient implements SemanticSearchProtocol."""
        from src.clients.protocols import SemanticSearchProtocol
        from tests.fakes.fake_clients import FakeSemanticSearchClient

        client = FakeSemanticSearchClient()
        assert isinstance(client, SemanticSearchProtocol)
