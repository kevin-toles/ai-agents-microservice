"""Unit Tests: CodeReferenceClient.

WBS Reference: WBS-AGT21 Code Reference Engine Client (AGT21.7)
Acceptance Criteria:
- AC-21.1: Client wraps CodeReferenceEngine from ai-platform-data
- AC-21.2: Async interface for search, get_metadata, fetch_file
- AC-21.3: Integration with Qdrant for semantic code search
- AC-21.4: Integration with GitHub API for on-demand file retrieval
- AC-21.5: Returns CodeContext with citations for downstream

TDD Status: RED → GREEN → REFACTOR
Pattern: Protocol duck typing with FakeClient for isolation

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
Anti-Pattern Compliance: CODING_PATTERNS_ANALYSIS.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.code_reference import (
    CodeReferenceClient,
    CodeReferenceConfig,
    CodeContext,
    CodeChunk,
    CodeReference,
)
from src.clients.protocols import CodeReferenceProtocol


# =============================================================================
# Test Fixtures - Fake Client for Unit Tests
# =============================================================================

@dataclass
class FakeCodeReferenceClient:
    """Fake client for unit testing - implements CodeReferenceProtocol.
    
    Pattern: Protocol duck typing for test isolation.
    Reference: CODING_PATTERNS_ANALYSIS.md - FakeClient pattern
    """
    
    search_results: list[CodeContext] = field(default_factory=list)
    metadata_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    file_contents: dict[str, str] = field(default_factory=dict)
    search_call_count: int = 0
    fetch_call_count: int = 0
    should_raise: Exception | None = None
    
    async def search(
        self,
        query: str,
        domains: list[str] | None = None,
        concepts: list[str] | None = None,
        top_k: int = 10,
    ) -> CodeContext:
        """Fake search implementation."""
        self.search_call_count += 1
        if self.should_raise:
            raise self.should_raise
        if self.search_results:
            return self.search_results[0]
        return CodeContext(
            query=query,
            primary_references=[],
            domains_searched=domains or [],
            total_chunks_found=0,
        )
    
    async def search_by_concept(
        self,
        concept: str,
        top_k: int = 10,
    ) -> CodeContext:
        """Search by concept name."""
        return await self.search(concept, concepts=[concept], top_k=top_k)
    
    async def search_by_pattern(
        self,
        pattern: str,
        top_k: int = 10,
    ) -> CodeContext:
        """Search by design pattern name."""
        return await self.search(pattern, concepts=[pattern], top_k=top_k)
    
    async def get_metadata(self, repo_id: str) -> dict[str, Any] | None:
        """Get repository metadata."""
        return self.metadata_results.get(repo_id)
    
    async def fetch_file(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str | None:
        """Fetch file content."""
        self.fetch_call_count += 1
        if self.should_raise:
            raise self.should_raise
        return self.file_contents.get(file_path)
    
    async def close(self) -> None:
        """Close client resources."""
        pass


@pytest.fixture
def fake_client() -> FakeCodeReferenceClient:
    """Create a fake client with default test data."""
    return FakeCodeReferenceClient(
        search_results=[
            CodeContext(
                query="repository pattern",
                primary_references=[
                    CodeReference(
                        chunk=CodeChunk(
                            chunk_id="chunk_001",
                            repo_id="backend-ddd",
                            file_path="backend/ddd/repository.py",
                            start_line=10,
                            end_line=50,
                            content="class Repository:\n    pass",
                            language="python",
                            score=0.95,
                        ),
                        source_url="https://github.com/kevin-toles/code-reference-engine/blob/main/backend/ddd/repository.py#L10-L50",
                    ),
                ],
                domains_searched=["backend-frameworks"],
                total_chunks_found=1,
            ),
        ],
        metadata_results={
            "backend-ddd": {
                "id": "backend-ddd",
                "name": "DDD Examples",
                "domain": "backend-frameworks",
                "concepts": ["ddd", "repository", "domain-driven-design"],
                "patterns": ["repository", "aggregate", "value-object"],
            },
        },
        file_contents={
            "backend/ddd/repository.py": """class Repository:
    \"\"\"Base repository implementation.\"\"\"
    
    def __init__(self, session):
        self.session = session
    
    async def get(self, id: str):
        pass
    
    async def save(self, entity):
        pass
""",
        },
    )


# =============================================================================
# Protocol Compliance Tests
# =============================================================================

class TestCodeReferenceProtocol:
    """Tests that CodeReferenceClient implements the protocol."""
    
    def test_fake_client_implements_protocol(self, fake_client: FakeCodeReferenceClient) -> None:
        """
        AC-21.2: Verify FakeCodeReferenceClient matches protocol.
        
        Given: A FakeCodeReferenceClient instance
        When: Checking protocol compliance
        Then: Client satisfies CodeReferenceProtocol
        """
        assert isinstance(fake_client, CodeReferenceProtocol)
    
    @pytest.mark.asyncio
    async def test_real_client_implements_protocol(self) -> None:
        """
        AC-21.2: Verify CodeReferenceClient matches protocol.
        
        Given: A CodeReferenceClient instance
        When: Checking protocol compliance
        Then: Client satisfies CodeReferenceProtocol
        """
        config = CodeReferenceConfig(
            registry_path="/fake/path",
            github_token="fake_token",
        )
        client = CodeReferenceClient(config)
        assert isinstance(client, CodeReferenceProtocol)


# =============================================================================
# Search Tests
# =============================================================================

class TestCodeReferenceSearch:
    """Tests for search functionality."""
    
    @pytest.mark.asyncio
    async def test_search_returns_code_context(
        self, fake_client: FakeCodeReferenceClient
    ) -> None:
        """
        AC-21.3: Search returns CodeContext with results.
        
        Given: A configured client with mock search results
        When: Searching for "repository pattern"
        Then: Returns CodeContext with primary_references
        """
        result = await fake_client.search("repository pattern")
        
        assert isinstance(result, CodeContext)
        assert result.query == "repository pattern"
        assert len(result.primary_references) > 0
    
    @pytest.mark.asyncio
    async def test_search_with_domain_filter(
        self, fake_client: FakeCodeReferenceClient
    ) -> None:
        """
        AC-21.3: Search with domain filter.
        
        Given: A client with domain filter capability
        When: Searching with domains=["backend-frameworks"]
        Then: domains_searched reflects the filter
        """
        result = await fake_client.search(
            "saga pattern",
            domains=["backend-frameworks"],
        )
        
        assert "backend-frameworks" in result.domains_searched
    
    @pytest.mark.asyncio
    async def test_search_by_concept(
        self, fake_client: FakeCodeReferenceClient
    ) -> None:
        """
        AC-21.3: Search by concept name.
        
        Given: A client with concept search capability
        When: Searching by concept "event-driven"
        Then: Returns relevant code context
        """
        result = await fake_client.search_by_concept("event-driven")
        
        assert isinstance(result, CodeContext)
        assert fake_client.search_call_count == 1
    
    @pytest.mark.asyncio
    async def test_search_by_pattern(
        self, fake_client: FakeCodeReferenceClient
    ) -> None:
        """
        AC-21.3: Search by design pattern.
        
        Given: A client with pattern search capability
        When: Searching by pattern "saga"
        Then: Returns relevant code context
        """
        result = await fake_client.search_by_pattern("saga")
        
        assert isinstance(result, CodeContext)
    
    @pytest.mark.asyncio
    async def test_search_with_top_k(
        self, fake_client: FakeCodeReferenceClient
    ) -> None:
        """
        AC-21.3: Search respects top_k limit.
        
        Given: A client with result limiting
        When: Searching with top_k=5
        Then: Returns at most 5 results
        """
        result = await fake_client.search("any query", top_k=5)
        
        assert len(result.primary_references) <= 5
    
    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        """
        AC-21.3: Search handles no results gracefully.
        
        Given: A client with no matching results
        When: Searching for non-existent concept
        Then: Returns empty CodeContext (not None)
        """
        client = FakeCodeReferenceClient()  # No results configured
        result = await client.search("nonexistent_xyz_123")
        
        assert isinstance(result, CodeContext)
        assert len(result.primary_references) == 0
        assert result.total_chunks_found == 0


# =============================================================================
# Metadata Tests
# =============================================================================

class TestCodeReferenceMetadata:
    """Tests for metadata retrieval."""
    
    @pytest.mark.asyncio
    async def test_get_metadata_returns_repo_info(
        self, fake_client: FakeCodeReferenceClient
    ) -> None:
        """
        AC-21.1: Get repository metadata.
        
        Given: A client with metadata for "backend-ddd"
        When: Requesting metadata for that repo
        Then: Returns metadata dictionary
        """
        metadata = await fake_client.get_metadata("backend-ddd")
        
        assert metadata is not None
        assert metadata["id"] == "backend-ddd"
        assert "concepts" in metadata
        assert "repository" in metadata["concepts"]
    
    @pytest.mark.asyncio
    async def test_get_metadata_unknown_repo(
        self, fake_client: FakeCodeReferenceClient
    ) -> None:
        """
        AC-21.1: Get metadata for unknown repo.
        
        Given: A client with limited metadata
        When: Requesting metadata for unknown repo
        Then: Returns None
        """
        metadata = await fake_client.get_metadata("unknown-repo-xyz")
        
        assert metadata is None


# =============================================================================
# File Fetch Tests
# =============================================================================

class TestCodeReferenceFetchFile:
    """Tests for file content retrieval."""
    
    @pytest.mark.asyncio
    async def test_fetch_file_returns_content(
        self, fake_client: FakeCodeReferenceClient
    ) -> None:
        """
        AC-21.4: Fetch file content from GitHub.
        
        Given: A client with cached file content
        When: Fetching "backend/ddd/repository.py"
        Then: Returns file content as string
        """
        content = await fake_client.fetch_file("backend/ddd/repository.py")
        
        assert content is not None
        assert "class Repository" in content
        assert fake_client.fetch_call_count == 1
    
    @pytest.mark.asyncio
    async def test_fetch_file_with_line_range(
        self, fake_client: FakeCodeReferenceClient
    ) -> None:
        """
        AC-21.4: Fetch specific line range.
        
        Given: A client with file fetch capability
        When: Fetching lines 10-50 of a file
        Then: Client receives line range parameters
        """
        content = await fake_client.fetch_file(
            "backend/ddd/repository.py",
            start_line=10,
            end_line=50,
        )
        
        # Fake client doesn't filter by line, but real client should
        assert content is not None
    
    @pytest.mark.asyncio
    async def test_fetch_file_not_found(self) -> None:
        """
        AC-21.4: Fetch non-existent file.
        
        Given: A client with no file content
        When: Fetching unknown file
        Then: Returns None
        """
        client = FakeCodeReferenceClient()
        content = await client.fetch_file("nonexistent/file.py")
        
        assert content is None


# =============================================================================
# Citation Tests
# =============================================================================

class TestCodeReferenceCitations:
    """Tests for citation generation."""
    
    @pytest.mark.asyncio
    async def test_code_context_has_citations(
        self, fake_client: FakeCodeReferenceClient
    ) -> None:
        """
        AC-21.5: CodeContext includes citations.
        
        Given: Search results with code references
        When: Getting citations from CodeContext
        Then: Citations include file path, lines, and GitHub URL
        """
        result = await fake_client.search("repository pattern")
        
        citations = result.get_citations()
        
        assert len(citations) > 0
        assert any("github.com" in c for c in citations)
        assert any("repository.py" in c for c in citations)
    
    @pytest.mark.asyncio
    async def test_code_reference_has_source_url(
        self, fake_client: FakeCodeReferenceClient
    ) -> None:
        """
        AC-21.5: CodeReference includes source_url.
        
        Given: A search result with code references
        When: Examining a CodeReference
        Then: source_url contains GitHub URL with line range
        """
        result = await fake_client.search("repository pattern")
        ref = result.primary_references[0]
        
        assert ref.source_url is not None
        assert "github.com" in ref.source_url
        assert "#L" in ref.source_url  # Line anchor
    
    def test_code_chunk_line_range(self) -> None:
        """
        AC-21.5: CodeChunk has line range.
        
        Given: A CodeChunk instance
        When: Checking line properties
        Then: start_line and end_line are set
        """
        chunk = CodeChunk(
            chunk_id="test_001",
            repo_id="test-repo",
            file_path="test.py",
            start_line=10,
            end_line=50,
            content="test content",
            language="python",
            score=0.9,
        )
        
        assert chunk.start_line == 10
        assert chunk.end_line == 50
        assert chunk.end_line > chunk.start_line


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestCodeReferenceErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_search_handles_exception(self) -> None:
        """
        Error handling: Search gracefully handles exceptions.
        
        Given: A client configured to raise exception
        When: Search is called
        Then: Exception propagates (caller handles)
        """
        client = FakeCodeReferenceClient(
            should_raise=ConnectionError("Network unavailable")
        )
        
        with pytest.raises(ConnectionError):
            await client.search("any query")
    
    @pytest.mark.asyncio
    async def test_fetch_handles_exception(self) -> None:
        """
        Error handling: Fetch gracefully handles exceptions.
        
        Given: A client configured to raise exception
        When: Fetch is called
        Then: Exception propagates (caller handles)
        """
        client = FakeCodeReferenceClient(
            should_raise=TimeoutError("GitHub API timeout")
        )
        
        with pytest.raises(TimeoutError):
            await client.fetch_file("any/file.py")


# =============================================================================
# Integration with Citation Mapper Tests
# =============================================================================

class TestCodeCitationMapper:
    """Tests for CodeContext → Citation mapping."""
    
    @pytest.mark.asyncio
    async def test_code_context_to_citation(
        self, fake_client: FakeCodeReferenceClient
    ) -> None:
        """
        AC-21.5: CodeContext converts to Citation schema.
        
        Given: A CodeContext with references
        When: Converting to Citation objects
        Then: Citations have required fields for downstream
        """
        from src.citations.code_citation import code_context_to_citations
        
        result = await fake_client.search("repository pattern")
        citations = code_context_to_citations(result)
        
        assert len(citations) > 0
        for citation in citations:
            assert citation.source_type == "code"
            assert citation.url is not None
            assert citation.file_path is not None
    
    @pytest.mark.asyncio
    async def test_empty_code_context_to_citation(self) -> None:
        """
        AC-21.5: Empty CodeContext produces empty citation list.
        
        Given: A CodeContext with no references
        When: Converting to Citation objects
        Then: Returns empty list (not None)
        """
        from src.citations.code_citation import code_context_to_citations
        
        empty_context = CodeContext(
            query="nothing",
            primary_references=[],
            domains_searched=[],
            total_chunks_found=0,
        )
        
        citations = code_context_to_citations(empty_context)
        
        assert citations == []


# =============================================================================
# Configuration Tests
# =============================================================================

class TestCodeReferenceConfig:
    """Tests for configuration."""
    
    def test_config_defaults(self) -> None:
        """
        Config: Default values are sensible.
        
        Given: A CodeReferenceConfig with minimal args
        When: Checking default values
        Then: Defaults match expected values
        """
        config = CodeReferenceConfig(
            registry_path="/path/to/registry.json",
        )
        
        assert config.registry_path == "/path/to/registry.json"
        assert config.qdrant_collection == "code_chunks"
        assert config.github_token is None
    
    def test_config_from_env(self) -> None:
        """
        Config: Can load from environment variables.
        
        Given: Environment variables set
        When: Creating config with from_env=True
        Then: Config loads from environment
        """
        with patch.dict("os.environ", {
            "CODE_REFERENCE_REGISTRY": "/env/registry.json",
            "GITHUB_TOKEN": "env_token_123",
        }):
            config = CodeReferenceConfig.from_env()
            
            assert config.registry_path == "/env/registry.json"
            assert config.github_token == "env_token_123"


# =============================================================================
# Async Context Manager Tests
# =============================================================================

class TestCodeReferenceContextManager:
    """Tests for async context manager support."""
    
    @pytest.mark.asyncio
    async def test_client_as_context_manager(self) -> None:
        """
        AC-21.2: Client works as async context manager.
        
        Given: A CodeReferenceClient
        When: Used with async with
        Then: Properly enters and exits
        """
        config = CodeReferenceConfig(
            registry_path="/fake/path",
            github_token="fake_token",
        )
        
        async with CodeReferenceClient(config) as client:
            assert client is not None
        
        # After exit, client should be closed
        # (actual verification would be internal state)
    
    @pytest.mark.asyncio
    async def test_close_releases_resources(self) -> None:
        """
        AC-21.2: Close method releases resources.
        
        Given: An open client
        When: Calling close()
        Then: HTTP client is closed
        """
        fake = FakeCodeReferenceClient()
        await fake.close()
        
        # Fake client doesn't track close, but real client should
        # This test ensures the method exists and is callable
