"""Unit Tests: Neo4jClient.

WBS Reference: WBS-AGT22 Neo4j Graph Integration (AGT22.8)
Acceptance Criteria:
- AC-22.1: Client connects to Neo4j for graph traversal
- AC-22.2: Query book → chapter → concept relationships
- AC-22.3: Query concept → code-reference-engine file mappings
- AC-22.4: Query cross-repo pattern relationships
- AC-22.5: Results include metadata for citation generation

TDD Status: RED → GREEN → REFACTOR
Pattern: Protocol duck typing with FakeNeo4jClient for isolation

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
Anti-Pattern Compliance: CODING_PATTERNS_ANALYSIS.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.clients.neo4j_client import (
    Neo4jClient,
    Neo4jClientConfig,
)
from src.clients.protocols import Neo4jClientProtocol
from src.schemas.graph_models import (
    Concept,
    CodeFileReference,
    PatternRelationship,
    GraphReference,
)


# =============================================================================
# Test Fixtures - FakeNeo4jClient for Unit Tests
# =============================================================================

@dataclass
class FakeNeo4jClient:
    """Fake client for unit testing - implements Neo4jClientProtocol.
    
    Pattern: Protocol duck typing for test isolation.
    Reference: CODING_PATTERNS_ANALYSIS.md - FakeClient pattern
    """
    
    # Preconfigured query results
    concepts_for_chapter: dict[str, list[Concept]] = field(default_factory=dict)
    code_for_concept: dict[str, list[CodeFileReference]] = field(default_factory=dict)
    related_patterns: dict[str, list[PatternRelationship]] = field(default_factory=dict)
    chapters_for_concept: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    
    # Call tracking
    query_count: int = 0
    last_cypher: str = ""
    should_raise: Exception | None = None
    _connected: bool = False
    
    async def connect(self) -> None:
        """Fake connect."""
        if self.should_raise:
            raise self.should_raise
        self._connected = True
    
    async def close(self) -> None:
        """Fake close."""
        self._connected = False
    
    async def __aenter__(self) -> "FakeNeo4jClient":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def health_check(self) -> bool:
        """Fake health check."""
        return self._connected
    
    async def get_concepts_for_chapter(
        self,
        chapter_id: str,
    ) -> list[Concept]:
        """Get concepts linked to a chapter.
        
        AC-22.2: Query book → chapter → concept relationships
        """
        self.query_count += 1
        if self.should_raise:
            raise self.should_raise
        return self.concepts_for_chapter.get(chapter_id, [])
    
    async def get_code_for_concept(
        self,
        concept: str,
    ) -> list[CodeFileReference]:
        """Get code file references for a concept.
        
        AC-22.3: Query concept → code-reference-engine file mappings
        """
        self.query_count += 1
        if self.should_raise:
            raise self.should_raise
        return self.code_for_concept.get(concept, [])
    
    async def get_related_patterns(
        self,
        pattern: str,
    ) -> list[PatternRelationship]:
        """Get cross-repo pattern relationships.
        
        AC-22.4: Query cross-repo pattern relationships
        """
        self.query_count += 1
        if self.should_raise:
            raise self.should_raise
        return self.related_patterns.get(pattern, [])
    
    async def get_chapters_for_concept(
        self,
        concept: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get chapters linked to a concept."""
        self.query_count += 1
        results = self.chapters_for_concept.get(concept, [])
        return results[:limit]


@pytest.fixture
def fake_client() -> FakeNeo4jClient:
    """Create a fake client with default test data."""
    return FakeNeo4jClient(
        concepts_for_chapter={
            "ch_001": [
                Concept(
                    concept_id="ddd",
                    name="Domain-Driven Design",
                    tier=1,
                    keywords=["bounded-context", "aggregate", "entity"],
                ),
                Concept(
                    concept_id="repository-pattern",
                    name="Repository Pattern",
                    tier=2,
                    keywords=["data-access", "persistence", "abstraction"],
                ),
            ],
        },
        code_for_concept={
            "repository-pattern": [
                CodeFileReference(
                    file_path="backend/ddd/repository.py",
                    repo_id="code-reference-engine",
                    start_line=10,
                    end_line=50,
                    language="python",
                    github_url="https://github.com/org/code-reference-engine/blob/main/backend/ddd/repository.py#L10-L50",
                ),
            ],
        },
        related_patterns={
            "saga": [
                PatternRelationship(
                    source_pattern="saga",
                    related_pattern="event-sourcing",
                    relationship_type="PARALLEL",
                    repos=["backend-patterns", "distributed-systems"],
                    similarity_score=0.85,
                ),
                PatternRelationship(
                    source_pattern="saga",
                    related_pattern="choreography",
                    relationship_type="PERPENDICULAR",
                    repos=["microservices-examples"],
                    similarity_score=0.78,
                ),
            ],
        },
        chapters_for_concept={
            "ddd": [
                {
                    "chapter_id": "ch_001",
                    "title": "Introduction to DDD",
                    "book_id": "ddd-distilled",
                    "tier": 1,
                },
            ],
        },
    )


# =============================================================================
# Protocol Compliance Tests
# =============================================================================

class TestNeo4jClientProtocol:
    """Tests that Neo4jClient implements the protocol."""
    
    def test_fake_client_implements_protocol(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.1: Verify FakeNeo4jClient matches protocol.
        
        Given: A FakeNeo4jClient instance
        When: Checking protocol compliance
        Then: Client satisfies Neo4jClientProtocol
        """
        assert isinstance(fake_client, Neo4jClientProtocol)
    
    @pytest.mark.asyncio
    async def test_real_client_implements_protocol(self) -> None:
        """
        AC-22.1: Verify Neo4jClient matches protocol.
        
        Given: A Neo4jClient instance
        When: Checking protocol compliance
        Then: Client satisfies Neo4jClientProtocol
        """
        config = Neo4jClientConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
        )
        client = Neo4jClient(config)
        assert isinstance(client, Neo4jClientProtocol)


# =============================================================================
# Connection Tests
# =============================================================================

class TestNeo4jClientConnection:
    """Tests for connection management."""
    
    @pytest.mark.asyncio
    async def test_connect_success(self, fake_client: FakeNeo4jClient) -> None:
        """
        AC-22.1: Client connects to Neo4j successfully.
        
        Given: A configured client
        When: Connecting
        Then: Connection succeeds
        """
        await fake_client.connect()
        assert fake_client._connected
    
    @pytest.mark.asyncio
    async def test_context_manager(self, fake_client: FakeNeo4jClient) -> None:
        """
        AC-22.1: Client works as async context manager.
        
        Given: A configured client
        When: Used with async with
        Then: Properly enters and exits
        """
        async with fake_client as client:
            assert client._connected
        assert not fake_client._connected
    
    @pytest.mark.asyncio
    async def test_health_check(self, fake_client: FakeNeo4jClient) -> None:
        """
        AC-22.1: Health check returns connection status.
        
        Given: A connected client
        When: Calling health_check
        Then: Returns True
        """
        async with fake_client:
            result = await fake_client.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self) -> None:
        """
        AC-22.1: Connection errors are handled gracefully.
        
        Given: A client configured to fail connection
        When: Attempting to connect
        Then: Raises appropriate exception
        """
        client = FakeNeo4jClient(
            should_raise=ConnectionError("Neo4j unavailable")
        )
        
        with pytest.raises(ConnectionError):
            await client.connect()


# =============================================================================
# Get Concepts for Chapter Tests
# =============================================================================

class TestGetConceptsForChapter:
    """Tests for AC-22.2: book → chapter → concept relationships."""
    
    @pytest.mark.asyncio
    async def test_get_concepts_returns_list(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.2: Returns list of concepts for chapter.
        
        Given: A chapter with linked concepts
        When: Querying concepts for chapter
        Then: Returns list of Concept objects
        """
        concepts = await fake_client.get_concepts_for_chapter("ch_001")
        
        assert isinstance(concepts, list)
        assert len(concepts) == 2
        assert all(isinstance(c, Concept) for c in concepts)
    
    @pytest.mark.asyncio
    async def test_concept_has_required_fields(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.2: Each Concept has required fields.
        
        Given: A query result
        When: Examining a Concept
        Then: Has concept_id, name, tier, keywords
        """
        concepts = await fake_client.get_concepts_for_chapter("ch_001")
        concept = concepts[0]
        
        assert concept.concept_id == "ddd"
        assert concept.name == "Domain-Driven Design"
        assert concept.tier == 1
        assert "bounded-context" in concept.keywords
    
    @pytest.mark.asyncio
    async def test_unknown_chapter_returns_empty(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.2: Unknown chapter returns empty list.
        
        Given: A non-existent chapter ID
        When: Querying concepts
        Then: Returns empty list (not None)
        """
        concepts = await fake_client.get_concepts_for_chapter("unknown_ch")
        
        assert concepts == []


# =============================================================================
# Get Code for Concept Tests
# =============================================================================

class TestGetCodeForConcept:
    """Tests for AC-22.3: concept → code file mappings."""
    
    @pytest.mark.asyncio
    async def test_get_code_returns_list(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.3: Returns list of code file references.
        
        Given: A concept with linked code files
        When: Querying code for concept
        Then: Returns list of CodeFileReference objects
        """
        refs = await fake_client.get_code_for_concept("repository-pattern")
        
        assert isinstance(refs, list)
        assert len(refs) == 1
        assert all(isinstance(r, CodeFileReference) for r in refs)
    
    @pytest.mark.asyncio
    async def test_code_reference_has_required_fields(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.3: Each CodeFileReference has required fields.
        
        Given: A query result
        When: Examining a CodeFileReference
        Then: Has file_path, repo_id, line range, github_url
        """
        refs = await fake_client.get_code_for_concept("repository-pattern")
        ref = refs[0]
        
        assert ref.file_path == "backend/ddd/repository.py"
        assert ref.repo_id == "code-reference-engine"
        assert ref.start_line == 10
        assert ref.end_line == 50
        assert "github.com" in ref.github_url
    
    @pytest.mark.asyncio
    async def test_unknown_concept_returns_empty(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.3: Unknown concept returns empty list.
        
        Given: A non-existent concept
        When: Querying code
        Then: Returns empty list
        """
        refs = await fake_client.get_code_for_concept("unknown-concept")
        
        assert refs == []


# =============================================================================
# Get Related Patterns Tests
# =============================================================================

class TestGetRelatedPatterns:
    """Tests for AC-22.4: cross-repo pattern relationships."""
    
    @pytest.mark.asyncio
    async def test_get_related_patterns_returns_list(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.4: Returns list of pattern relationships.
        
        Given: A pattern with cross-repo relationships
        When: Querying related patterns
        Then: Returns list of PatternRelationship objects
        """
        patterns = await fake_client.get_related_patterns("saga")
        
        assert isinstance(patterns, list)
        assert len(patterns) == 2
        assert all(isinstance(p, PatternRelationship) for p in patterns)
    
    @pytest.mark.asyncio
    async def test_pattern_relationship_has_required_fields(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.4: Each PatternRelationship has required fields.
        
        Given: A query result
        When: Examining a PatternRelationship
        Then: Has source, related, type, repos, similarity
        """
        patterns = await fake_client.get_related_patterns("saga")
        pattern = patterns[0]
        
        assert pattern.source_pattern == "saga"
        assert pattern.related_pattern == "event-sourcing"
        assert pattern.relationship_type in ["PARALLEL", "PERPENDICULAR", "SKIP_TIER"]
        assert "backend-patterns" in pattern.repos
        assert 0.0 <= pattern.similarity_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_cross_repo_results(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.4: Results span multiple repositories.
        
        Given: A pattern with cross-repo relationships
        When: Examining results
        Then: Multiple repos are represented
        """
        patterns = await fake_client.get_related_patterns("saga")
        
        all_repos = set()
        for p in patterns:
            all_repos.update(p.repos)
        
        assert len(all_repos) >= 2  # Cross-repo means multiple repos


# =============================================================================
# Citation Metadata Tests
# =============================================================================

class TestCitationMetadata:
    """Tests for AC-22.5: results include citation metadata."""
    
    @pytest.mark.asyncio
    async def test_concept_has_citation_metadata(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.5: Concept includes metadata for citation.
        
        Given: A concept query result
        When: Creating citation
        Then: Has enough data for citation generation
        """
        concepts = await fake_client.get_concepts_for_chapter("ch_001")
        concept = concepts[0]
        
        # Required for citation
        assert concept.concept_id is not None
        assert concept.name is not None
    
    @pytest.mark.asyncio
    async def test_code_reference_has_citation_metadata(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.5: CodeFileReference includes citation metadata.
        
        Given: A code reference query result
        When: Creating citation
        Then: Has file path, line range, GitHub URL
        """
        refs = await fake_client.get_code_for_concept("repository-pattern")
        ref = refs[0]
        
        # Required for citation: file path with line anchors
        assert ref.github_url is not None
        assert "#L" in ref.github_url  # Line anchor
        assert ref.file_path is not None
    
    @pytest.mark.asyncio
    async def test_graph_reference_to_citation(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.5: GraphReference converts to citation.
        
        Given: A GraphReference object
        When: Converting to citation
        Then: Produces valid citation string
        """
        concepts = await fake_client.get_concepts_for_chapter("ch_001")
        
        # Create GraphReference from concept
        ref = GraphReference(
            node_type="Concept",
            node_id=concepts[0].concept_id,
            name=concepts[0].name,
            source_query="get_concepts_for_chapter",
            metadata={"tier": concepts[0].tier},
        )
        
        citation = ref.to_citation()
        
        assert "Concept" in citation
        assert concepts[0].name in citation


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_query_error_propagates(self) -> None:
        """
        Error handling: Query errors propagate appropriately.
        
        Given: A client configured to raise on query
        When: Executing query
        Then: Exception propagates
        """
        client = FakeNeo4jClient(
            should_raise=RuntimeError("Query failed")
        )
        
        with pytest.raises(RuntimeError):
            await client.get_concepts_for_chapter("ch_001")


# =============================================================================
# Configuration Tests
# =============================================================================

class TestNeo4jClientConfig:
    """Tests for configuration."""
    
    def test_config_defaults(self) -> None:
        """
        Config: Default values are sensible.
        
        Given: A Neo4jClientConfig with minimal args
        When: Checking default values
        Then: Defaults match expected values
        """
        config = Neo4jClientConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
        )
        
        assert config.uri == "bolt://localhost:7687"
        assert config.database == "neo4j"  # Default database
        assert config.max_connection_pool_size == 50
    
    def test_config_from_env(self) -> None:
        """
        Config: Can load from environment variables.
        
        Given: Environment variables set
        When: Creating config with from_env
        Then: Config loads from environment
        """
        from unittest.mock import patch
        
        with patch.dict("os.environ", {
            "NEO4J_URI": "bolt://test:7687",
            "NEO4J_USER": "test_user",
            "NEO4J_PASSWORD": "test_pass",
        }):
            config = Neo4jClientConfig.from_env()
            
            assert config.uri == "bolt://test:7687"
            assert config.user == "test_user"


# =============================================================================
# Graph Citation Mapper Tests
# =============================================================================

class TestGraphCitationMapper:
    """Tests for GraphReference → Citation mapping."""
    
    @pytest.mark.asyncio
    async def test_concept_to_citation(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.5: Concept converts to GraphCitation.
        
        Given: A Concept from Neo4j query
        When: Converting to GraphCitation
        Then: Citation has required fields
        """
        from src.citations.graph_citation import concept_to_citation
        
        concepts = await fake_client.get_concepts_for_chapter("ch_001")
        citation = concept_to_citation(concepts[0])
        
        assert citation.source_type == "graph"
        assert citation.node_type == "Concept"
        assert citation.node_id == "ddd"
    
    @pytest.mark.asyncio
    async def test_code_file_to_citation(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.5: CodeFileReference converts to GraphCitation.
        
        Given: A CodeFileReference from Neo4j query
        When: Converting to GraphCitation
        Then: Citation includes code-specific fields
        """
        from src.citations.graph_citation import code_file_to_citation
        
        refs = await fake_client.get_code_for_concept("repository-pattern")
        citation = code_file_to_citation(refs[0])
        
        assert citation.source_type == "graph"
        assert citation.node_type == "CodeFile"
        assert citation.file_path == "backend/ddd/repository.py"
        assert citation.github_url is not None
    
    @pytest.mark.asyncio
    async def test_pattern_to_citation(
        self, fake_client: FakeNeo4jClient
    ) -> None:
        """
        AC-22.5: PatternRelationship converts to GraphCitation.
        
        Given: A PatternRelationship from Neo4j query
        When: Converting to GraphCitation
        Then: Citation includes relationship info
        """
        from src.citations.graph_citation import pattern_to_citation
        
        patterns = await fake_client.get_related_patterns("saga")
        citation = pattern_to_citation(patterns[0])
        
        assert citation.source_type == "graph"
        assert citation.node_type == "Pattern"
        assert "saga" in citation.description.lower()
