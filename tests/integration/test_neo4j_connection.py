"""Integration Tests: Neo4j Connectivity.

PCON-8.1: ai-agents → Neo4j connectivity test
Acceptance Criteria: AC-8.1 - ai-agents → Neo4j connectivity test passes

Tests the direct connection from ai-agents to the canonical Neo4j database
at ai-platform-neo4j:7687.
"""

from __future__ import annotations

import os

import pytest

# Mark all tests as integration tests
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


# =============================================================================
# Configuration
# =============================================================================

NEO4J_URI = os.environ.get("AI_AGENTS_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("AI_AGENTS_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("AI_AGENTS_NEO4J_PASSWORD", "devpassword")


# =============================================================================
# Test Classes
# =============================================================================


class TestNeo4jConnectivity:
    """PCON-8.1: Neo4j connectivity tests.
    
    AC-8.1: ai-agents → Neo4j connectivity test passes
    
    Verifies:
    - Connection to ai-platform-neo4j:7687
    - Basic query execution
    - Schema verification (Books, Chapters, Concepts exist)
    """

    @pytest.mark.asyncio
    async def test_neo4j_connection_via_driver(self) -> None:
        """Test direct Neo4j connection using neo4j driver.
        
        Given: Neo4j running at ai-platform-neo4j:7687
        When: Connect and run basic query
        Then: Returns valid result
        """
        from neo4j import AsyncGraphDatabase
        
        driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        
        try:
            async with driver.session() as session:
                result = await session.run("RETURN 1 as n")
                record = await result.single()
                assert record is not None
                assert record["n"] == 1
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_neo4j_schema_has_books(self) -> None:
        """Test that Neo4j has Book nodes from seeding.
        
        Given: Neo4j seeded with PCON-3
        When: Query for Book count
        Then: Returns > 0 books
        """
        from neo4j import AsyncGraphDatabase
        
        driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        
        try:
            async with driver.session() as session:
                result = await session.run("MATCH (b:Book) RETURN count(b) as count")
                record = await result.single()
                assert record is not None
                count = record["count"]
                assert count > 0, f"Expected > 0 Books, got {count}"
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_neo4j_schema_has_chapters(self) -> None:
        """Test that Neo4j has Chapter nodes from seeding.
        
        Given: Neo4j seeded with PCON-3
        When: Query for Chapter count
        Then: Returns > 0 chapters
        """
        from neo4j import AsyncGraphDatabase
        
        driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        
        try:
            async with driver.session() as session:
                result = await session.run("MATCH (c:Chapter) RETURN count(c) as count")
                record = await result.single()
                assert record is not None
                count = record["count"]
                assert count > 0, f"Expected > 0 Chapters, got {count}"
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_neo4j_via_client(self) -> None:
        """Test Neo4j connection via ai-agents Neo4jClient.
        
        Given: Neo4j running and Neo4jClient configured
        When: Use client to query
        Then: Returns valid results
        """
        from src.clients.neo4j_client import Neo4jClient, Neo4jClientConfig
        
        config = Neo4jClientConfig(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
        )
        
        client = Neo4jClient(config)
        
        try:
            # Connect first
            await client.connect()
            
            # Verify connection via health check
            is_healthy = await client.health_check()
            assert is_healthy, "Neo4jClient health check failed"
            
            # Try a basic query
            chapters = await client.search_chapters(
                concepts=["software", "design"],
                limit=5,
            )
            # May return empty if no matching data, but shouldn't error
            assert isinstance(chapters, list)
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_neo4j_book_chapter_relationships(self) -> None:
        """Test that Book-Chapter relationships exist.
        
        Given: Neo4j seeded with books and chapters
        When: Query HAS_CHAPTER relationships
        Then: Returns > 0 relationships
        """
        from neo4j import AsyncGraphDatabase
        
        driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        
        try:
            async with driver.session() as session:
                result = await session.run(
                    "MATCH (:Book)-[r:HAS_CHAPTER]->(:Chapter) RETURN count(r) as count"
                )
                record = await result.single()
                assert record is not None
                count = record["count"]
                assert count > 0, f"Expected > 0 HAS_CHAPTER relationships, got {count}"
        finally:
            await driver.close()
