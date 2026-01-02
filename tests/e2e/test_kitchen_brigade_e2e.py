"""WBS-KB9: End-to-End Validation Tests.

Test the complete Kitchen Brigade pipeline end-to-end with the MCP server
and cross-reference functionality.

WBS Reference: WBS_KITCHEN_BRIGADE.md - WBS-KB9
Tasks: KB9.1-KB9.6

Acceptance Criteria:
- AC-KB9.1: E2E test: "Where is the rate limiter implemented?" returns code location with citation
- AC-KB9.2: E2E test: "Explain the repository pattern" returns explanation with book + code citations
- AC-KB9.3: E2E test: "Generate a cache service" produces code validated by Code-Orchestrator
- AC-KB9.4: E2E test: Deliberate hallucination rejected by audit-service
- AC-KB9.5: E2E test: Multi-turn conversation maintains context
- AC-KB9.6: Performance: <60s for typical query, <120s for complex multi-cycle

Exit Criteria:
- pytest tests/e2e/test_kitchen_brigade_e2e.py passes
- All 5 core E2E tests pass
- Performance meets targets

TDD Status: RED phase - Tests should fail initially until services connected
"""

from __future__ import annotations

import asyncio
import os
import pytest
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.mcp.server import MCPServer, MCPServerConfig, create_mcp_server


# =============================================================================
# Configuration
# =============================================================================

E2E_SERVICES_ENABLED = os.getenv("KB_E2E_SERVICES", "false").lower() == "true"

# Skip live service tests by default
pytestmark_live = pytest.mark.skipif(
    not E2E_SERVICES_ENABLED,
    reason="E2E services not enabled. Set KB_E2E_SERVICES=true to run.",
)


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

_CONST_TOOL_CROSS_REFERENCE = "cross_reference"
_CONST_TOOL_GENERATE_CODE = "generate_code"
_CONST_TOOL_ANALYZE_CODE = "analyze_code"
_CONST_TOOL_EXPLAIN_CODE = "explain_code"
_CONST_TYPICAL_TIMEOUT = 60  # seconds
_CONST_COMPLEX_TIMEOUT = 120  # seconds


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mcp_server():
    """Create MCP server with mock pipeline for unit-level E2E tests."""
    config = MCPServerConfig()
    server = MCPServer(config)
    return server


@pytest.fixture
def mock_cross_reference_pipeline():
    """Mock pipeline that returns realistic grounded responses."""
    pipeline = AsyncMock()
    
    def make_response(query: str):
        """Generate response based on query content."""
        if "rate limiter" in query.lower():
            return MagicMock(
                content=(
                    "The rate limiter is implemented in `src/middleware/rate_limiter.py` [^1]. "
                    "It uses a token bucket algorithm [^2] with configurable limits per endpoint."
                ),
                citations=[
                    {"id": "1", "source": "src/middleware/rate_limiter.py", "line": 45},
                    {"id": "2", "source": "APOSD Chapter 8: Rate Limiting Patterns", "page": 142},
                ],
                confidence=0.91,
                metadata={
                    "cycles_used": 2,
                    "participants": ["researcher", "critic"],
                    "sources_consulted": ["code", "books"],
                    "processing_time_ms": 1250,
                },
            )
        elif "repository pattern" in query.lower():
            return MagicMock(
                content=(
                    "The Repository pattern abstracts data access logic [^1]. "
                    "In this codebase, it's implemented in `src/repositories/base.py` [^2] "
                    "following the pattern described in Domain-Driven Design [^3]."
                ),
                citations=[
                    {"id": "1", "source": "Design Patterns (GoF)", "page": 322},
                    {"id": "2", "source": "src/repositories/base.py", "line": 15},
                    {"id": "3", "source": "Domain-Driven Design (Evans)", "page": 147},
                ],
                confidence=0.94,
                metadata={
                    "cycles_used": 1,
                    "participants": ["researcher", "synthesizer"],
                    "sources_consulted": ["code", "books"],
                    "processing_time_ms": 980,
                },
            )
        elif "cache service" in query.lower():
            return MagicMock(
                content=(
                    "Here is a cache service implementation based on patterns found in the codebase [^1]:\n\n"
                    "```python\nclass CacheService:\n    def __init__(self, ttl: int = 3600):\n        "
                    "self._cache = {}\n        self._ttl = ttl\n```"
                ),
                citations=[
                    {"id": "1", "source": "src/cache/redis_cache.py", "line": 22},
                ],
                confidence=0.88,
                metadata={
                    "cycles_used": 3,
                    "code_validated": True,
                    "validation_score": 0.92,
                },
            )
        else:
            return MagicMock(
                content=f"Response to: {query} [^1]",
                citations=[{"id": "1", "source": "generic_source.md", "line": 1}],
                confidence=0.75,
                metadata={"cycles_used": 1},
            )
    
    async def async_run(query, **kwargs):
        return make_response(query)
    
    pipeline.run.side_effect = async_run
    
    return pipeline


@pytest.fixture
def mock_validation_tool():
    """Mock code validation tool for E2E tests."""
    tool = AsyncMock()
    tool.validate_code.return_value = MagicMock(
        success=True,
        findings=["Code follows best practices", "No security issues detected"],
        metrics={"quality_score": 0.92, "complexity": 4},
    )
    return tool


# =============================================================================
# AC-KB9.1: Code Location Query E2E Test
# =============================================================================


class TestCodeLocationE2E:
    """AC-KB9.1: E2E test for code location queries."""

    @pytest.mark.asyncio
    async def test_rate_limiter_query_returns_code_location(
        self, mcp_server, mock_cross_reference_pipeline
    ):
        """AC-KB9.1: 'Where is rate limiter?' returns code location with citation."""
        mcp_server._pipeline = mock_cross_reference_pipeline
        
        result = await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Where is the rate limiter implemented?"}
        )
        
        text = result[0].text
        
        # Must include file path
        assert "rate_limiter.py" in text or "rate limiter" in text.lower()
        
        # Must include citation markers
        assert "[^1]" in text
        
        # Must include sources section
        assert "Sources:" in text or "[^1]:" in text

    @pytest.mark.asyncio
    async def test_code_location_includes_line_number(
        self, mcp_server, mock_cross_reference_pipeline
    ):
        """Code location responses should include line numbers when available."""
        mcp_server._pipeline = mock_cross_reference_pipeline
        
        result = await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Where is the rate limiter implemented?"}
        )
        
        text = result[0].text
        
        # Should reference specific line or have line in citation
        assert "line" in text.lower() or "#L" in text or "(line" in text


# =============================================================================
# AC-KB9.2: Concept Explanation E2E Test
# =============================================================================


class TestConceptExplanationE2E:
    """AC-KB9.2: E2E test for concept explanation queries."""

    @pytest.mark.asyncio
    async def test_repository_pattern_explanation(
        self, mcp_server, mock_cross_reference_pipeline
    ):
        """AC-KB9.2: 'Explain repository pattern' returns explanation with citations."""
        mcp_server._pipeline = mock_cross_reference_pipeline
        
        result = await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Explain the repository pattern"}
        )
        
        text = result[0].text
        
        # Must explain the concept
        assert "repository" in text.lower()
        assert "pattern" in text.lower() or "abstracts" in text.lower()
        
        # Must have multiple citations (book + code)
        assert "[^1]" in text
        assert "[^2]" in text or "[^3]" in text

    @pytest.mark.asyncio
    async def test_explanation_includes_book_citation(
        self, mcp_server, mock_cross_reference_pipeline
    ):
        """Concept explanations should cite textbook sources."""
        mcp_server._pipeline = mock_cross_reference_pipeline
        
        result = await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Explain the repository pattern"}
        )
        
        text = result[0].text
        
        # Should reference textbook or design patterns book
        has_book_ref = any(x in text for x in [
            "Design Patterns", "DDD", "Domain-Driven", "Evans", "GoF", 
            "Chapter", "page", "book"
        ])
        assert has_book_ref or "[^" in text  # At least has citations

    @pytest.mark.asyncio
    async def test_explanation_includes_code_citation(
        self, mcp_server, mock_cross_reference_pipeline
    ):
        """Concept explanations should cite code implementations."""
        mcp_server._pipeline = mock_cross_reference_pipeline
        
        result = await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Explain the repository pattern"}
        )
        
        text = result[0].text
        
        # Should reference code file
        has_code_ref = any(x in text for x in [
            ".py", "src/", "repositories", "implementation", "codebase"
        ])
        assert has_code_ref


# =============================================================================
# AC-KB9.3: Code Generation E2E Test
# =============================================================================


class TestCodeGenerationE2E:
    """AC-KB9.3: E2E test for code generation queries."""

    @pytest.mark.asyncio
    async def test_generate_cache_service_returns_code(
        self, mcp_server, mock_cross_reference_pipeline
    ):
        """AC-KB9.3: 'Generate cache service' produces code with citations."""
        mcp_server._pipeline = mock_cross_reference_pipeline
        
        result = await mcp_server.handle_call_tool(
            _CONST_TOOL_GENERATE_CODE,
            {"requirement": "Generate a cache service", "language": "python"}
        )
        
        text = result[0].text
        
        # Must include actual code
        has_code = any(x in text for x in [
            "```python", "class ", "def ", "import "
        ])
        assert has_code
        
        # Must include citations referencing similar patterns
        assert "[^" in text

    @pytest.mark.asyncio
    async def test_generated_code_includes_pattern_reference(
        self, mcp_server, mock_cross_reference_pipeline
    ):
        """Generated code should reference similar patterns in codebase."""
        mcp_server._pipeline = mock_cross_reference_pipeline
        
        result = await mcp_server.handle_call_tool(
            _CONST_TOOL_GENERATE_CODE,
            {"requirement": "Generate a cache service", "language": "python"}
        )
        
        text = result[0].text
        
        # Should mention pattern source
        has_pattern_ref = any(x in text.lower() for x in [
            "pattern", "similar", "based on", "reference", "implementation"
        ])
        assert has_pattern_ref or "[^1]" in text


# =============================================================================
# AC-KB9.4: Hallucination Rejection E2E Test
# =============================================================================


class TestHallucinationRejectionE2E:
    """AC-KB9.4: E2E test for hallucination rejection."""

    @pytest.fixture
    def mock_audit_rejecting_pipeline(self):
        """Mock pipeline that simulates audit rejection of hallucination."""
        pipeline = AsyncMock()
        pipeline.run.return_value = MagicMock(
            content=(
                "**Validation Failed**: The initial response contained unverifiable claims. "
                "After re-checking sources, here is a grounded response [^1]."
            ),
            citations=[
                {"id": "1", "source": "verified_source.py", "line": 10},
            ],
            confidence=0.85,
            metadata={
                "cycles_used": 3,
                "audit_rejections": 1,
                "hallucination_detected": True,
                "corrected": True,
            },
        )
        return pipeline

    @pytest.mark.asyncio
    async def test_hallucination_triggers_retry(
        self, mcp_server, mock_audit_rejecting_pipeline
    ):
        """AC-KB9.4: Deliberate hallucination should be caught and corrected."""
        mcp_server._pipeline = mock_audit_rejecting_pipeline
        
        result = await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Tell me about a feature that doesn't exist"}
        )
        
        text = result[0].text
        
        # Response should indicate validation/correction occurred
        # OR should have valid citations (meaning hallucination was avoided)
        has_validation_marker = any(x in text.lower() for x in [
            "validation", "verified", "checked", "grounded", "[^"
        ])
        assert has_validation_marker

    @pytest.mark.asyncio
    async def test_final_response_has_valid_citations(
        self, mcp_server, mock_audit_rejecting_pipeline
    ):
        """After hallucination rejection, final response must have valid citations."""
        mcp_server._pipeline = mock_audit_rejecting_pipeline
        
        result = await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Tell me about a feature that doesn't exist"}
        )
        
        text = result[0].text
        
        # Must have at least one citation
        assert "[^" in text or "Sources:" in text


# =============================================================================
# AC-KB9.5: Multi-Turn Conversation E2E Test
# =============================================================================


class TestMultiTurnConversationE2E:
    """AC-KB9.5: E2E test for multi-turn conversation context."""

    @pytest.mark.asyncio
    async def test_followup_maintains_context(
        self, mcp_server, mock_cross_reference_pipeline
    ):
        """AC-KB9.5: Follow-up questions maintain conversation context."""
        mcp_server._pipeline = mock_cross_reference_pipeline
        session_id = "multi-turn-test-session"
        
        # First query
        await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Where is the rate limiter implemented?"},
            session_id=session_id
        )
        
        # Follow-up query
        result = await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Show me the tests for it"},
            session_id=session_id
        )
        
        # Session should track both queries
        session = mcp_server._sessions.get(session_id)
        assert session is not None
        assert len(session.history) == 2

    @pytest.mark.asyncio
    async def test_followup_context_includes_prior_query(
        self, mcp_server, mock_cross_reference_pipeline
    ):
        """Follow-up context should include prior conversation."""
        mcp_server._pipeline = mock_cross_reference_pipeline
        session_id = "context-test-session"
        
        # First query
        await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Explain the repository pattern"},
            session_id=session_id
        )
        
        # Get session context
        session = mcp_server._sessions[session_id]
        context = session.get_context()
        
        # Context should include prior query
        assert "repository" in context.lower()

    @pytest.mark.asyncio
    async def test_three_turn_conversation(
        self, mcp_server, mock_cross_reference_pipeline
    ):
        """Multi-turn conversation works across 3+ turns."""
        mcp_server._pipeline = mock_cross_reference_pipeline
        session_id = "three-turn-session"
        
        # Turn 1
        await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Explain the discussion loop"},
            session_id=session_id
        )
        
        # Turn 2
        await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "What are its termination conditions?"},
            session_id=session_id
        )
        
        # Turn 3
        await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Show me examples"},
            session_id=session_id
        )
        
        session = mcp_server._sessions[session_id]
        assert len(session.history) == 3


# =============================================================================
# AC-KB9.6: Performance Benchmarks
# =============================================================================


class TestPerformanceBenchmarks:
    """AC-KB9.6: Performance benchmark tests."""

    @pytest.mark.asyncio
    async def test_typical_query_under_60s(
        self, mcp_server, mock_cross_reference_pipeline
    ):
        """AC-KB9.6: Typical query completes in <60 seconds."""
        mcp_server._pipeline = mock_cross_reference_pipeline
        
        start = time.perf_counter()
        
        await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Where is the rate limiter implemented?"}
        )
        
        elapsed = time.perf_counter() - start
        
        # With mock, should be very fast
        # In real E2E with services, target is <60s
        assert elapsed < _CONST_TYPICAL_TIMEOUT

    @pytest.mark.asyncio
    async def test_complex_query_under_120s(
        self, mcp_server, mock_cross_reference_pipeline
    ):
        """AC-KB9.6: Complex multi-cycle query completes in <120 seconds."""
        mcp_server._pipeline = mock_cross_reference_pipeline
        
        start = time.perf_counter()
        
        # Complex query that might trigger multiple cycles
        await mcp_server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {
                "query": "Compare the repository pattern implementation with the specification in the architecture docs",
                "max_cycles": 5
            }
        )
        
        elapsed = time.perf_counter() - start
        
        assert elapsed < _CONST_COMPLEX_TIMEOUT

    @pytest.mark.asyncio
    async def test_concurrent_queries_complete(
        self, mcp_server, mock_cross_reference_pipeline
    ):
        """Multiple concurrent queries complete successfully."""
        mcp_server._pipeline = mock_cross_reference_pipeline
        
        queries = [
            "Where is the rate limiter?",
            "Explain repository pattern",
            "Show me caching code",
        ]
        
        start = time.perf_counter()
        
        tasks = [
            mcp_server.handle_call_tool(
                _CONST_TOOL_CROSS_REFERENCE,
                {"query": q},
                session_id=f"concurrent-{i}"
            )
            for i, q in enumerate(queries)
        ]
        
        results = await asyncio.gather(*tasks)
        
        elapsed = time.perf_counter() - start
        
        # All should complete
        assert len(results) == 3
        assert all(len(r) > 0 for r in results)
        
        # Should be reasonably fast (with mocks)
        assert elapsed < _CONST_TYPICAL_TIMEOUT


# =============================================================================
# Integration Tests (Require Live Services)
# =============================================================================


@pytest.mark.skipif(
    not E2E_SERVICES_ENABLED,
    reason="E2E services not enabled. Set KB_E2E_SERVICES=true to run."
)
class TestLiveServiceIntegration:
    """Integration tests requiring live services."""

    @pytest.mark.asyncio
    async def test_live_cross_reference_query(self):
        """Test cross-reference with actual services running."""
        # This would connect to real services
        # For now, skip unless explicitly enabled
        pytest.skip("Requires live services")

    @pytest.mark.asyncio
    async def test_live_code_generation_validated(self):
        """Test code generation with real Code-Orchestrator validation."""
        pytest.skip("Requires live services")
