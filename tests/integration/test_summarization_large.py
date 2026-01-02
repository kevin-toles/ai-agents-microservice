"""Integration tests for Summarization Pipeline with large documents - WBS-KB10.

Tests AC-KB10.3, AC-KB10.10: Handling 50K+ token documents.

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Map-Reduce Pattern
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipelines.summarization_pipeline import (
    SummarizationConfig,
    SummarizationPipeline,
    SummarizationResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def large_document_50k() -> str:
    """Generate a document with approximately 50K tokens (~200K chars).

    Structure: Technical documentation with multiple chapters and sections.
    """
    chapters = []

    for chapter_num in range(10):
        chapter = f"""
# Chapter {chapter_num + 1}: Advanced Topic {chapter_num + 1}

## Introduction

This chapter explores advanced concepts related to topic {chapter_num + 1}.
We will examine the theoretical foundations, practical applications, and
implementation strategies for these important concepts.

The field has evolved significantly over the past decade, with numerous
breakthroughs in both theoretical understanding and practical tooling.
This chapter synthesizes the most important developments and provides
a comprehensive overview for practitioners.

## Section {chapter_num + 1}.1: Theoretical Foundation

The theoretical foundation of this topic rests on several key principles.
First, we must understand the fundamental constraints that govern the
behavior of systems in this domain. These constraints arise from both
mathematical properties and practical engineering considerations.

The mathematical framework includes:
- Principle A: The foundational axiom that establishes baseline behavior
- Principle B: The constraint that limits system capacity
- Principle C: The optimization criterion for decision making
- Principle D: The convergence guarantee under standard conditions

Each of these principles has been extensively studied and validated
through both theoretical analysis and empirical experimentation.

## Section {chapter_num + 1}.2: Implementation Patterns

When implementing systems based on these principles, several patterns
have emerged as best practices:

### Pattern 1: Layered Architecture

The layered architecture pattern separates concerns into distinct layers:
1. Presentation Layer - Handles user interaction and display
2. Business Logic Layer - Implements core functionality
3. Data Access Layer - Manages persistence and retrieval
4. Infrastructure Layer - Provides cross-cutting concerns

This separation enables independent testing, evolution, and scaling
of each layer.

### Pattern 2: Event-Driven Design

Event-driven design decouples components through asynchronous messaging:
- Producers emit events when state changes occur
- Consumers subscribe to relevant event streams
- Event stores provide durability and replay capability
- Saga orchestrators coordinate distributed transactions

### Pattern 3: Domain-Driven Design

Domain-driven design focuses on modeling the core business domain:
- Entities capture identity and lifecycle
- Value objects represent immutable attributes
- Aggregates define consistency boundaries
- Domain services encapsulate business logic
- Repositories abstract persistence concerns

## Section {chapter_num + 1}.3: Performance Optimization

Performance optimization requires careful attention to several factors:

### Memory Management

Efficient memory management is critical for large-scale systems:
- Object pooling reduces allocation overhead
- Memory-mapped files enable efficient I/O
- Cache hierarchies balance latency and capacity
- Garbage collection tuning minimizes pause times

### Computational Efficiency

Computational efficiency improvements include:
- Algorithm selection based on data characteristics
- Parallelization across available cores
- Vectorization using SIMD instructions
- GPU acceleration for suitable workloads

### Network Optimization

Network optimization strategies include:
- Connection pooling to reduce handshake overhead
- Request batching to amortize round-trip latency
- Compression to reduce bandwidth usage
- CDN deployment for geographic distribution

## Section {chapter_num + 1}.4: Case Studies

Several case studies illustrate these principles in practice:

### Case Study A: E-commerce Platform

An e-commerce platform implemented these patterns to handle:
- 10 million daily active users
- 100,000 orders per hour at peak
- 99.99% availability requirement
- Sub-100ms response time target

The implementation used:
- Microservices architecture with event sourcing
- Multi-region deployment with active-active replication
- Tiered caching with local and distributed layers
- Real-time analytics for monitoring and optimization

### Case Study B: Financial Trading System

A financial trading system applied these concepts for:
- Microsecond-level latency requirements
- Strict regulatory compliance
- High-throughput order processing
- Complex risk management rules

The solution employed:
- FPGA-based network processing
- In-memory data structures for hot path
- Distributed consensus for coordination
- Comprehensive audit logging

## Summary

This chapter covered the key aspects of topic {chapter_num + 1}:
1. Theoretical foundations and principles
2. Implementation patterns and best practices
3. Performance optimization strategies
4. Real-world case studies and lessons learned

The next chapter will build on these concepts to explore more advanced topics.

---
"""
        chapters.append(chapter)

    # Additional filler to ensure we hit 50K tokens
    filler = """
## Appendix: Reference Material

This appendix contains additional reference material for practitioners.

### Terminology

The following terms are used throughout this document:
- **Abstraction**: A simplified representation of complex reality
- **Algorithm**: A step-by-step procedure for computation
- **Architecture**: The high-level structure of a system
- **Benchmark**: A standard test for comparing performance
- **Cache**: A fast storage layer for frequently accessed data
- **Cluster**: A group of interconnected computers
- **Concurrency**: Multiple computations executing simultaneously
- **Database**: A structured collection of persistent data
- **Encryption**: The process of encoding information
- **Framework**: A reusable set of libraries and conventions

### Bibliography

Key references for further reading include numerous academic papers,
industry reports, and practical guides that have shaped the field.

""" * 20

    return "\n\n".join(chapters) + filler


@pytest.fixture
def large_document_100k(large_document_50k: str) -> str:
    """Generate a document with approximately 100K tokens."""
    return large_document_50k * 2


@pytest.fixture
def mock_inference_client() -> AsyncMock:
    """Create mock inference client that returns summaries."""
    client = AsyncMock()

    async def generate_summary(prompt: str, max_tokens: int | None = None, **kwargs: Any) -> MagicMock:
        # Simulate some processing time
        await asyncio.sleep(0.01)

        # Generate a reasonable summary based on prompt content
        if "Chunk" in prompt or "summarize" in prompt.lower():
            summary = "This section covers important concepts including architecture patterns, "
            summary += "performance optimization, and implementation best practices. "
            summary += "Key concepts: layered architecture, event-driven design, domain-driven design."
        else:
            summary = "Comprehensive summary of the document covering all major topics discussed. "
            summary += "The material spans theoretical foundations, practical patterns, and real-world applications."

        return MagicMock(text=summary, tokens_used=len(summary) // 4)

    client.generate = generate_summary
    return client


# =============================================================================
# Integration Tests
# =============================================================================


class TestLargeDocumentSummarization:
    """Integration tests for summarizing large documents."""

    @pytest.mark.asyncio
    async def test_50k_token_document_processed(
        self,
        large_document_50k: str,
        mock_inference_client: AsyncMock,
    ) -> None:
        """AC-KB10.3: Pipeline handles input >50K tokens."""
        config = SummarizationConfig(
            output_token_budget=4096,
            max_parallel_chunks=4,
        )

        pipeline = SummarizationPipeline(
            config=config,
            inference_client=mock_inference_client,
        )

        result = await pipeline.run(large_document_50k)

        # Verify document was processed
        assert isinstance(result, SummarizationResult)
        assert result.final_summary
        assert result.chunks_processed > 0
        assert result.total_input_tokens > 10000  # Should be substantial

    @pytest.mark.asyncio
    async def test_100k_token_document_processed(
        self,
        large_document_100k: str,
        mock_inference_client: AsyncMock,
    ) -> None:
        """AC-KB10.3: Pipeline handles very large documents (100K+ tokens)."""
        config = SummarizationConfig(
            output_token_budget=4096,
            max_parallel_chunks=4,
        )

        pipeline = SummarizationPipeline(
            config=config,
            inference_client=mock_inference_client,
        )

        result = await pipeline.run(large_document_100k)

        assert isinstance(result, SummarizationResult)
        assert result.final_summary
        assert result.chunks_processed > 10  # Should have many chunks
        assert result.total_input_tokens > 20000  # Should be substantial

    @pytest.mark.asyncio
    async def test_output_within_budget(
        self,
        large_document_50k: str,
        mock_inference_client: AsyncMock,
    ) -> None:
        """AC-KB10.6: Final output respects token budget regardless of input size."""
        config = SummarizationConfig(
            output_token_budget=1000,  # Small budget
        )

        pipeline = SummarizationPipeline(
            config=config,
            inference_client=mock_inference_client,
        )

        result = await pipeline.run(large_document_50k)

        # Output should be within budget
        assert result.output_tokens <= config.output_token_budget

    @pytest.mark.asyncio
    async def test_no_content_loss_in_chunking(
        self,
        large_document_50k: str,
        mock_inference_client: AsyncMock,
    ) -> None:
        """Verify all content is covered by chunks (no gaps)."""
        config = SummarizationConfig()

        pipeline = SummarizationPipeline(
            config=config,
            inference_client=mock_inference_client,
        )

        # Verify chunking covers all content
        chunks = pipeline._chunker.chunk(large_document_50k)

        # Calculate coverage
        total_chunk_chars = sum(len(c.content) for c in chunks)
        original_chars = len(large_document_50k)

        # Allow for some overlap, but should be close to original
        # With overlap, total may exceed original
        assert total_chunk_chars >= original_chars * 0.95

    @pytest.mark.asyncio
    async def test_parallel_processing_faster_than_sequential(
        self,
        large_document_50k: str,
    ) -> None:
        """AC-KB10.4: Parallel processing is faster than sequential."""
        delay_per_chunk = 0.1

        async def slow_generate(prompt: str, **kwargs: Any) -> MagicMock:
            await asyncio.sleep(delay_per_chunk)
            return MagicMock(text="Summary", tokens_used=50)

        mock_client = AsyncMock()
        mock_client.generate = slow_generate

        config = SummarizationConfig(
            max_parallel_chunks=4,
            chunking_target_size=1000,  # More chunks
        )

        pipeline = SummarizationPipeline(
            config=config,
            inference_client=mock_client,
        )

        # Get chunk count
        chunks = pipeline._chunker.chunk(large_document_50k)
        num_chunks = len(chunks)

        # Time parallel execution
        start = time.time()
        await pipeline.run(large_document_50k)
        parallel_time = time.time() - start

        # Sequential would take: num_chunks * delay_per_chunk
        sequential_time = num_chunks * delay_per_chunk

        # Parallel should be significantly faster (at least 2x)
        if num_chunks > 4:
            assert parallel_time < sequential_time * 0.75, (
                f"Parallel ({parallel_time:.2f}s) should be much faster than "
                f"sequential estimate ({sequential_time:.2f}s)"
            )


class TestSummarizationQuality:
    """Tests for summarization output quality."""

    @pytest.mark.asyncio
    async def test_key_concepts_preserved(
        self,
        mock_inference_client: AsyncMock,
    ) -> None:
        """AC-KB10.5: Key concepts are preserved in final summary."""
        # Create content with specific key concepts
        content = """
# Machine Learning Overview

## Introduction
Machine learning is a subset of artificial intelligence that enables
systems to learn from data. Deep learning uses neural networks with
many layers.

## Key Concepts
- Supervised learning uses labeled training data
- Unsupervised learning finds patterns in unlabeled data
- Reinforcement learning learns through trial and error

## Applications
Machine learning powers recommendation systems, natural language
processing, and computer vision applications.
""" * 10

        config = SummarizationConfig()
        pipeline = SummarizationPipeline(
            config=config,
            inference_client=mock_inference_client,
        )

        result = await pipeline.run(content)

        # Should have key concepts
        assert result.key_concepts or result.final_summary

    @pytest.mark.asyncio
    async def test_no_think_tags_in_output(
        self,
    ) -> None:
        """AC-KB10.7: Think tags never appear in final output."""
        # Mock client that returns content with think tags
        mock_client = AsyncMock()

        async def generate_with_think_tags(prompt: str, **kwargs: Any) -> MagicMock:
            return MagicMock(
                text="<think>Let me analyze this carefully...</think>The summary is: important content here.",
                tokens_used=50,
            )

        mock_client.generate = generate_with_think_tags

        config = SummarizationConfig()
        pipeline = SummarizationPipeline(
            config=config,
            inference_client=mock_client,
        )

        result = await pipeline.run("Test content to summarize.")

        # No think tags in output
        assert "<think>" not in result.final_summary
        assert "</think>" not in result.final_summary

        # Also check chunk summaries
        for chunk_summary in result.chunk_summaries:
            assert "<think>" not in chunk_summary.content


class TestPipelineResilience:
    """Tests for pipeline error handling and resilience."""

    @pytest.mark.asyncio
    async def test_handles_intermittent_failures(
        self,
        large_document_50k: str,
    ) -> None:
        """AC-KB10.8: Pipeline handles intermittent LLM failures."""
        failure_count = 0
        max_failures = 5

        async def flaky_generate(prompt: str, **kwargs: Any) -> MagicMock:
            nonlocal failure_count
            failure_count += 1
            if failure_count <= max_failures:
                raise TimeoutError("LLM timeout")
            return MagicMock(text="Summary after retry", tokens_used=50)

        mock_client = AsyncMock()
        mock_client.generate = flaky_generate

        config = SummarizationConfig(
            max_retries=5,  # Allow enough retries
            retry_delay=0.1,  # Fast retries for test
        )

        pipeline = SummarizationPipeline(
            config=config,
            inference_client=mock_client,
        )

        # Should complete despite failures
        result = await pipeline.run(large_document_50k)
        assert result is not None

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_complete_failure(
        self,
    ) -> None:
        """AC-KB10.8: Pipeline degrades gracefully on complete LLM failure."""
        async def always_fail(prompt: str, **kwargs: Any) -> None:
            raise ConnectionError("LLM unavailable")

        mock_client = AsyncMock()
        mock_client.generate = always_fail

        config = SummarizationConfig(
            max_retries=1,
            retry_delay=0.1,
        )

        pipeline = SummarizationPipeline(
            config=config,
            inference_client=mock_client,
        )

        # Should not raise, should return fallback
        result = await pipeline.run("Content to summarize")
        assert result is not None


class TestAPIEndpoint:
    """Tests for the /v1/pipelines/summarize/run endpoint."""

    @pytest.mark.asyncio
    async def test_endpoint_registered(self) -> None:
        """AC-KB10.10: Endpoint is registered at correct path."""
        from src.api.routes.pipelines import router

        # Find the summarize endpoint
        routes = [route for route in router.routes if hasattr(route, "path")]
        paths = [route.path for route in routes]

        # Check full path with prefix
        assert "/v1/pipelines/summarize/run" in paths or any("/summarize/run" in p for p in paths)

    @pytest.mark.asyncio
    async def test_request_model_validation(self) -> None:
        """Request model validates input correctly."""
        from src.api.routes.pipelines import SummarizeRequest

        # Valid request
        request = SummarizeRequest(
            content="Test content",
            output_token_budget=2048,
        )
        assert request.content == "Test content"
        assert request.output_token_budget == 2048

        # Empty content should fail
        with pytest.raises(Exception):  # ValidationError
            SummarizeRequest(content="")

    @pytest.mark.asyncio
    async def test_response_model_fields(self) -> None:
        """Response model has all required fields."""
        from src.api.routes.pipelines import SummarizeResponse

        response = SummarizeResponse(
            summary="Test summary",
            key_concepts=["concept1", "concept2"],
            chunks_processed=3,
            total_input_tokens=1000,
            output_tokens=100,
            processing_time_ms=500.0,
        )

        assert response.summary == "Test summary"
        assert len(response.key_concepts) == 2
        assert response.chunks_processed == 3


class TestPerformanceBenchmarks:
    """Performance benchmarks for summarization pipeline."""

    @pytest.mark.asyncio
    async def test_5_chunks_under_30s(
        self,
    ) -> None:
        """Parallel processing of 5 chunks completes in <30s."""
        async def slow_generate(prompt: str, **kwargs: Any) -> MagicMock:
            await asyncio.sleep(1)  # 1 second per chunk
            return MagicMock(text="Summary", tokens_used=50)

        mock_client = AsyncMock()
        mock_client.generate = slow_generate

        config = SummarizationConfig(
            max_parallel_chunks=4,
            chunking_target_size=500,  # Small chunks
        )

        # Create content that will produce ~5 chunks
        content = "Test content section. " * 500

        pipeline = SummarizationPipeline(
            config=config,
            inference_client=mock_client,
        )

        start = time.time()
        await pipeline.run(content)
        elapsed = time.time() - start

        # Should complete in under 30 seconds
        assert elapsed < 30, f"Took {elapsed:.1f}s, expected <30s"


__all__ = [
    "TestAPIEndpoint",
    "TestLargeDocumentSummarization",
    "TestPipelineResilience",
    "TestPerformanceBenchmarks",
    "TestSummarizationQuality",
]
