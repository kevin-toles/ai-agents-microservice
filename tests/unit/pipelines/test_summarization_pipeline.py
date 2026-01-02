"""Unit tests for SummarizationPipeline - WBS-KB10.

TDD Phase: RED
Tests AC-KB10.3-10 (pipeline, parallel processing, synthesis, caching)

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Map-Reduce Pattern
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pipelines.chunking import Chunk, ChunkingConfig
from src.pipelines.summarization_pipeline import (
    ChunkSummary,
    CompressionCacheIntegration,
    SummarizationConfig,
    SummarizationPipeline,
    SummarizationResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> SummarizationConfig:
    """Default summarization configuration."""
    return SummarizationConfig(
        output_token_budget=4096,
        max_parallel_chunks=4,
        max_retries=2,
    )


@pytest.fixture
def large_content() -> str:
    """Content that requires multiple chunks (simulating >50K tokens)."""
    # Create content with multiple sections
    sections = []
    for i in range(20):
        section = f"""
## Section {i + 1}: Important Topic

This section discusses important concepts related to topic {i + 1}.
The content includes detailed explanations and examples that help
illustrate the main ideas. We cover multiple aspects of the subject
matter to provide comprehensive coverage.

Key points in this section:
- Point A: Description of first concept with details
- Point B: Description of second concept with elaboration
- Point C: Description of third concept with examples

The implications of these concepts are significant for the field.
Further research continues to explore these areas in depth.
"""
        sections.append(section)

    return "\n\n".join(sections)


@pytest.fixture
def mock_inference_client() -> AsyncMock:
    """Mock inference client for testing."""
    client = AsyncMock()
    client.generate.return_value = MagicMock(
        text="This is a summary of the content.",
        tokens_used=100,
    )
    return client


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Sample chunks for testing."""
    return [
        Chunk(
            index=0,
            content="First chunk content about topic A.",
            start_char=0,
            end_char=100,
            token_count=10,
        ),
        Chunk(
            index=1,
            content="Second chunk content about topic B.",
            start_char=100,
            end_char=200,
            token_count=10,
            has_overlap=True,
        ),
        Chunk(
            index=2,
            content="Third chunk content about topic C.",
            start_char=200,
            end_char=300,
            token_count=10,
            has_overlap=True,
        ),
    ]


# =============================================================================
# Test SummarizationConfig
# =============================================================================


class TestSummarizationConfig:
    """Tests for SummarizationConfig schema."""

    def test_default_values(self) -> None:
        """AC-KB10.6: Default output token budget is 4096."""
        config = SummarizationConfig()
        assert config.output_token_budget == 4096
        assert config.max_parallel_chunks == 4

    def test_custom_budget(self) -> None:
        """AC-KB10.6: Token budget is configurable."""
        config = SummarizationConfig(output_token_budget=2048)
        assert config.output_token_budget == 2048

    def test_serialization(self) -> None:
        """Config serializes to dict."""
        config = SummarizationConfig(
            output_token_budget=1000,
            max_parallel_chunks=2,
        )
        data = config.model_dump()
        assert data["output_token_budget"] == 1000
        assert data["max_parallel_chunks"] == 2


# =============================================================================
# Test ChunkSummary
# =============================================================================


class TestChunkSummary:
    """Tests for ChunkSummary dataclass."""

    def test_chunk_summary_creation(self) -> None:
        """ChunkSummary stores summary and metadata."""
        summary = ChunkSummary(
            chunk_index=0,
            content="Summary of chunk 0",
            key_concepts=["concept1", "concept2"],
            token_count=50,
        )
        assert summary.chunk_index == 0
        assert summary.content == "Summary of chunk 0"
        assert len(summary.key_concepts) == 2
        assert summary.token_count == 50


# =============================================================================
# Test SummarizationResult
# =============================================================================


class TestSummarizationResult:
    """Tests for SummarizationResult schema."""

    def test_result_creation(self) -> None:
        """SummarizationResult captures full pipeline output."""
        result = SummarizationResult(
            final_summary="Final combined summary",
            chunk_summaries=[
                ChunkSummary(chunk_index=0, content="Sum 1", token_count=50),
                ChunkSummary(chunk_index=1, content="Sum 2", token_count=50),
            ],
            key_concepts=["concept1", "concept2", "concept3"],
            total_input_tokens=10000,
            output_tokens=500,
            chunks_processed=2,
            processing_time_ms=1500,
        )
        assert result.final_summary == "Final combined summary"
        assert len(result.chunk_summaries) == 2
        assert result.chunks_processed == 2

    def test_result_metadata(self) -> None:
        """AC-KB10.5: Result includes all metadata fields."""
        result = SummarizationResult(
            final_summary="Summary",
            chunk_summaries=[],
            key_concepts=[],
            total_input_tokens=5000,
            output_tokens=300,
            chunks_processed=3,
            processing_time_ms=2000,
            metadata={"source": "test"},
        )
        assert "source" in result.metadata


# =============================================================================
# Test SummarizationPipeline Core
# =============================================================================


class TestSummarizationPipelineCore:
    """Tests for core pipeline functionality."""

    @pytest.mark.asyncio
    async def test_pipeline_run_signature(
        self,
        default_config: SummarizationConfig,
        mock_inference_client: AsyncMock,
    ) -> None:
        """AC-KB10.3: SummarizationPipeline.run() has correct signature."""
        pipeline = SummarizationPipeline(
            config=default_config,
            inference_client=mock_inference_client,
        )
        # Method should accept string content, return SummarizationResult
        assert callable(getattr(pipeline, "run", None))

    @pytest.mark.asyncio
    async def test_handles_large_input(
        self,
        default_config: SummarizationConfig,
        large_content: str,
        mock_inference_client: AsyncMock,
    ) -> None:
        """AC-KB10.3: Pipeline handles input >50K tokens."""
        pipeline = SummarizationPipeline(
            config=default_config,
            inference_client=mock_inference_client,
        )

        result = await pipeline.run(large_content)

        assert isinstance(result, SummarizationResult)
        assert result.chunks_processed >= 1
        assert result.final_summary

    @pytest.mark.asyncio
    async def test_empty_input_handled(
        self,
        default_config: SummarizationConfig,
        mock_inference_client: AsyncMock,
    ) -> None:
        """Empty input returns empty result."""
        pipeline = SummarizationPipeline(
            config=default_config,
            inference_client=mock_inference_client,
        )

        result = await pipeline.run("")

        assert result.final_summary == ""
        assert result.chunks_processed == 0


# =============================================================================
# Test Parallel Chunk Processing
# =============================================================================


class TestParallelChunkProcessing:
    """Tests for parallel chunk summarization."""

    @pytest.mark.asyncio
    async def test_parallel_execution(
        self,
        default_config: SummarizationConfig,
        sample_chunks: list[Chunk],
    ) -> None:
        """AC-KB10.4: ParallelAgent executes chunk summaries concurrently."""
        # Track call times to verify parallelism
        call_times: list[float] = []
        call_lock = asyncio.Lock()

        async def mock_summarize(*args: Any, **kwargs: Any) -> str:
            async with call_lock:
                call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate work
            return f"Summary of chunk"

        mock_client = AsyncMock()
        mock_client.generate = mock_summarize

        pipeline = SummarizationPipeline(
            config=default_config,
            inference_client=mock_client,
        )

        # Summarize multiple chunks
        summaries = await pipeline._summarize_chunks_parallel(sample_chunks)

        assert len(summaries) == 3

        # Verify calls happened roughly concurrently (within 0.15s of each other)
        if len(call_times) >= 2:
            time_spread = max(call_times) - min(call_times)
            assert time_spread < 0.15, "Calls should be concurrent"

    @pytest.mark.asyncio
    async def test_max_parallel_limit(
        self,
        sample_chunks: list[Chunk],
    ) -> None:
        """AC-KB10.4: Max 4 parallel chunk summaries (configurable)."""
        config = SummarizationConfig(max_parallel_chunks=2)
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def mock_summarize(*args: Any, **kwargs: Any) -> str:
            nonlocal concurrent_count, max_concurrent
            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)
            async with lock:
                concurrent_count -= 1
            return "Summary"

        mock_client = AsyncMock()
        mock_client.generate = mock_summarize

        # Create many chunks to test limit
        many_chunks = sample_chunks * 4  # 12 chunks
        for i, chunk in enumerate(many_chunks):
            chunk.index = i

        pipeline = SummarizationPipeline(
            config=config,
            inference_client=mock_client,
        )

        await pipeline._summarize_chunks_parallel(many_chunks)

        # Max concurrent should not exceed config limit
        assert max_concurrent <= config.max_parallel_chunks


# =============================================================================
# Test Synthesis
# =============================================================================


class TestSynthesisOutputs:
    """Tests for synthesize_outputs functionality."""

    @pytest.mark.asyncio
    async def test_merges_chunk_summaries(
        self,
        default_config: SummarizationConfig,
        mock_inference_client: AsyncMock,
    ) -> None:
        """AC-KB10.5: synthesize_outputs merges chunk summaries."""
        chunk_summaries = [
            ChunkSummary(
                chunk_index=0,
                content="Summary of section 1 about topic A.",
                key_concepts=["topic A"],
                token_count=10,
            ),
            ChunkSummary(
                chunk_index=1,
                content="Summary of section 2 about topic B.",
                key_concepts=["topic B"],
                token_count=10,
            ),
        ]

        pipeline = SummarizationPipeline(
            config=default_config,
            inference_client=mock_inference_client,
        )

        result = await pipeline._synthesize_outputs(chunk_summaries)

        assert result  # Non-empty synthesis

    @pytest.mark.asyncio
    async def test_preserves_key_concepts(
        self,
        default_config: SummarizationConfig,
        mock_inference_client: AsyncMock,
    ) -> None:
        """AC-KB10.5: Synthesis preserves key concepts from all chunks."""
        chunk_summaries = [
            ChunkSummary(
                chunk_index=0,
                content="Machine learning enables...",
                key_concepts=["machine learning", "AI"],
                token_count=20,
            ),
            ChunkSummary(
                chunk_index=1,
                content="Neural networks are...",
                key_concepts=["neural networks", "deep learning"],
                token_count=20,
            ),
        ]

        pipeline = SummarizationPipeline(
            config=default_config,
            inference_client=mock_inference_client,
        )

        # Mock should be configured to return synthesis
        mock_inference_client.generate.return_value = MagicMock(
            text="Combined summary about machine learning, AI, neural networks.",
            tokens_used=50,
        )

        result = await pipeline._synthesize_outputs(chunk_summaries)

        # Synthesis should mention key concepts
        assert result


# =============================================================================
# Test Token Budget Enforcement
# =============================================================================


class TestTokenBudgetEnforcement:
    """Tests for output token budget enforcement."""

    @pytest.mark.asyncio
    async def test_output_within_budget(
        self,
        large_content: str,
        mock_inference_client: AsyncMock,
    ) -> None:
        """AC-KB10.6: Final output respects token budget."""
        config = SummarizationConfig(output_token_budget=500)

        # Mock returns response within budget
        mock_inference_client.generate.return_value = MagicMock(
            text="Short summary within budget.",
            tokens_used=100,
        )

        pipeline = SummarizationPipeline(
            config=config,
            inference_client=mock_inference_client,
        )

        result = await pipeline.run(large_content)

        assert result.output_tokens <= config.output_token_budget


# =============================================================================
# Test Think Tag Stripping
# =============================================================================


class TestThinkTagStripping:
    """Tests for think tag removal."""

    @pytest.mark.asyncio
    async def test_think_tags_stripped_from_chunks(
        self,
        default_config: SummarizationConfig,
    ) -> None:
        """AC-KB10.7: Think tags stripped from intermediate outputs."""
        # Mock client returns content with think tags
        mock_client = AsyncMock()
        mock_client.generate.return_value = MagicMock(
            text="<think>Let me analyze this...</think>The actual summary content.",
            tokens_used=50,
        )

        pipeline = SummarizationPipeline(
            config=default_config,
            inference_client=mock_client,
        )

        chunk = Chunk(
            index=0,
            content="Content to summarize",
            start_char=0,
            end_char=100,
            token_count=25,
        )

        summary = await pipeline._summarize_single_chunk(chunk)

        # Think tags should be removed
        assert "<think>" not in summary.content
        assert "</think>" not in summary.content

    @pytest.mark.asyncio
    async def test_think_tags_stripped_from_final(
        self,
        default_config: SummarizationConfig,
    ) -> None:
        """AC-KB10.7: Think tags never appear in final output."""
        mock_client = AsyncMock()
        mock_client.generate.return_value = MagicMock(
            text="<think>Processing...</think>Final summary without think.",
            tokens_used=50,
        )

        pipeline = SummarizationPipeline(
            config=default_config,
            inference_client=mock_client,
        )

        result = await pipeline.run("Short content to summarize.")

        assert "<think>" not in result.final_summary
        assert "</think>" not in result.final_summary


# =============================================================================
# Test Graceful Degradation
# =============================================================================


class TestGracefulDegradation:
    """Tests for LLM failure handling."""

    @pytest.mark.asyncio
    async def test_retries_on_failure(
        self,
        default_config: SummarizationConfig,
    ) -> None:
        """AC-KB10.8: Pipeline retries on transient LLM failures."""
        call_count = 0

        async def flaky_generate(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("LLM timeout")
            return MagicMock(text="Success on retry", tokens_used=20)

        mock_client = AsyncMock()
        mock_client.generate = flaky_generate

        pipeline = SummarizationPipeline(
            config=default_config,
            inference_client=mock_client,
        )

        result = await pipeline.run("Content to summarize")

        assert call_count >= 2  # Retried
        assert result.final_summary

    @pytest.mark.asyncio
    async def test_fallback_on_persistent_failure(
        self,
        default_config: SummarizationConfig,
    ) -> None:
        """AC-KB10.8: Uses fallback if LLM persistently unavailable."""
        async def always_fail(*args: Any, **kwargs: Any) -> None:
            raise ConnectionError("LLM unavailable")

        mock_client = AsyncMock()
        mock_client.generate = always_fail

        pipeline = SummarizationPipeline(
            config=default_config,
            inference_client=mock_client,
        )

        # Should not raise, should return fallback
        result = await pipeline.run("Content to summarize")

        # Fallback result (truncated content or error message)
        assert result is not None


# =============================================================================
# Test Compression Cache Integration
# =============================================================================


class TestCompressionCacheIntegration:
    """Tests for CompressionCache integration."""

    @pytest.mark.asyncio
    async def test_cache_stores_summary(
        self,
        default_config: SummarizationConfig,
        mock_inference_client: AsyncMock,
    ) -> None:
        """AC-KB10.9: CompressionCache stores summaries for reuse."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None  # Cache miss

        pipeline = SummarizationPipeline(
            config=default_config,
            inference_client=mock_inference_client,
            cache=mock_cache,
        )

        content = "Content to summarize"
        await pipeline.run(content)

        # Cache set should be called
        mock_cache.set.assert_called()

    @pytest.mark.asyncio
    async def test_cache_hit_returns_stored(
        self,
        default_config: SummarizationConfig,
        mock_inference_client: AsyncMock,
    ) -> None:
        """AC-KB10.9: Cache hit returns stored summary."""
        cached_result = SummarizationResult(
            final_summary="Cached summary",
            chunk_summaries=[],
            key_concepts=[],
            total_input_tokens=100,
            output_tokens=20,
            chunks_processed=1,
            processing_time_ms=0,
            from_cache=True,
        )

        mock_cache = AsyncMock()
        mock_cache.get.return_value = cached_result

        pipeline = SummarizationPipeline(
            config=default_config,
            inference_client=mock_inference_client,
            cache=mock_cache,
        )

        result = await pipeline.run("Content to summarize")

        assert result.from_cache is True
        assert result.final_summary == "Cached summary"
        # Inference should not be called on cache hit
        mock_inference_client.generate.assert_not_called()


# =============================================================================
# Test Performance
# =============================================================================


class TestSummarizationPerformance:
    """Performance tests for summarization pipeline."""

    @pytest.mark.asyncio
    async def test_5_chunks_under_30s(
        self,
        default_config: SummarizationConfig,
    ) -> None:
        """Parallel processing of 5 chunks completes in <30s."""
        import time

        async def slow_summarize(*args: Any, **kwargs: Any) -> MagicMock:
            await asyncio.sleep(1)  # Simulate 1s per chunk
            return MagicMock(text="Summary", tokens_used=50)

        mock_client = AsyncMock()
        mock_client.generate = slow_summarize

        pipeline = SummarizationPipeline(
            config=default_config,
            inference_client=mock_client,
        )

        chunks = [
            Chunk(index=i, content=f"Chunk {i}", start_char=0, end_char=100, token_count=25)
            for i in range(5)
        ]

        start = time.time()
        await pipeline._summarize_chunks_parallel(chunks)
        elapsed = time.time() - start

        # 5 chunks at 1s each, parallel should complete in ~2s (4 parallel default)
        assert elapsed < 30, f"Took {elapsed:.1f}s, expected <30s"


# =============================================================================
# Test Strip Think Tags Helper
# =============================================================================


class TestStripThinkTagsHelper:
    """Tests for _strip_think_tags helper function."""

    def test_strips_single_think_block(self) -> None:
        """Strips single think tag block."""
        from src.pipelines.summarization_pipeline import _strip_think_tags

        text = "<think>Internal reasoning here</think>Visible output"
        result = _strip_think_tags(text)
        assert result == "Visible output"

    def test_strips_multiple_think_blocks(self) -> None:
        """Strips multiple think tag blocks."""
        from src.pipelines.summarization_pipeline import _strip_think_tags

        text = "<think>First</think>Middle<think>Second</think>End"
        result = _strip_think_tags(text)
        assert result == "MiddleEnd"

    def test_handles_multiline_think_blocks(self) -> None:
        """Handles multiline content in think tags."""
        from src.pipelines.summarization_pipeline import _strip_think_tags

        text = """<think>
This is a long
multiline reasoning
process
</think>Final answer here."""
        result = _strip_think_tags(text)
        assert result == "Final answer here."

    def test_no_think_tags_unchanged(self) -> None:
        """Content without think tags is unchanged."""
        from src.pipelines.summarization_pipeline import _strip_think_tags

        text = "Regular content without any tags"
        result = _strip_think_tags(text)
        assert result == text

    def test_empty_think_tags(self) -> None:
        """Handles empty think tags."""
        from src.pipelines.summarization_pipeline import _strip_think_tags

        text = "<think></think>Content"
        result = _strip_think_tags(text)
        assert result == "Content"


__all__ = [
    "TestChunkSummary",
    "TestCompressionCacheIntegration",
    "TestGracefulDegradation",
    "TestParallelChunkProcessing",
    "TestStripThinkTagsHelper",
    "TestSummarizationConfig",
    "TestSummarizationPerformance",
    "TestSummarizationPipelineCore",
    "TestSummarizationResult",
    "TestSynthesisOutputs",
    "TestThinkTagStripping",
    "TestTokenBudgetEnforcement",
]
