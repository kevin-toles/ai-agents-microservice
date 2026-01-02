"""Summarization Pipeline (Map-Reduce) - WBS-KB10.

Implements the Map-Reduce pattern for long content summarization:
1. Chunk content at semantic boundaries
2. Summarize chunks in parallel
3. Synthesize final summary

Acceptance Criteria:
- AC-KB10.3: SummarizationPipeline.run() handles input >50K tokens
- AC-KB10.4: ParallelAgent executes chunk summaries concurrently (max 4)
- AC-KB10.5: synthesize_outputs merges chunk summaries preserving key concepts
- AC-KB10.6: Final output respects token budget (default 4096)
- AC-KB10.7: Think tags stripped from intermediate outputs
- AC-KB10.8: Pipeline gracefully degrades if LLM unavailable
- AC-KB10.9: CompressionCache stores summaries for reuse
- AC-KB10.10: Pipeline registered as /v1/pipelines/summarize/run

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Map-Reduce Pattern
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from src.functions.utils.token_utils import estimate_tokens
from src.pipelines.chunking import Chunk, ChunkingConfig, ChunkingStrategy


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def _strip_think_tags(text: str) -> str:
    """Strip <think>...</think> tags from LLM output.

    AC-KB10.7: Think tags stripped from intermediate outputs.
    Thinking models (DeepSeek-R1) may spend tokens on reasoning
    that should not appear in final output.

    Args:
        text: Text potentially containing think tags

    Returns:
        Text with think tags and their content removed
    """
    # Pattern matches <think>...</think> including multiline content
    pattern = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    return pattern.sub("", text).strip()


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class InferenceClientProtocol(Protocol):
    """Protocol for inference client duck typing."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate text from prompt."""
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache duck typing."""

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache."""
        ...


# =============================================================================
# Configuration Schema
# =============================================================================


class SummarizationConfig(BaseModel):
    """Configuration for summarization pipeline.

    Attributes:
        output_token_budget: Maximum tokens for final output (default 4096)
        max_parallel_chunks: Maximum concurrent chunk summarizations (default 4)
        chunk_summary_budget: Max tokens per chunk summary
        max_retries: Retries on LLM failure
        retry_delay: Delay between retries in seconds
        cache_ttl: Cache TTL in seconds (default 24 hours)

    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Token Budget Allocation
    """

    model_config = ConfigDict(frozen=True)

    output_token_budget: int = Field(
        default=4096,
        ge=100,
        le=32000,
        description="Maximum tokens for final output",
    )
    max_parallel_chunks: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum concurrent chunk summarizations",
    )
    chunk_summary_budget: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum tokens per chunk summary",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Retries on LLM failure",
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Delay between retries in seconds",
    )
    cache_ttl: int = Field(
        default=86400,  # 24 hours
        ge=60,
        le=604800,  # 7 days
        description="Cache TTL in seconds",
    )

    # Chunking config embedded
    chunking_target_size: int = Field(
        default=2000,
        ge=100,
        le=8000,
        description="Target chunk size in tokens",
    )
    chunking_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap tokens between chunks",
    )


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ChunkSummary:
    """Summary of a single chunk.

    Attributes:
        chunk_index: Index of the source chunk
        content: Summary text
        key_concepts: Key concepts extracted from chunk
        token_count: Estimated tokens in summary
        metadata: Additional metadata
    """

    chunk_index: int
    content: str
    key_concepts: list[str] = field(default_factory=list)
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


class SummarizationResult(BaseModel):
    """Result from summarization pipeline.

    Attributes:
        final_summary: The synthesized final summary
        chunk_summaries: Individual chunk summaries
        key_concepts: All key concepts across chunks
        total_input_tokens: Total tokens in input
        output_tokens: Tokens in final output
        chunks_processed: Number of chunks processed
        processing_time_ms: Total processing time
        from_cache: Whether result was from cache
        metadata: Additional metadata
    """

    model_config = ConfigDict(frozen=False)

    final_summary: str = ""
    chunk_summaries: list[ChunkSummary] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    total_input_tokens: int = 0
    output_tokens: int = 0
    chunks_processed: int = 0
    processing_time_ms: float = 0.0
    from_cache: bool = False
    metadata: dict = Field(default_factory=dict)


# =============================================================================
# Compression Cache Integration
# =============================================================================


class CompressionCacheIntegration:
    """Integration layer for CompressionCache.

    Wraps cache operations with serialization/deserialization
    for SummarizationResult objects.
    """

    def __init__(self, cache: CacheProtocol, ttl: int = 86400) -> None:
        """Initialize cache integration.

        Args:
            cache: Cache instance
            ttl: Time-to-live in seconds
        """
        self._cache = cache
        self._ttl = ttl

    def _generate_key(self, content: str) -> str:
        """Generate cache key from content hash."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:32]
        return f"summarize:{content_hash}"

    async def get(self, content: str) -> SummarizationResult | None:
        """Get cached result for content.

        Args:
            content: Original content

        Returns:
            Cached result or None
        """
        key = self._generate_key(content)
        cached = await self._cache.get(key)

        if cached is None:
            return None

        if isinstance(cached, SummarizationResult):
            return cached

        # Deserialize if stored as dict
        if isinstance(cached, dict):
            return SummarizationResult(**cached)

        return None

    async def set(
        self,
        content: str,
        result: SummarizationResult,
    ) -> bool:
        """Cache result for content.

        Args:
            content: Original content
            result: Result to cache

        Returns:
            True if cached successfully
        """
        key = self._generate_key(content)
        return await self._cache.set(key, result, ttl=self._ttl)


# =============================================================================
# Summarization Pipeline
# =============================================================================


class SummarizationPipeline:
    """Map-Reduce summarization pipeline for long content.

    Implements the pattern:
    1. Chunk content at semantic boundaries
    2. Summarize chunks in parallel (map)
    3. Synthesize final summary (reduce)

    Example:
        >>> config = SummarizationConfig(output_token_budget=4096)
        >>> pipeline = SummarizationPipeline(config, inference_client)
        >>> result = await pipeline.run(long_document)

    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Map-Reduce Pattern
    """

    # Prompt templates
    CHUNK_SUMMARY_PROMPT = """Summarize the following content, extracting key concepts and main ideas.
Be concise but preserve important details.

Content:
{content}

Provide a summary in 2-4 paragraphs. List key concepts at the end.

Summary:"""

    SYNTHESIS_PROMPT = """Combine the following chunk summaries into a coherent, comprehensive summary.
Preserve all key concepts and ensure the final summary flows naturally.
Remove redundancy but don't lose important information.

Chunk Summaries:
{summaries}

Key Concepts from all chunks:
{concepts}

Create a unified summary that captures the essence of the entire content:"""

    def __init__(
        self,
        config: SummarizationConfig,
        inference_client: InferenceClientProtocol,
        cache: CacheProtocol | None = None,
    ) -> None:
        """Initialize summarization pipeline.

        Args:
            config: Pipeline configuration
            inference_client: Client for LLM inference
            cache: Optional cache for storing results
        """
        self.config = config
        self._inference = inference_client
        self._cache_integration = (
            CompressionCacheIntegration(cache, config.cache_ttl)
            if cache else None
        )

        # Initialize chunking strategy
        chunking_config = ChunkingConfig(
            target_chunk_size=config.chunking_target_size,
            overlap_tokens=config.chunking_overlap,
        )
        self._chunker = ChunkingStrategy(chunking_config)

        # Semaphore for parallel limit
        self._semaphore = asyncio.Semaphore(config.max_parallel_chunks)

    async def run(self, content: str) -> SummarizationResult:
        """Run summarization pipeline on content.

        AC-KB10.3: Handles input >50K tokens via chunking
        AC-KB10.9: Uses cache when available

        Args:
            content: Content to summarize

        Returns:
            SummarizationResult with final summary and metadata
        """
        start_time = time.time()

        # Handle empty input
        if not content or not content.strip():
            return SummarizationResult(
                final_summary="",
                chunks_processed=0,
                processing_time_ms=0,
            )

        # Check cache
        if self._cache_integration:
            cached = await self._cache_integration.get(content)
            if cached:
                cached.from_cache = True
                return cached

        # Estimate input tokens
        total_input_tokens = estimate_tokens(content)

        # Chunk content
        chunks = self._chunker.chunk(content)

        if not chunks:
            return SummarizationResult(
                final_summary="",
                total_input_tokens=total_input_tokens,
                chunks_processed=0,
                processing_time_ms=_elapsed_ms(start_time),
            )

        # Summarize chunks in parallel
        chunk_summaries = await self._summarize_chunks_parallel(chunks)

        # Synthesize final summary
        final_summary = await self._synthesize_outputs(chunk_summaries)

        # Collect all key concepts
        all_concepts = []
        for cs in chunk_summaries:
            all_concepts.extend(cs.key_concepts)
        unique_concepts = list(dict.fromkeys(all_concepts))  # Preserve order, dedupe

        # Build result
        result = SummarizationResult(
            final_summary=final_summary,
            chunk_summaries=chunk_summaries,
            key_concepts=unique_concepts,
            total_input_tokens=total_input_tokens,
            output_tokens=estimate_tokens(final_summary),
            chunks_processed=len(chunks),
            processing_time_ms=_elapsed_ms(start_time),
        )

        # Cache result
        if self._cache_integration:
            await self._cache_integration.set(content, result)

        return result

    async def _summarize_chunks_parallel(
        self,
        chunks: list[Chunk],
    ) -> list[ChunkSummary]:
        """Summarize chunks in parallel with concurrency limit.

        AC-KB10.4: Max 4 parallel (configurable via max_parallel_chunks)

        Args:
            chunks: Chunks to summarize

        Returns:
            List of chunk summaries
        """
        tasks = [
            self._summarize_single_chunk(chunk)
            for chunk in chunks
        ]

        summaries = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions, log them
        results = []
        for i, summary in enumerate(summaries):
            if isinstance(summary, Exception):
                logger.warning(f"Chunk {i} summarization failed: {summary}")
                # Create fallback summary
                results.append(
                    ChunkSummary(
                        chunk_index=i,
                        content=f"[Summary unavailable for chunk {i}]",
                        token_count=10,
                    )
                )
            else:
                results.append(summary)

        return results

    async def _summarize_single_chunk(self, chunk: Chunk) -> ChunkSummary:
        """Summarize a single chunk with retry logic.

        AC-KB10.7: Strips think tags from output
        AC-KB10.8: Retries on failure, falls back gracefully

        Args:
            chunk: Chunk to summarize

        Returns:
            ChunkSummary
        """
        async with self._semaphore:
            prompt = self.CHUNK_SUMMARY_PROMPT.format(content=chunk.content)

            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await self._inference.generate(
                        prompt=prompt,
                        max_tokens=self.config.chunk_summary_budget,
                    )

                    # Extract text from response
                    text = _extract_text(response)

                    # Strip think tags
                    text = _strip_think_tags(text)

                    # Extract key concepts (simple heuristic)
                    concepts = self._extract_concepts(text)

                    return ChunkSummary(
                        chunk_index=chunk.index,
                        content=text,
                        key_concepts=concepts,
                        token_count=estimate_tokens(text),
                    )

                except Exception as e:
                    logger.warning(
                        f"Chunk {chunk.index} attempt {attempt + 1} failed: {e}"
                    )
                    if attempt < self.config.max_retries:
                        await asyncio.sleep(self.config.retry_delay)
                    continue

            # Fallback: return truncated original content
            fallback_content = chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content
            return ChunkSummary(
                chunk_index=chunk.index,
                content=f"[Fallback] {fallback_content}",
                token_count=estimate_tokens(fallback_content),
            )

    async def _synthesize_outputs(
        self,
        chunk_summaries: list[ChunkSummary],
    ) -> str:
        """Synthesize chunk summaries into final summary.

        AC-KB10.5: Merges summaries preserving key concepts
        AC-KB10.6: Respects output token budget

        Args:
            chunk_summaries: Summaries to synthesize

        Returns:
            Final synthesized summary
        """
        if not chunk_summaries:
            return ""

        # If only one chunk, return its summary directly
        if len(chunk_summaries) == 1:
            return _strip_think_tags(chunk_summaries[0].content)

        # Build summaries text
        summaries_text = "\n\n".join(
            f"[Chunk {cs.chunk_index + 1}]\n{cs.content}"
            for cs in chunk_summaries
        )

        # Collect all concepts
        all_concepts = []
        for cs in chunk_summaries:
            all_concepts.extend(cs.key_concepts)
        concepts_text = ", ".join(dict.fromkeys(all_concepts))  # Dedupe

        prompt = self.SYNTHESIS_PROMPT.format(
            summaries=summaries_text,
            concepts=concepts_text or "N/A",
        )

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._inference.generate(
                    prompt=prompt,
                    max_tokens=self.config.output_token_budget,
                )

                text = _extract_text(response)
                return _strip_think_tags(text)

            except Exception as e:
                logger.warning(f"Synthesis attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)
                continue

        # Fallback: concatenate chunk summaries
        return "\n\n".join(cs.content for cs in chunk_summaries)

    def _extract_concepts(self, text: str) -> list[str]:
        """Extract key concepts from summary text.

        Simple heuristic: look for "Key concepts:" section or
        extract capitalized noun phrases.

        Args:
            text: Summary text

        Returns:
            List of key concepts
        """
        concepts = []

        # Look for explicit key concepts section
        match = re.search(
            r"(?:key concepts?|main (?:ideas?|points?)):\s*(.+?)(?:\n\n|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        if match:
            concepts_text = match.group(1)
            # Split by common delimiters
            items = re.split(r"[,;\n•\-\*]", concepts_text)
            for item in items:
                item = item.strip()
                if item and len(item) > 2:
                    concepts.append(item)

        # If no explicit section, extract capitalized terms
        if not concepts:
            # Find capitalized phrases (2+ words)
            caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
            concepts = list(dict.fromkeys(caps))[:5]  # Take top 5

        return concepts[:10]  # Limit to 10 concepts


# =============================================================================
# Utility Functions
# =============================================================================


def _elapsed_ms(start_time: float) -> float:
    """Calculate elapsed time in milliseconds."""
    return (time.time() - start_time) * 1000


def _extract_text(response: Any) -> str:
    """Extract text from various response formats."""
    if isinstance(response, str):
        return response

    if hasattr(response, "text"):
        return response.text

    if hasattr(response, "content"):
        return response.content

    if isinstance(response, dict):
        return response.get("text", response.get("content", str(response)))

    return str(response)


__all__ = [
    "ChunkSummary",
    "CompressionCacheIntegration",
    "SummarizationConfig",
    "SummarizationPipeline",
    "SummarizationResult",
    "_strip_think_tags",
]
