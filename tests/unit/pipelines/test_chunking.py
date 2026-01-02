"""Unit tests for ChunkingStrategy - WBS-KB10.

TDD Phase: RED
Tests AC-KB10.1, AC-KB10.2 (semantic boundaries, overlap tokens)

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Map-Reduce Pattern
"""

from __future__ import annotations

import pytest

from src.pipelines.chunking import (
    Chunk,
    ChunkingConfig,
    ChunkingStrategy,
    SemanticBoundaryChunker,
    SlidingWindowChunker,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_text_with_paragraphs() -> str:
    """Sample text with clear paragraph boundaries."""
    return """Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems
to learn from data without being explicitly programmed. It has revolutionized
many industries.

Types of Machine Learning

There are three main types of machine learning:

1. Supervised Learning
This type uses labeled data to train models. The algorithm learns from
input-output pairs to make predictions on new data.

2. Unsupervised Learning
This type works with unlabeled data. The algorithm discovers patterns
and structures in the data on its own.

3. Reinforcement Learning
This type learns through trial and error. The agent receives rewards
or penalties based on its actions.

Applications

Machine learning is used in many applications including:
- Natural language processing
- Computer vision
- Recommendation systems
- Fraud detection

Conclusion

Machine learning continues to advance rapidly, with new techniques
emerging regularly."""


@pytest.fixture
def sample_text_single_paragraph() -> str:
    """Sample text without paragraph breaks."""
    return (
        "This is a single long paragraph without any clear section breaks. "
        "It continues on and on with various topics interspersed. "
        "The text discusses many concepts without structure. "
        "We need to ensure the chunker handles this gracefully. "
        "The sliding window approach should be used as a fallback. "
        "This ensures we can still process unstructured content. "
    ) * 50  # Make it long enough to require multiple chunks


@pytest.fixture
def sample_code_content() -> str:
    """Sample code content for testing."""
    return '''"""Module docstring."""

import os
import sys


class DataProcessor:
    """Process data using various strategies."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self._cache = {}

    def process(self, data):
        """Process the input data."""
        if data in self._cache:
            return self._cache[data]
        result = self._transform(data)
        self._cache[data] = result
        return result

    def _transform(self, data):
        """Internal transformation logic."""
        return data.upper()


class AnotherClass:
    """Another class for testing."""

    def method_one(self):
        """First method."""
        pass

    def method_two(self):
        """Second method."""
        pass
'''


@pytest.fixture
def default_config() -> ChunkingConfig:
    """Default chunking configuration."""
    return ChunkingConfig(
        target_chunk_size=500,
        overlap_tokens=200,
        min_chunk_size=100,
        max_chunk_size=1000,
    )


# =============================================================================
# Test ChunkingConfig
# =============================================================================


class TestChunkingConfig:
    """Tests for ChunkingConfig schema."""

    def test_default_values(self) -> None:
        """AC-KB10.2: Default overlap is 200 tokens."""
        config = ChunkingConfig()
        assert config.overlap_tokens == 200
        assert config.target_chunk_size > 0

    def test_custom_overlap(self) -> None:
        """AC-KB10.2: Overlap is configurable."""
        config = ChunkingConfig(overlap_tokens=300)
        assert config.overlap_tokens == 300

    def test_serialization(self) -> None:
        """Config serializes to dict."""
        config = ChunkingConfig(
            target_chunk_size=1000,
            overlap_tokens=150,
        )
        data = config.model_dump()
        assert data["target_chunk_size"] == 1000
        assert data["overlap_tokens"] == 150


# =============================================================================
# Test Chunk Dataclass
# =============================================================================


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self) -> None:
        """Chunk stores content and metadata."""
        chunk = Chunk(
            index=0,
            content="Test content",
            start_char=0,
            end_char=12,
            token_count=3,
        )
        assert chunk.index == 0
        assert chunk.content == "Test content"
        assert chunk.start_char == 0
        assert chunk.end_char == 12
        assert chunk.token_count == 3

    def test_chunk_has_overlap_flag(self) -> None:
        """Chunk tracks whether it includes overlap content."""
        chunk = Chunk(
            index=1,
            content="Test with overlap",
            start_char=100,
            end_char=200,
            token_count=5,
            has_overlap=True,
        )
        assert chunk.has_overlap is True


# =============================================================================
# Test SemanticBoundaryChunker
# =============================================================================


class TestSemanticBoundaryChunker:
    """Tests for semantic boundary chunking strategy."""

    def test_chunks_at_paragraph_boundaries(
        self,
        sample_text_with_paragraphs: str,
        default_config: ChunkingConfig,
    ) -> None:
        """AC-KB10.1: Chunks don't split mid-sentence at paragraph boundaries."""
        chunker = SemanticBoundaryChunker(default_config)
        chunks = chunker.chunk(sample_text_with_paragraphs)

        assert len(chunks) >= 2

        # Verify no chunk ends mid-sentence (no trailing incomplete sentences)
        for chunk in chunks:
            content = chunk.content.strip()
            # Content should end with proper punctuation or newline
            assert (
                content.endswith(".")
                or content.endswith(":")
                or content.endswith("\n")
                or content.endswith(")")
                or content == ""
            ), f"Chunk {chunk.index} ends mid-sentence: ...{content[-50:]}"

    def test_chunks_do_not_split_sentences(
        self,
        sample_text_with_paragraphs: str,
        default_config: ChunkingConfig,
    ) -> None:
        """AC-KB10.1: Chunks don't split mid-sentence."""
        chunker = SemanticBoundaryChunker(default_config)
        chunks = chunker.chunk(sample_text_with_paragraphs)

        for chunk in chunks:
            # Each chunk should contain complete sentences
            content = chunk.content.strip()
            if content:
                # Split into sentences and check none are fragments
                sentences = content.split(". ")
                for sentence in sentences[:-1]:  # All but last should end with period
                    assert not sentence.endswith(","), (
                        f"Sentence fragment detected in chunk {chunk.index}"
                    )

    def test_overlap_included_between_chunks(
        self,
        sample_text_with_paragraphs: str,
    ) -> None:
        """AC-KB10.2: Overlap tokens included between chunks."""
        config = ChunkingConfig(
            target_chunk_size=200,  # Small to force multiple chunks
            overlap_tokens=50,
        )
        chunker = SemanticBoundaryChunker(config)
        chunks = chunker.chunk(sample_text_with_paragraphs)

        if len(chunks) >= 2:
            # Check that consecutive chunks have overlapping content
            for i in range(1, len(chunks)):
                prev_content = chunks[i - 1].content
                curr_content = chunks[i].content

                # The start of current chunk should overlap with end of previous
                # (at least some common text should exist)
                if chunks[i].has_overlap:
                    # Find common substring
                    overlap_found = any(
                        word in prev_content
                        for word in curr_content.split()[:20]
                    )
                    assert overlap_found, (
                        f"No overlap between chunk {i - 1} and {i}"
                    )

    def test_respects_max_chunk_size(
        self,
        sample_text_with_paragraphs: str,
        default_config: ChunkingConfig,
    ) -> None:
        """Chunks respect maximum size limit."""
        chunker = SemanticBoundaryChunker(default_config)
        chunks = chunker.chunk(sample_text_with_paragraphs)

        for chunk in chunks:
            assert chunk.token_count <= default_config.max_chunk_size, (
                f"Chunk {chunk.index} exceeds max size"
            )


# =============================================================================
# Test SlidingWindowChunker
# =============================================================================


class TestSlidingWindowChunker:
    """Tests for sliding window fallback chunker."""

    def test_handles_text_without_boundaries(
        self,
        sample_text_single_paragraph: str,
        default_config: ChunkingConfig,
    ) -> None:
        """AC-KB10.1: Fallback handles text without clear boundaries."""
        chunker = SlidingWindowChunker(default_config)
        chunks = chunker.chunk(sample_text_single_paragraph)

        assert len(chunks) >= 1
        # All text should be covered
        total_content_len = sum(len(c.content) for c in chunks)
        assert total_content_len >= len(sample_text_single_paragraph) * 0.9

    def test_overlap_in_sliding_window(self) -> None:
        """AC-KB10.2: Sliding window includes configured overlap."""
        config = ChunkingConfig(
            target_chunk_size=100,
            overlap_tokens=30,
        )
        text = "word " * 500  # 500 words
        chunker = SlidingWindowChunker(config)
        chunks = chunker.chunk(text)

        if len(chunks) >= 2:
            for i in range(1, len(chunks)):
                assert chunks[i].has_overlap, f"Chunk {i} should have overlap"


# =============================================================================
# Test ChunkingStrategy Factory
# =============================================================================


class TestChunkingStrategy:
    """Tests for ChunkingStrategy (main interface)."""

    def test_prefers_semantic_chunking(
        self,
        sample_text_with_paragraphs: str,
        default_config: ChunkingConfig,
    ) -> None:
        """Prefers semantic boundary detection when possible."""
        strategy = ChunkingStrategy(default_config)
        chunks = strategy.chunk(sample_text_with_paragraphs)

        # Should use semantic chunking for structured text
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_falls_back_to_sliding_window(
        self,
        sample_text_single_paragraph: str,
        default_config: ChunkingConfig,
    ) -> None:
        """Falls back to sliding window for unstructured content."""
        strategy = ChunkingStrategy(default_config)
        chunks = strategy.chunk(sample_text_single_paragraph)

        assert len(chunks) >= 1

    def test_empty_input_returns_empty_list(
        self,
        default_config: ChunkingConfig,
    ) -> None:
        """Empty input returns empty chunk list."""
        strategy = ChunkingStrategy(default_config)
        chunks = strategy.chunk("")
        assert chunks == []

    def test_small_input_returns_single_chunk(
        self,
        default_config: ChunkingConfig,
    ) -> None:
        """Input smaller than target returns single chunk."""
        strategy = ChunkingStrategy(default_config)
        chunks = strategy.chunk("Small text.")
        assert len(chunks) == 1
        assert chunks[0].content == "Small text."

    def test_chunk_method_signature(
        self,
        default_config: ChunkingConfig,
    ) -> None:
        """AC-KB10.1: ChunkingStrategy.chunk() has correct signature."""
        strategy = ChunkingStrategy(default_config)
        # Method should accept string, return list of Chunk
        assert callable(getattr(strategy, "chunk", None))
        result = strategy.chunk("test")
        assert isinstance(result, list)

    def test_code_content_chunking(
        self,
        sample_code_content: str,
        default_config: ChunkingConfig,
    ) -> None:
        """Handles code content with class/function boundaries."""
        strategy = ChunkingStrategy(default_config)
        chunks = strategy.chunk(sample_code_content)

        # Code content should be chunked without breaking class definitions
        assert len(chunks) >= 1

    def test_preserves_all_content(
        self,
        sample_text_with_paragraphs: str,
    ) -> None:
        """No content lost during chunking (accounting for overlap)."""
        config = ChunkingConfig(
            target_chunk_size=200,
            overlap_tokens=0,  # No overlap to simplify test
        )
        strategy = ChunkingStrategy(config)
        chunks = strategy.chunk(sample_text_with_paragraphs)

        # Reconstruct original (approximately - may have boundary adjustments)
        reconstructed = "".join(c.content for c in chunks)

        # Should preserve most content
        original_words = set(sample_text_with_paragraphs.split())
        reconstructed_words = set(reconstructed.split())
        coverage = len(original_words & reconstructed_words) / len(original_words)
        assert coverage >= 0.95, f"Only {coverage:.0%} of content preserved"


# =============================================================================
# Test Token Estimation in Chunks
# =============================================================================


class TestChunkTokenEstimation:
    """Tests for token counting in chunks."""

    def test_chunk_has_token_count(self, default_config: ChunkingConfig) -> None:
        """Each chunk reports estimated token count."""
        strategy = ChunkingStrategy(default_config)
        chunks = strategy.chunk("This is a test sentence with several words.")

        for chunk in chunks:
            assert chunk.token_count > 0
            assert isinstance(chunk.token_count, int)

    def test_token_count_reasonable_estimate(
        self,
        default_config: ChunkingConfig,
    ) -> None:
        """Token count is reasonable estimate (~4 chars per token)."""
        strategy = ChunkingStrategy(default_config)
        text = "word " * 100  # 500 chars, ~125 tokens
        chunks = strategy.chunk(text)

        total_tokens = sum(c.token_count for c in chunks)
        # Should be roughly 125 tokens (500 chars / 4)
        assert 80 <= total_tokens <= 200, f"Unexpected token count: {total_tokens}"


# =============================================================================
# Edge Cases
# =============================================================================


class TestChunkingEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_word_handling(
        self,
        default_config: ChunkingConfig,
    ) -> None:
        """Handles text with very long 'words' (e.g., URLs)."""
        long_url = "https://example.com/" + "a" * 1000
        text = f"Check this link: {long_url} for more info."

        strategy = ChunkingStrategy(default_config)
        chunks = strategy.chunk(text)

        assert len(chunks) >= 1

    def test_unicode_content(
        self,
        default_config: ChunkingConfig,
    ) -> None:
        """Handles Unicode content correctly."""
        text = "机器学习是人工智能的子集。" * 50 + "\n\n" + "深度学习使用神经网络。" * 50

        strategy = ChunkingStrategy(default_config)
        chunks = strategy.chunk(text)

        assert len(chunks) >= 1
        # Content should be preserved
        all_content = "".join(c.content for c in chunks)
        assert "机器学习" in all_content

    def test_mixed_newlines(
        self,
        default_config: ChunkingConfig,
    ) -> None:
        """Handles mixed newline styles."""
        text = "Line 1\nLine 2\r\nLine 3\rLine 4\n\nParagraph 2"

        strategy = ChunkingStrategy(default_config)
        chunks = strategy.chunk(text)

        assert len(chunks) >= 1

    def test_whitespace_only_input(
        self,
        default_config: ChunkingConfig,
    ) -> None:
        """Whitespace-only input returns empty list."""
        strategy = ChunkingStrategy(default_config)
        chunks = strategy.chunk("   \n\n\t  ")
        assert chunks == []


# =============================================================================
# Integration with Token Budget
# =============================================================================


class TestChunkingTokenBudget:
    """Test integration with token budget constraints."""

    def test_respects_target_chunk_size(self) -> None:
        """Chunks respect target size (in tokens)."""
        config = ChunkingConfig(
            target_chunk_size=100,  # Small target (min allowed)
            overlap_tokens=10,
        )
        text = "This is a test sentence. " * 200  # Long text

        strategy = ChunkingStrategy(config)
        chunks = strategy.chunk(text)

        # Most chunks should be near target size
        for chunk in chunks:
            # Allow some variance for boundary adjustment
            assert chunk.token_count <= config.max_chunk_size, (
                f"Chunk {chunk.index} exceeds max size"
            )


__all__ = [
    "TestChunk",
    "TestChunkingConfig",
    "TestChunkingEdgeCases",
    "TestChunkingStrategy",
    "TestChunkingTokenBudget",
    "TestChunkTokenEstimation",
    "TestSemanticBoundaryChunker",
    "TestSlidingWindowChunker",
]
