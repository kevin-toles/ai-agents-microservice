"""Chunking Strategy for Map-Reduce Summarization - WBS-KB10.

Implements semantic boundary detection and sliding window fallback
for splitting long content into manageable chunks.

Acceptance Criteria:
- AC-KB10.1: ChunkingStrategy.chunk() splits at semantic boundaries
- AC-KB10.2: Chunk overlap configurable (default 200 tokens)

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Map-Reduce Pattern
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from src.functions.utils.token_utils import estimate_tokens, tokens_to_chars


if TYPE_CHECKING:
    pass


# =============================================================================
# Configuration Schema
# =============================================================================


class ChunkingConfig(BaseModel):
    """Configuration for chunking strategy.

    Attributes:
        target_chunk_size: Target size for each chunk in tokens
        overlap_tokens: Number of tokens to overlap between chunks
        min_chunk_size: Minimum chunk size in tokens (avoid tiny chunks)
        max_chunk_size: Maximum chunk size in tokens (hard limit)

    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Token Budget Allocation
    """

    model_config = ConfigDict(frozen=True)

    target_chunk_size: int = Field(
        default=2000,
        ge=100,
        le=8000,
        description="Target size for each chunk in tokens",
    )
    overlap_tokens: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Number of tokens to overlap between chunks (default 200)",
    )
    min_chunk_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Minimum chunk size in tokens",
    )
    max_chunk_size: int = Field(
        default=4000,
        ge=500,
        le=16000,
        description="Maximum chunk size in tokens (hard limit)",
    )


# =============================================================================
# Chunk Dataclass
# =============================================================================


@dataclass
class Chunk:
    """A chunk of content with metadata.

    Attributes:
        index: Zero-based index of this chunk
        content: The text content of the chunk
        start_char: Starting character position in original text
        end_char: Ending character position in original text
        token_count: Estimated token count for this chunk
        has_overlap: Whether this chunk includes overlap from previous
        metadata: Additional chunk metadata
    """

    index: int
    content: str
    start_char: int
    end_char: int
    token_count: int
    has_overlap: bool = False
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Base Chunker Protocol
# =============================================================================


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize chunker with configuration."""
        self.config = config

    @abstractmethod
    def chunk(self, content: str) -> list[Chunk]:
        """Split content into chunks.

        Args:
            content: Text content to chunk

        Returns:
            List of Chunk objects
        """
        ...

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return estimate_tokens(text)

    def _create_chunk(
        self,
        index: int,
        content: str,
        start_char: int,
        end_char: int,
        has_overlap: bool = False,
    ) -> Chunk:
        """Create a Chunk with token estimation."""
        return Chunk(
            index=index,
            content=content,
            start_char=start_char,
            end_char=end_char,
            token_count=self._estimate_tokens(content),
            has_overlap=has_overlap,
        )


# =============================================================================
# Semantic Boundary Chunker
# =============================================================================


class SemanticBoundaryChunker(BaseChunker):
    """Chunks content at semantic boundaries (paragraphs, sections).

    Prefers breaking at:
    1. Section headers (markdown headers, numbered sections)
    2. Paragraph breaks (double newlines)
    3. Sentence boundaries (periods followed by space/newline)

    Falls back to word boundaries if no semantic boundary found.
    """

    # Regex patterns for semantic boundaries
    SECTION_PATTERN = re.compile(
        r"(?:^|\n)(?:#{1,6}\s|(?:\d+\.)+\s|[A-Z][^.]*:(?:\n|$))",
        re.MULTILINE,
    )
    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    CODE_BLOCK_PATTERN = re.compile(r"(?:^|\n)(?:class |def |async def )", re.MULTILINE)

    def chunk(self, content: str) -> list[Chunk]:
        """Split content at semantic boundaries.

        AC-KB10.1: Chunks don't split mid-sentence at boundaries.
        AC-KB10.2: Includes configured overlap between chunks.
        """
        if not content or not content.strip():
            return []

        chunks: list[Chunk] = []
        target_chars = tokens_to_chars(self.config.target_chunk_size)
        overlap_chars = tokens_to_chars(self.config.overlap_tokens)
        min_chars = tokens_to_chars(self.config.min_chunk_size)
        max_chars = tokens_to_chars(self.config.max_chunk_size)

        # Find all potential boundaries
        boundaries = self._find_boundaries(content)

        # Build chunks respecting boundaries
        current_start = 0
        chunk_index = 0

        while current_start < len(content):
            # Calculate target end position
            target_end = min(current_start + target_chars, len(content))

            # Find best boundary near target
            chunk_end = self._find_best_boundary(
                content,
                boundaries,
                current_start,
                target_end,
                max_chars,
            )

            # Extract chunk content
            chunk_content = content[current_start:chunk_end]

            # Skip if chunk is too small and not the last chunk
            if len(chunk_content.strip()) < min_chars and chunk_end < len(content):
                # Extend to next boundary
                chunk_end = self._find_next_boundary(
                    boundaries,
                    chunk_end,
                    len(content),
                )
                chunk_content = content[current_start:chunk_end]

            if chunk_content.strip():
                chunks.append(
                    self._create_chunk(
                        index=chunk_index,
                        content=chunk_content,
                        start_char=current_start,
                        end_char=chunk_end,
                        has_overlap=chunk_index > 0 and overlap_chars > 0,
                    )
                )
                chunk_index += 1

            # Move to next chunk start (with overlap)
            next_start = chunk_end - overlap_chars
            if next_start <= current_start:
                next_start = chunk_end  # Prevent infinite loop
            current_start = max(next_start, current_start + min_chars)

            # Handle end of content
            if current_start >= len(content):
                break

        return chunks

    def _find_boundaries(self, content: str) -> list[int]:
        """Find all potential semantic boundaries in content."""
        boundaries: set[int] = {0, len(content)}

        # Section headers
        for match in self.SECTION_PATTERN.finditer(content):
            boundaries.add(match.start())

        # Paragraph breaks
        for match in self.PARAGRAPH_PATTERN.finditer(content):
            boundaries.add(match.end())

        # Sentence endings
        for match in self.SENTENCE_PATTERN.finditer(content):
            boundaries.add(match.start())

        # Code block boundaries
        for match in self.CODE_BLOCK_PATTERN.finditer(content):
            boundaries.add(match.start())

        return sorted(boundaries)

    def _find_best_boundary(
        self,
        content: str,
        boundaries: list[int],
        start: int,
        target_end: int,
        max_end: int,
    ) -> int:
        """Find best boundary position near target end."""
        absolute_max = min(start + max_end, len(content))

        # Find boundaries within acceptable range
        valid_boundaries = [
            b for b in boundaries
            if start < b <= absolute_max
        ]

        if not valid_boundaries:
            # No boundary found, use word boundary
            return self._find_word_boundary(content, target_end, absolute_max)

        # Find boundary closest to target
        target_boundary = min(
            valid_boundaries,
            key=lambda b: abs(b - target_end),
        )

        return target_boundary

    def _find_word_boundary(
        self,
        content: str,
        target: int,
        max_pos: int,
    ) -> int:
        """Find nearest word boundary near target position."""
        # Search forward for space
        for i in range(target, min(target + 100, max_pos)):
            if i < len(content) and content[i] in " \n\t":
                return i

        # Search backward if no forward boundary found
        for i in range(target, max(target - 100, 0), -1):
            if content[i] in " \n\t":
                return i + 1

        return min(target, len(content))

    def _find_next_boundary(
        self,
        boundaries: list[int],
        current: int,
        max_pos: int,
    ) -> int:
        """Find next boundary after current position."""
        for b in boundaries:
            if b > current:
                return min(b, max_pos)
        return max_pos


# =============================================================================
# Sliding Window Chunker
# =============================================================================


class SlidingWindowChunker(BaseChunker):
    """Fallback chunker using sliding window with overlap.

    Used when content lacks clear semantic boundaries.
    Splits at word boundaries within target window.
    """

    def chunk(self, content: str) -> list[Chunk]:
        """Split content using sliding window approach.

        AC-KB10.1: Fallback for content without clear boundaries.
        AC-KB10.2: Includes configured overlap between chunks.
        """
        if not content or not content.strip():
            return []

        chunks: list[Chunk] = []
        target_chars = tokens_to_chars(self.config.target_chunk_size)
        overlap_chars = tokens_to_chars(self.config.overlap_tokens)
        step_size = max(target_chars - overlap_chars, 100)

        current_pos = 0
        chunk_index = 0

        while current_pos < len(content):
            # Calculate chunk end
            target_end = min(current_pos + target_chars, len(content))

            # Find word boundary near target end
            chunk_end = self._find_word_boundary(content, target_end)

            # Extract chunk
            chunk_content = content[current_pos:chunk_end]

            if chunk_content.strip():
                chunks.append(
                    self._create_chunk(
                        index=chunk_index,
                        content=chunk_content,
                        start_char=current_pos,
                        end_char=chunk_end,
                        has_overlap=chunk_index > 0,
                    )
                )
                chunk_index += 1

            # Move window
            current_pos += step_size

            if chunk_end >= len(content):
                break

        return chunks

    def _find_word_boundary(self, content: str, target: int) -> int:
        """Find nearest word boundary near target position."""
        if target >= len(content):
            return len(content)

        # Search forward for space
        for i in range(target, min(target + 50, len(content))):
            if content[i] in " \n\t":
                return i

        # Search backward if no forward boundary found
        for i in range(target, max(target - 50, 0), -1):
            if content[i] in " \n\t":
                return i + 1

        return target


# =============================================================================
# Main ChunkingStrategy
# =============================================================================


class ChunkingStrategy:
    """Main chunking interface with automatic strategy selection.

    Prefers semantic boundary detection but falls back to sliding
    window for unstructured content.

    Example:
        >>> config = ChunkingConfig(target_chunk_size=1000, overlap_tokens=200)
        >>> strategy = ChunkingStrategy(config)
        >>> chunks = strategy.chunk(long_document)

    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Map-Reduce Pattern
    """

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize chunking strategy.

        Args:
            config: Chunking configuration
        """
        self.config = config
        self._semantic_chunker = SemanticBoundaryChunker(config)
        self._window_chunker = SlidingWindowChunker(config)

    def chunk(self, content: str) -> list[Chunk]:
        """Split content into chunks.

        Automatically selects best strategy based on content structure.

        AC-KB10.1: Splits at semantic boundaries (paragraphs, sections)
        AC-KB10.2: Includes configurable overlap (default 200 tokens)

        Args:
            content: Text content to chunk

        Returns:
            List of Chunk objects with metadata
        """
        if not content or not content.strip():
            return []

        # Try semantic chunking first
        if self._has_semantic_boundaries(content):
            return self._semantic_chunker.chunk(content)

        # Fall back to sliding window
        return self._window_chunker.chunk(content)

    def _has_semantic_boundaries(self, content: str) -> bool:
        """Detect if content has semantic boundaries.

        Returns True if content has:
        - Paragraph breaks (double newlines)
        - Section headers
        - Multiple sentences
        """
        # Check for paragraph breaks
        if "\n\n" in content:
            return True

        # Check for section patterns
        if re.search(r"^#{1,6}\s", content, re.MULTILINE):
            return True

        # Check for numbered sections
        if re.search(r"^\d+\.", content, re.MULTILINE):
            return True

        # Check for code structure
        if re.search(r"^(?:class |def |async def )", content, re.MULTILINE):
            return True

        # Check for multiple sentences
        sentence_count = len(re.findall(r"[.!?]\s+[A-Z]", content))
        return sentence_count >= 3


__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkingConfig",
    "ChunkingStrategy",
    "SemanticBoundaryChunker",
    "SlidingWindowChunker",
]
