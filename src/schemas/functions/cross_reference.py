"""Schemas for cross_reference function.

WBS-AGT13: cross_reference Function

This module defines the input/output schemas for the cross_reference
function, which finds related content across knowledge bases via
semantic search.

Acceptance Criteria:
- AC-13.1: Queries semantic-search-service for related content
- AC-13.2: Returns CrossReferenceResult with matches, relevance_scores
- AC-13.3: Context budget: 2048 input / 4096 output
- AC-13.4: Default preset: S4
- AC-13.5: Integrates with Qdrant via semantic-search-service

Exit Criteria:
- Each Match has source, content, relevance_score (0.0-1.0)
- FakeSemanticSearchClient used in unit tests
- Integration test hits real semantic-search-service:8081

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 8
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class MatchType(str, Enum):
    """Match type enum for cross_reference search.

    Per architecture:
    - semantic: Vector similarity search
    - keyword: Traditional keyword/BM25 search
    - hybrid: Combination of semantic and keyword
    """

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class CrossReferenceInput(BaseModel):
    """Input schema for cross_reference function.

    AC-13.1: Queries semantic-search-service for related content.

    Per architecture:
    - query_artifact: Source content to search for
    - search_scope: Which repositories to search
    - match_type: semantic | keyword | hybrid
    - top_k: Maximum results to return

    Attributes:
        query_artifact: Source content to find related content for
        search_scope: List of repository names to search (empty = all)
        match_type: Type of matching algorithm to use
        top_k: Maximum number of results to return
    """

    query_artifact: str = Field(
        ...,
        description="Source content to find related content for",
        min_length=1,
    )
    search_scope: list[str] = Field(
        default_factory=list,
        description="List of repository names to search (empty = all)",
    )
    match_type: MatchType = Field(
        default=MatchType.SEMANTIC,
        description="Type of matching algorithm to use",
    )
    top_k: int = Field(
        default=10,
        gt=0,
        le=1000,
        description="Maximum number of results to return",
    )

    @field_validator("query_artifact")
    @classmethod
    def validate_query_artifact_not_whitespace(cls, v: str) -> str:
        """Validate query_artifact is not just whitespace."""
        if not v.strip():
            raise ValueError("query_artifact cannot be empty or whitespace only")
        return v


class Reference(BaseModel):
    """A single reference match from cross_reference search.

    Exit Criteria: Each Match has source, content, relevance_score (0.0-1.0).

    Represents a matched item from semantic search with its source,
    content, and relevance score.

    Attributes:
        source: Source identifier (e.g., repo/file path)
        content: The matched content/chunk
        relevance_score: Similarity score between 0.0 and 1.0
        source_type: Optional type of source (code, book, doc)
        line_range: Optional line range for code sources
        commit_hash: Optional commit hash for code sources
    """

    source: str = Field(
        ...,
        description="Source identifier (e.g., repo/file path)",
    )
    content: str = Field(
        ...,
        description="The matched content/chunk",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score between 0.0 and 1.0",
    )
    source_type: str | None = Field(
        default=None,
        description="Type of source (code, book, doc)",
    )
    line_range: str | None = Field(
        default=None,
        description="Line range for code sources (e.g., '12-45')",
    )
    commit_hash: str | None = Field(
        default=None,
        description="Commit hash for code sources",
    )


class Citation(BaseModel):
    """A citation for footnote generation.

    Per architecture: citations: list[Citation] for footnotes.
    Used to generate Chicago-style footnotes from references.

    Attributes:
        marker: Footnote marker (e.g., '^1', '^2')
        formatted_citation: Chicago-style formatted citation
        reference_source: Optional back-reference to source
    """

    marker: str = Field(
        ...,
        description="Footnote marker (e.g., '^1', '^2')",
    )
    formatted_citation: str = Field(
        ...,
        description="Chicago-style formatted citation",
    )
    reference_source: str | None = Field(
        default=None,
        description="Optional back-reference to source identifier",
    )


class CrossReferenceResult(BaseModel):
    """Output schema for cross_reference function.

    AC-13.2: Returns CrossReferenceResult with matches, relevance_scores.

    Per architecture:
    - references: list[Reference]
    - similarity_scores: list
    - compressed_context: str (for downstream)
    - citations: list[Citation] (for footnotes)

    Attributes:
        references: List of matched references
        similarity_scores: List of similarity scores (parallel to references)
        compressed_context: Summarized context for downstream functions
        citations: List of formatted citations for footnotes
    """

    references: list[Reference] = Field(
        ...,
        description="List of matched references",
    )
    similarity_scores: list[float] = Field(
        ...,
        description="List of similarity scores (parallel to references)",
    )
    compressed_context: str = Field(
        ...,
        description="Summarized context for downstream functions",
    )
    citations: list[Citation] = Field(
        ...,
        description="List of formatted citations for footnotes",
    )

    @model_validator(mode="after")
    def validate_scores_match_references(self) -> "CrossReferenceResult":
        """Validate similarity_scores count matches references count."""
        if len(self.similarity_scores) != len(self.references):
            raise ValueError(
                f"similarity_scores count ({len(self.similarity_scores)}) "
                f"must match references count ({len(self.references)})"
            )
        return self

    @field_validator("similarity_scores")
    @classmethod
    def validate_scores_range(cls, v: list[float]) -> list[float]:
        """Validate each score is between 0.0 and 1.0."""
        for i, score in enumerate(v):
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"similarity_scores[{i}] = {score} must be between 0.0 and 1.0"
                )
        return v
