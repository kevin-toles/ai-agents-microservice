"""Synthesize node - Steps 7-9 of workflow.

Validates matches and synthesizes scholarly annotation with citations.

Pattern: LangGraph workflow node
Source: TIER_RELATIONSHIP_DIAGRAM.md Steps 7-9
"""

from datetime import UTC, datetime
from typing import Protocol

from src.agents.cross_reference.state import (
    ChapterMatch,
    Citation,
    CrossReferenceResult,
    CrossReferenceState,
    TierCoverage,
)


class SynthesisLLMClient(Protocol):
    """Protocol for synthesis LLM client dependency injection."""
    
    async def generate_annotation(
        self,
        source_content: str,
        matches: list[dict],
    ) -> str:
        """Generate scholarly annotation from source and matches.
        
        Args:
            source_content: Source chapter content
            matches: List of matched chapter dicts with content
            
        Returns:
            Scholarly annotation text with inline citation markers
        """
        ...


# Global client reference for dependency injection
_synthesis_client: SynthesisLLMClient | None = None
_model_name: str = "gpt-4"


def set_synthesis_client(client: SynthesisLLMClient | None, model_name: str = "gpt-4") -> None:
    """Set the synthesis LLM client.
    
    Args:
        client: LLM client implementing generate_annotation, or None to reset
        model_name: Name of the LLM model being used
    """
    global _synthesis_client, _model_name
    _synthesis_client = client
    _model_name = model_name


def get_synthesis_client() -> SynthesisLLMClient | None:
    """Get the current synthesis client."""
    return _synthesis_client


async def synthesize(state: CrossReferenceState) -> dict:
    """Synthesize annotation from retrieved content.
    
    This covers Steps 7-9 of the workflow:
    - Step 7: Validate & Synthesize (Genuine Relevance Check)
    - Step 8: Structure Annotation by Tier Priority
    - Step 9: Output Scholarly Annotation with Citations
    
    Produces Chicago-style citations organized by tier priority:
    1. Tier 1 (Architecture Spine) - REQUIRED
    2. Tier 2 (Implementation) - REQUIRED  
    3. Tier 3 (Engineering Practices) - OPTIONAL
    
    Args:
        state: Current workflow state
        
    Returns:
        Dict with result to merge into state
    """
    result: dict = {"current_node": "synthesize"}
    
    # Get validated matches
    matches = state.validated_matches
    if not matches:
        # Return empty result
        empty_result = CrossReferenceResult(
            annotation="No relevant cross-references found.",
            citations=[],
            traversal_paths=state.traversal_paths,
            tier_coverage=_calculate_tier_coverage([]),
            matches=[],
            processing_time_ms=_calculate_processing_time(state.started_at),
            model_used=_model_name,
        )
        result["result"] = empty_result
        return result
    
    # Sort matches by tier priority (1, 2, 3)
    sorted_matches = sorted(matches, key=lambda m: m.tier)
    
    # Create citations in Chicago format
    citations = _create_citations(sorted_matches)
    
    # Calculate tier coverage
    tier_coverage = _calculate_tier_coverage(sorted_matches)
    
    # Get synthesis client
    client = get_synthesis_client()
    
    if client is not None:
        try:
            # Generate annotation using LLM
            match_dicts = [
                {
                    "book": m.book,
                    "chapter": m.chapter,
                    "title": m.title,
                    "tier": m.tier,
                    "content": m.content or "",
                }
                for m in sorted_matches
            ]
            annotation = await client.generate_annotation(
                source_content=state.source.content,
                matches=match_dicts,
            )
        except Exception:
            # Fall back to default annotation
            annotation = _generate_default_annotation(sorted_matches, citations)
    else:
        # No LLM client - generate default annotation
        annotation = _generate_default_annotation(sorted_matches, citations)
    
    # Create result
    cross_ref_result = CrossReferenceResult(
        annotation=annotation,
        citations=citations,
        traversal_paths=state.traversal_paths,
        tier_coverage=tier_coverage,
        matches=sorted_matches,
        processing_time_ms=_calculate_processing_time(state.started_at),
        model_used=_model_name,
    )
    
    result["result"] = cross_ref_result
    return result


def _create_citations(matches: list[ChapterMatch]) -> list[Citation]:
    """Create Chicago-style citations from matches.
    
    Args:
        matches: List of chapter matches sorted by tier
        
    Returns:
        List of Citation objects
    """
    citations = []
    for match in matches:
        citation = Citation(
            book=match.book,
            chapter=match.chapter,
            chapter_title=match.title,
            pages=match.page_range,
            tier=match.tier,
        )
        citations.append(citation)
    return citations


def _calculate_tier_coverage(matches: list[ChapterMatch]) -> list[TierCoverage]:
    """Calculate tier coverage statistics.
    
    Args:
        matches: List of chapter matches
        
    Returns:
        List of TierCoverage for tiers 1-3
    """
    tier_names = {
        1: "Architecture Spine",
        2: "Implementation",
        3: "Engineering Practices",
    }
    
    # Count by tier
    tier_books: dict[int, set[str]] = {1: set(), 2: set(), 3: set()}
    tier_chapters: dict[int, int] = {1: 0, 2: 0, 3: 0}
    
    for match in matches:
        tier = match.tier
        if tier in tier_books:
            tier_books[tier].add(match.book)
            tier_chapters[tier] += 1
    
    coverage = []
    for tier in [1, 2, 3]:
        tc = TierCoverage(
            tier=tier,
            tier_name=tier_names[tier],
            books_referenced=len(tier_books[tier]),
            chapters_referenced=tier_chapters[tier],
            has_coverage=tier_chapters[tier] > 0,
        )
        coverage.append(tc)
    
    return coverage


def _calculate_processing_time(started_at: datetime) -> float:
    """Calculate processing time in milliseconds.
    
    Args:
        started_at: Workflow start time
        
    Returns:
        Processing time in milliseconds
    """
    delta = datetime.now(UTC) - started_at
    return delta.total_seconds() * 1000


def _generate_default_annotation(
    matches: list[ChapterMatch],
    citations: list[Citation],
) -> str:
    """Generate a default annotation without LLM.
    
    Args:
        matches: Sorted chapter matches
        citations: Generated citations
        
    Returns:
        Default annotation text
    """
    if not matches:
        return "No relevant cross-references found."
    
    # Group by tier
    tier_matches: dict[int, list[ChapterMatch]] = {1: [], 2: [], 3: []}
    for match in matches:
        if match.tier in tier_matches:
            tier_matches[match.tier].append(match)
    
    parts = []
    footnote_num = 1
    
    # Tier 1 - Architecture
    if tier_matches[1]:
        tier1_refs = []
        for m in tier_matches[1]:
            tier1_refs.append(f"{m.book} Ch. {m.chapter}[^{footnote_num}]")
            footnote_num += 1
        parts.append(f"**Architecture Context:** See {', '.join(tier1_refs)}.")
    
    # Tier 2 - Implementation
    if tier_matches[2]:
        tier2_refs = []
        for m in tier_matches[2]:
            tier2_refs.append(f"{m.book} Ch. {m.chapter}[^{footnote_num}]")
            footnote_num += 1
        parts.append(f"**Implementation Details:** Refer to {', '.join(tier2_refs)}.")
    
    # Tier 3 - Practices
    if tier_matches[3]:
        tier3_refs = []
        for m in tier_matches[3]:
            tier3_refs.append(f"{m.book} Ch. {m.chapter}[^{footnote_num}]")
            footnote_num += 1
        parts.append(f"**Engineering Practices:** Also see {', '.join(tier3_refs)}.")
    
    annotation = "\n\n".join(parts)
    
    # Add footnotes
    annotation += "\n\n"
    for i, citation in enumerate(citations, 1):
        annotation += citation.to_chicago_format(i) + "\n"
    
    return annotation.strip()
