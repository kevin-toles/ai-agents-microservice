"""Analyze source node - Step 1 of workflow.

Reviews source chapter content and extracts key concepts for cross-referencing.

Pattern: LangGraph workflow node
Source: TIER_RELATIONSHIP_DIAGRAM.md Step 1
"""

from typing import Protocol

from src.agents.cross_reference.state import CrossReferenceState


class LLMClient(Protocol):
    """Protocol for LLM client dependency injection."""
    
    async def extract_concepts(self, text: str) -> list[str]:
        """Extract concepts from text using LLM."""
        ...


# Global client reference for dependency injection
_llm_client: LLMClient | None = None


def set_llm_client(client: LLMClient) -> None:
    """Set the LLM client for concept extraction.
    
    Args:
        client: LLM client implementing extract_concepts
    """
    global _llm_client
    _llm_client = client


def get_llm_client() -> LLMClient | None:
    """Get the current LLM client."""
    return _llm_client


async def analyze_source(state: CrossReferenceState) -> dict:
    """Analyze source chapter to extract key concepts.
    
    This is Step 1 of the 9-step workflow from TIER_RELATIONSHIP_DIAGRAM.md:
    "LLM Reviews Base Guideline + Enriched Metadata (Source Chapter)"
    
    Uses LLM to extract key concepts from the source chapter content.
    Falls back to empty list if content is empty or LLM unavailable.
    
    Args:
        state: Current workflow state with source chapter
        
    Returns:
        Dict with analyzed_concepts to merge into state
    """
    # Update current node
    result: dict = {"current_node": "analyze_source"}
    
    # Check for empty content
    content = state.source.content
    if not content or not content.strip():
        result["analyzed_concepts"] = []
        return result
    
    # Use LLM client if available
    client = get_llm_client()
    if client is not None:
        try:
            concepts = await client.extract_concepts(content)
            result["analyzed_concepts"] = concepts
            return result
        except Exception:
            # Fall back to empty on error
            result["analyzed_concepts"] = []
            result["errors"] = state.errors + ["LLM concept extraction failed"]
            return result
    
    # No LLM client - fall back to source keywords
    concepts = list(state.source.keywords)
    if state.source.concepts:
        concepts.extend(state.source.concepts)
    
    # Deduplicate
    result["analyzed_concepts"] = list(set(concepts))
    return result
