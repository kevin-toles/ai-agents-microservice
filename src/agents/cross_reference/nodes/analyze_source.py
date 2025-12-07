"""Analyze source node - Step 1 of workflow.

Reviews source chapter content and extracts key concepts for cross-referencing.

Pattern: LangGraph workflow node
Source: TIER_RELATIONSHIP_DIAGRAM.md Step 1
"""

from src.agents.cross_reference.state import CrossReferenceState


async def analyze_source(state: CrossReferenceState) -> CrossReferenceState:
    """Analyze source chapter to extract key concepts.
    
    This is Step 1 of the 9-step workflow from TIER_RELATIONSHIP_DIAGRAM.md:
    "LLM Reviews Base Guideline + Enriched Metadata (Source Chapter)"
    
    The node:
    - Extracts keywords from source chapter
    - Identifies key concepts
    - Prepares search queries for taxonomy traversal
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with analyzed_concepts populated
    """
    state.current_node = "analyze_source"
    
    # TODO: Implement LLM-based analysis
    # For now, use source keywords as concepts
    concepts = list(state.source.keywords)
    if state.source.concepts:
        concepts.extend(state.source.concepts)
    
    # Deduplicate
    state.analyzed_concepts = list(set(concepts))
    
    return state
