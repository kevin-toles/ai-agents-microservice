"""Search taxonomy node - Step 2-4 of workflow.

Searches the taxonomy for books/chapters related to source concepts.

Pattern: LangGraph workflow node
Source: TIER_RELATIONSHIP_DIAGRAM.md Steps 2-4
"""

from src.agents.cross_reference.state import CrossReferenceState


async def search_taxonomy(state: CrossReferenceState) -> CrossReferenceState:
    """Search taxonomy for related content.
    
    This covers Steps 2-4 of the 9-step workflow:
    - Step 2: Review Taxonomy Structure
    - Step 3: Review Companion Book Metadata
    - Step 4: Cross-Reference Keywords & Match Concepts
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with taxonomy_matches populated
    """
    state.current_node = "search_taxonomy"
    
    # TODO: Implement taxonomy search via semantic-search-service
    # Uses tools: search_taxonomy(), search_similar()
    
    return state
