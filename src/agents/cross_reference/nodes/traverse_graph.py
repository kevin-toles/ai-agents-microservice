"""Traverse graph node - Step 5 of workflow.

Executes spider web traversal across the taxonomy graph.

Pattern: LangGraph workflow node
Source: TIER_RELATIONSHIP_DIAGRAM.md Step 5, Spider Web Model
"""

from typing import Protocol

from src.agents.cross_reference.state import (
    ChapterMatch,
    CrossReferenceState,
    GraphNode,
    RelationshipType,
    TraversalPath,
)


class GraphClient(Protocol):
    """Protocol for graph client dependency injection."""
    
    async def get_neighbors(
        self,
        book: str,
        chapter: int,
        relationship_types: list[str] | None = None,
    ) -> list[dict]:
        """Get neighboring nodes for a chapter.
        
        Args:
            book: Book title
            chapter: Chapter number
            relationship_types: Optional filter for relationship types
            
        Returns:
            List of neighbor dicts with book, chapter, title, tier, 
            relationship_type, similarity
        """
        ...


# Global client reference for dependency injection
_graph_client: GraphClient | None = None


def set_graph_client(client: GraphClient) -> None:
    """Set the graph client for traversal.
    
    Args:
        client: Graph client implementing get_neighbors
    """
    global _graph_client
    _graph_client = client


def get_graph_client() -> GraphClient | None:
    """Get the current graph client."""
    return _graph_client


async def traverse_graph(state: CrossReferenceState) -> dict:
    """Execute spider web traversal.
    
    This is Step 5 of the workflow:
    "LLM Requests Specific Chapter Content (NO LIMITS)"
    
    Traverses the graph following:
    - PARALLEL: Same tier relationships
    - PERPENDICULAR: Adjacent tier relationships
    - SKIP_TIER: Non-adjacent tier relationships
    
    All relationships are BIDIRECTIONAL per the spider web model.
    Respects max_hops configuration and detects cycles.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dict with traversal_paths to merge into state
    """
    result: dict = {"current_node": "traverse_graph"}
    
    # Check for matches to traverse from
    matches = state.taxonomy_matches
    if not matches:
        result["traversal_paths"] = []
        return result
    
    # Get graph client
    client = get_graph_client()
    if client is None:
        result["traversal_paths"] = []
        result["errors"] = state.errors + ["Graph client not configured"]
        return result
    
    max_hops = state.config.max_hops
    traversal_paths: list[TraversalPath] = []
    
    try:
        for match in matches:
            # Start a path from each matched chapter
            path = await _traverse_from_node(
                client=client,
                start_match=match,
                max_hops=max_hops,
                visited=set(),
            )
            if path:
                traversal_paths.append(path)
        
        result["traversal_paths"] = traversal_paths
        return result
        
    except Exception as e:
        result["traversal_paths"] = []
        result["errors"] = state.errors + [f"Graph traversal failed: {e}"]
        return result


async def _traverse_from_node(
    client: GraphClient,
    start_match: ChapterMatch,
    max_hops: int,
    visited: set[tuple[str, int]],
) -> TraversalPath | None:
    """Traverse from a starting node up to max_hops.
    
    Args:
        client: Graph client
        start_match: Starting chapter match
        max_hops: Maximum traversal depth
        visited: Set of visited (book, chapter) tuples for cycle detection
        
    Returns:
        TraversalPath if traversal successful, None otherwise
    """
    if max_hops <= 0:
        return None
    
    # Create starting node
    start_key = (start_match.book, start_match.chapter)
    if start_key in visited:
        return None
    
    visited.add(start_key)
    
    start_node = GraphNode(
        book=start_match.book,
        chapter=start_match.chapter,
        title=start_match.title,
        tier=start_match.tier,
    )
    
    nodes = [start_node]
    relationships: list[RelationshipType] = []
    total_similarity = start_match.similarity
    
    # BFS traversal
    current_match = start_match
    hops_remaining = max_hops
    
    while hops_remaining > 0:
        neighbors = await client.get_neighbors(
            book=current_match.book,
            chapter=current_match.chapter,
        )
        
        if not neighbors:
            break
        
        # Find best unvisited neighbor
        best_neighbor = _find_best_neighbor(neighbors, visited)
        
        if best_neighbor is None:
            break
        
        # Process neighbor
        neighbor_node, rel_type, neighbor_match = _process_neighbor(
            best_neighbor, visited
        )
        nodes.append(neighbor_node)
        relationships.append(rel_type)
        
        total_similarity += best_neighbor.get("similarity", 0.0)
        current_match = neighbor_match
        hops_remaining -= 1
    
    # Determine path type
    path_type = "non_linear" if len(set(relationships)) > 1 else "linear"
    
    return TraversalPath(
        nodes=nodes,
        relationships=relationships,
        total_similarity=total_similarity,
        path_type=path_type,
    )


def _find_best_neighbor(
    neighbors: list[dict],
    visited: set[tuple[str, int]],
) -> dict | None:
    """Find the best unvisited neighbor by similarity.
    
    Args:
        neighbors: List of neighbor dicts
        visited: Set of visited (book, chapter) tuples
        
    Returns:
        Best neighbor dict or None if all visited
    """
    best_neighbor = None
    best_similarity = -1.0
    
    for neighbor in neighbors:
        neighbor_key = (neighbor.get("book", ""), neighbor.get("chapter", 0))
        if neighbor_key not in visited:
            sim = neighbor.get("similarity", 0.0)
            if sim > best_similarity:
                best_similarity = sim
                best_neighbor = neighbor
    
    return best_neighbor


def _process_neighbor(
    neighbor: dict,
    visited: set[tuple[str, int]],
) -> tuple[GraphNode, RelationshipType, ChapterMatch]:
    """Process a neighbor into node, relationship, and match.
    
    Args:
        neighbor: Neighbor dict from graph
        visited: Set of visited (book, chapter) tuples (will be updated)
        
    Returns:
        Tuple of (GraphNode, RelationshipType, ChapterMatch)
    """
    # Mark as visited
    neighbor_key = (neighbor["book"], neighbor["chapter"])
    visited.add(neighbor_key)
    
    # Create node
    neighbor_node = GraphNode(
        book=neighbor.get("book", ""),
        chapter=neighbor.get("chapter", 0),
        title=neighbor.get("title", ""),
        tier=neighbor.get("tier", 1),
    )
    
    # Parse relationship type
    rel_type_str = neighbor.get("relationship_type", "PARALLEL")
    try:
        rel_type = RelationshipType(rel_type_str)
    except ValueError:
        rel_type = RelationshipType.PARALLEL
    
    # Create match for next iteration
    neighbor_match = ChapterMatch(
        book=neighbor.get("book", ""),
        chapter=neighbor.get("chapter", 0),
        title=neighbor.get("title", ""),
        tier=neighbor.get("tier", 1),
        similarity=neighbor.get("similarity", 0.0),
    )
    
    return neighbor_node, rel_type, neighbor_match
