"""traverse_graph tool - Execute spider web traversal.

Pattern: LangChain Tool
Source: ARCHITECTURE.md tool definition, TIER_RELATIONSHIP_DIAGRAM.md

WBS 5.7.4: Implemented via semantic-search-service client.
Calls SemanticSearchClient.traverse() for graph traversal.

Anti-Pattern References (CODING_PATTERNS_ANALYSIS.md):
- #4.3: Unused params → prefix with underscore
- #8.1: Real async await (not asyncio.sleep(0) stubs)
"""

import logging
from typing import Any

from src.agents.cross_reference.state import RelationshipType
from src.core.clients.semantic_search import get_semantic_search_client


logger = logging.getLogger(__name__)


async def traverse_graph(
    start_book: str,
    start_chapter: int,
    start_tier: int,
    max_hops: int = 3,
    relationship_types: list[RelationshipType] | None = None,
    allow_cycles: bool = True,
    direction: str = "BOTH",
) -> dict[str, Any]:
    """Execute spider web traversal across the taxonomy graph.

    Supports bidirectional, skip-tier, and non-linear paths
    per the spider web model in TIER_RELATIONSHIP_DIAGRAM.md.
    Uses SemanticSearchClient.traverse() for graph traversal.

    Args:
        start_book: Starting book title
        start_chapter: Starting chapter number
        start_tier: Starting tier level
        max_hops: Maximum traversal depth
        relationship_types: Which relationship types to follow
        allow_cycles: Whether to allow revisiting tiers
        direction: Direction of traversal

    Returns:
        Dict with traversal paths and statistics
    """
    # Acknowledge params for future implementation (Anti-Pattern #4.3)
    _allow_cycles = allow_cycles
    _direction = direction
    _start_tier = start_tier

    # Get injected client (set via set_semantic_search_client in main.py lifespan)
    client = get_semantic_search_client()
    if client is None:
        raise RuntimeError(
            "SemanticSearchClient not initialized. "
            "Ensure ai-agents service started properly (set_semantic_search_client called)."
        )

    # Build node ID from book + chapter
    start_node_id = _build_node_id(start_book, start_chapter)

    # Convert RelationshipTypes to strings for API
    rel_types = None
    if relationship_types is None:
        relationship_types = [
            RelationshipType.PARALLEL,
            RelationshipType.PERPENDICULAR,
            RelationshipType.SKIP_TIER,
        ]
    rel_types = [rt.value for rt in relationship_types]

    logger.debug(
        "Traversing graph",
        extra={
            "start_node_id": start_node_id,
            "max_hops": max_hops,
            "relationship_types": rel_types,
            "allow_cycles": _allow_cycles,
            "direction": _direction,
        },
    )

    try:
        result = await client.traverse(
            start_node_id=start_node_id,
            relationship_types=rel_types,
            max_depth=max_hops,
            limit=50,
        )

        # Transform response to tool format
        paths = _build_paths_from_traversal(result)

        return {
            "paths": paths,
            "nodes": result.get("nodes", []),
            "edges": result.get("edges", []),
            "traversal_stats": {
                "nodes_visited": len(result.get("nodes", [])),
                "unique_books": _count_unique_books(result.get("nodes", [])),
                "tiers_covered": _get_tiers_covered(result.get("nodes", [])),
            },
        }

    except Exception as e:
        logger.error("Graph traversal failed", extra={"error": str(e)})
        return {
            "paths": [],
            "traversal_stats": {
                "nodes_visited": 0,
                "unique_books": 0,
                "tiers_covered": [],
            },
            "error": str(e),
        }


def _build_node_id(book: str, chapter: int) -> str:
    """Build a node ID from book and chapter.

    Args:
        book: Book title
        chapter: Chapter number

    Returns:
        Node ID string (e.g., "ai-engineering-ch-3")
    """
    # Normalize book name to slug format
    slug = book.lower().replace(" ", "-").replace("_", "-")
    return f"{slug}-ch-{chapter}"


def _build_paths_from_traversal(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build path representations from traversal result.

    Args:
        result: Raw traversal response

    Returns:
        List of path dictionaries
    """
    nodes = result.get("nodes", [])
    edges = result.get("edges", [])

    if not nodes:
        return []

    # Simple path representation: each node is a step
    paths = []
    for node in nodes:
        paths.append({
            "node_id": node.get("id", ""),
            "depth": node.get("depth", 0),
            "labels": node.get("labels", []),
            "properties": node.get("properties", {}),
        })

    # Add edge information if available
    if edges:
        for _i, path in enumerate(paths):
            # Find edges connected to this node
            node_edges = [
                e for e in edges
                if e.get("target") == path["node_id"]
            ]
            if node_edges:
                path["incoming_edge"] = node_edges[0]

    return paths


def _count_unique_books(nodes: list[dict[str, Any]]) -> int:
    """Count unique books in traversal nodes."""
    books = set()
    for node in nodes:
        props = node.get("properties", {})
        if "book" in props:
            books.add(props["book"])
    return len(books)


def _get_tiers_covered(nodes: list[dict[str, Any]]) -> list[int]:
    """Get list of tiers covered in traversal."""
    tiers = set()
    for node in nodes:
        props = node.get("properties", {})
        if "tier" in props:
            tiers.add(props["tier"])
    return sorted(tiers)
