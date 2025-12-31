"""Workflow nodes for Cross-Reference Agent.

Each node represents a step in the LangGraph workflow:
1. analyze_source - Understand source chapter
2. search_taxonomy - Find related books in taxonomy
3. traverse_graph - Follow spider web paths
4. retrieve_content - Get relevant chapter texts
5. synthesize - Generate annotation with citations

Pattern: LangGraph workflow nodes
Source: ARCHITECTURE.md, TIER_RELATIONSHIP_DIAGRAM.md
"""

from src.agents.cross_reference.nodes.analyze_source import (
    analyze_source,
    get_llm_client,
    set_llm_client,
)
from src.agents.cross_reference.nodes.retrieve_content import (
    get_content_client,
    retrieve_content,
    set_content_client,
)
from src.agents.cross_reference.nodes.search_taxonomy import (
    get_neo4j_client,
    search_taxonomy,
    set_neo4j_client,
)
from src.agents.cross_reference.nodes.synthesize import (
    get_synthesis_client,
    set_synthesis_client,
    synthesize,
)
from src.agents.cross_reference.nodes.traverse_graph import (
    get_graph_client,
    set_graph_client,
    traverse_graph,
)


__all__ = [
    # Workflow nodes
    "analyze_source",
    # Dependency injection - Content
    "get_content_client",
    # Dependency injection - Graph
    "get_graph_client",
    # Dependency injection - LLM
    "get_llm_client",
    # Dependency injection - Neo4j
    "get_neo4j_client",
    # Dependency injection - Synthesis
    "get_synthesis_client",
    "retrieve_content",
    "search_taxonomy",
    "set_content_client",
    "set_graph_client",
    "set_llm_client",
    "set_neo4j_client",
    "set_synthesis_client",
    "synthesize",
    "traverse_graph",
]
