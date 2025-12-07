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

from src.agents.cross_reference.nodes.analyze_source import analyze_source
from src.agents.cross_reference.nodes.search_taxonomy import search_taxonomy
from src.agents.cross_reference.nodes.traverse_graph import traverse_graph
from src.agents.cross_reference.nodes.retrieve_content import retrieve_content
from src.agents.cross_reference.nodes.synthesize import synthesize

__all__ = [
    "analyze_source",
    "search_taxonomy",
    "traverse_graph",
    "retrieve_content",
    "synthesize",
]
