"""Tools for Cross-Reference Agent.

These are the functions the agent can invoke to interact with
external systems (semantic-search-service, Neo4j, etc.).

Pattern: LangChain Tool interface
Source: ARCHITECTURE.md (ai-agents) tool definitions
"""

from src.agents.cross_reference.tools.content import get_chapter_text
from src.agents.cross_reference.tools.graph import traverse_graph
from src.agents.cross_reference.tools.metadata import get_chapter_metadata
from src.agents.cross_reference.tools.similarity import search_similar
from src.agents.cross_reference.tools.taxonomy import search_taxonomy


__all__ = [
    "get_chapter_metadata",
    "get_chapter_text",
    "search_similar",
    "search_taxonomy",
    "traverse_graph",
]
