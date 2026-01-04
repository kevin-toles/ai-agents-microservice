"""MCP Module - Server and Client Components.

WBS Reference: 
- WBS-KB8 - VS Code MCP Server
- WBS-PI5 - MCP Server Core
- WBS-PI5b - MCP Lifecycle Integration  
- WBS-PI6 - MCP Client & Toolbox

Server Components:
- McpServer: FastMCP-based server exposing agent functions as MCP tools
- McpServerManager: Lifecycle management for MCP server

Client Components:
- SemanticSearchMcpWrapper: Wraps semantic-search-service for hybrid RAG
- McpToolboxManager: Wraps genai-toolbox for Neo4j/Redis operations
"""
from src.mcp.semantic_search_wrapper import SemanticSearchMcpWrapper
from src.mcp.toolbox_manager import McpToolboxManager

__all__ = [
    "SemanticSearchMcpWrapper",
    "McpToolboxManager",
]
