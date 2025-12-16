"""MSEP Service Clients.

HTTP clients for Code-Orchestrator-Service and semantic-search-service.

Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-3
"""

from src.clients.code_orchestrator import CodeOrchestratorClient
from src.clients.protocols import CodeOrchestratorProtocol, SemanticSearchProtocol
from src.clients.semantic_search import MSEPSemanticSearchClient

__all__ = [
    "CodeOrchestratorClient",
    "CodeOrchestratorProtocol",
    "MSEPSemanticSearchClient",
    "SemanticSearchProtocol",
]
