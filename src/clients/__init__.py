"""MSEP Service Clients.

HTTP clients for Code-Orchestrator-Service, semantic-search-service,
and inference-service.

Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-3
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md - WBS-AGT7.7
"""

from src.clients.code_orchestrator import CodeOrchestratorClient
from src.clients.inference_service import (
    InferenceServiceClient,
    create_inference_client,
    MODEL_PREFERENCES,
    ModelInfo,
    ModelResolver,
)
from src.clients.protocols import CodeOrchestratorProtocol, SemanticSearchProtocol
from src.clients.semantic_search import MSEPSemanticSearchClient


__all__ = [
    "CodeOrchestratorClient",
    "CodeOrchestratorProtocol",
    "InferenceServiceClient",
    "MODEL_PREFERENCES",
    "MSEPSemanticSearchClient",
    "ModelInfo",
    "ModelResolver",
    "SemanticSearchProtocol",
    "create_inference_client",
]
