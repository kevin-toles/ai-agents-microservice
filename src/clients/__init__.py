"""MSEP Service Clients.

HTTP clients for Code-Orchestrator-Service, semantic-search-service,
inference-service, code-reference-engine, Neo4j graph database, and book passages.

Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-3
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md - WBS-AGT7.7
Reference: WBS-AGT21 - Code Reference Engine Client
Reference: WBS-AGT22 - Neo4j Graph Integration
Reference: WBS-AGT23 - Book/JSON Passage Retrieval
Reference: WBS-KB7 - Code-Orchestrator Tool Integration
"""

from src.clients.book_passage import (
    BookPassageClient,
    BookPassageClientConfig,
)
from src.clients.code_analysis import (
    CodeAnalysisClient,
    CodeAnalysisConfig,
    CodeAnalysisProtocol,
    CodeRankingResult,
    FakeCodeAnalysisClient,
    KeywordResult,
    TermValidationResult,
)
from src.clients.code_orchestrator import CodeOrchestratorClient
from src.clients.code_reference import (
    CodeChunk,
    CodeContext,
    CodeReference,
    CodeReferenceClient,
    CodeReferenceConfig,
)
from src.clients.inference_service import (
    InferenceServiceClient,
    create_inference_client,
    MODEL_PREFERENCES,
    ModelInfo,
    ModelResolver,
)
from src.clients.neo4j_client import (
    Neo4jClient,
    Neo4jClientConfig,
)
from src.clients.protocols import (
    BookPassageClientProtocol,
    CodeOrchestratorProtocol,
    CodeReferenceProtocol,
    Neo4jClientProtocol,
    SemanticSearchProtocol,
)
from src.clients.semantic_search import MSEPSemanticSearchClient
from src.clients.sonarqube import (
    ClaimValidationResult,
    FakeSonarQubeClient,
    SonarQubeAnalysisResult,
    SonarQubeClient,
    SonarQubeConfig,
    SonarQubeIssue,
    SonarQubeMetrics,
    SonarQubeProtocol,
)


__all__ = [
    "BookPassageClient",
    "BookPassageClientConfig",
    "BookPassageClientProtocol",
    "ClaimValidationResult",
    "CodeAnalysisClient",
    "CodeAnalysisConfig",
    "CodeAnalysisProtocol",
    "CodeChunk",
    "CodeContext",
    "CodeOrchestratorClient",
    "CodeOrchestratorProtocol",
    "CodeRankingResult",
    "CodeReference",
    "CodeReferenceClient",
    "CodeReferenceConfig",
    "CodeReferenceProtocol",
    "FakeCodeAnalysisClient",
    "FakeSonarQubeClient",
    "InferenceServiceClient",
    "KeywordResult",
    "MODEL_PREFERENCES",
    "MSEPSemanticSearchClient",
    "ModelInfo",
    "ModelResolver",
    "Neo4jClient",
    "Neo4jClientConfig",
    "Neo4jClientProtocol",
    "SemanticSearchProtocol",
    "SonarQubeAnalysisResult",
    "SonarQubeClient",
    "SonarQubeConfig",
    "SonarQubeIssue",
    "SonarQubeMetrics",
    "SonarQubeProtocol",
    "TermValidationResult",
    "create_inference_client",
]

