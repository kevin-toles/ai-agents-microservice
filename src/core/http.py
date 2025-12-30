"""HTTP client factory for downstream service communication.

Provides a factory pattern for creating HTTP clients to communicate with
Kitchen Brigade services. All clients use httpx for async HTTP operations.

Kitchen Brigade Services:
- inference-service :8085 (Line Cook) - LLM completions
- semantic-search-service :8081 (Cookbook) - Vector search
- audit-service :8084 (Auditor) - Citation tracking
- code-orchestrator :8083 (Sous Chef) - Code analysis

Pattern: Factory Pattern with Protocol Duck Typing
Reference: WBS-AGT2 AC-2.4, AGENT_FUNCTIONS_ARCHITECTURE.md
"""

from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, AsyncGenerator

import httpx

from src.core.config import Settings, get_settings
from src.core.logging import get_logger


logger = get_logger(__name__)


class ServiceName(str, Enum):
    """Kitchen Brigade service identifiers.
    
    Each service has a specific role in the architecture:
    - INFERENCE: Line Cook - handles LLM inference via llama.cpp
    - SEMANTIC_SEARCH: Cookbook - handles vector search via Qdrant
    - AUDIT: Auditor - tracks citations and generates footnotes
    - CODE_ORCHESTRATOR: Sous Chef - runs HuggingFace models
    - LLM_GATEWAY: Router - single entry point for external requests
    - CODE_REFERENCE: Pantry - code reference engine integration
    """
    INFERENCE = "inference-service"
    SEMANTIC_SEARCH = "semantic-search-service"
    AUDIT = "audit-service"
    CODE_ORCHESTRATOR = "code-orchestrator"
    LLM_GATEWAY = "llm-gateway"
    CODE_REFERENCE = "code-reference-engine"


class HTTPClientFactory:
    """Factory for creating HTTP clients to downstream services.
    
    Provides centralized client creation with:
    - Consistent timeout configuration
    - Service discovery via Settings
    - Request logging and tracing
    - Connection pooling
    
    Example:
        ```python
        factory = HTTPClientFactory()
        async with factory.get_client(ServiceName.INFERENCE) as client:
            response = await client.post("/v1/completions", json=payload)
        ```
    
    Pattern: Factory Pattern
    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
    """
    
    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the HTTP client factory.
        
        Args:
            settings: Application settings. Uses get_settings() if not provided.
        """
        self._settings = settings or get_settings()
        self._service_urls = self._build_service_url_map()
    
    def _build_service_url_map(self) -> dict[ServiceName, str]:
        """Build mapping of service names to URLs.
        
        Returns:
            Dictionary mapping ServiceName enum to base URL.
        """
        return {
            ServiceName.INFERENCE: self._settings.inference_service_url,
            ServiceName.SEMANTIC_SEARCH: self._settings.semantic_search_url,
            ServiceName.AUDIT: self._settings.audit_service_url,
            ServiceName.CODE_ORCHESTRATOR: self._settings.code_orchestrator_url,
            ServiceName.LLM_GATEWAY: self._settings.llm_gateway_url,
            ServiceName.CODE_REFERENCE: self._settings.code_reference_engine_url,
        }
    
    def get_base_url(self, service: ServiceName) -> str:
        """Get the base URL for a service.
        
        Args:
            service: The target service.
        
        Returns:
            Base URL for the service.
        
        Raises:
            ValueError: If service is not configured.
        """
        url = self._service_urls.get(service)
        if not url:
            raise ValueError(f"No URL configured for service: {service}")
        return url
    
    @asynccontextmanager
    async def get_client(
        self,
        service: ServiceName,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Get an HTTP client for a specific service.
        
        Args:
            service: Target service to communicate with.
            timeout: Request timeout in seconds. Uses settings default if not specified.
            **kwargs: Additional arguments passed to httpx.AsyncClient.
        
        Yields:
            Configured httpx.AsyncClient instance.
        
        Example:
            ```python
            async with factory.get_client(ServiceName.INFERENCE) as client:
                response = await client.post("/v1/completions", json={"prompt": "..."})
                result = response.json()
            ```
        """
        base_url = self.get_base_url(service)
        request_timeout = timeout or self._settings.http_timeout_seconds
        
        logger.debug(
            "Creating HTTP client",
            service=service.value,
            base_url=base_url,
            timeout=request_timeout,
        )
        
        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(request_timeout),
            **kwargs,
        ) as client:
            yield client
    
    def create_client(
        self,
        service: ServiceName,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> httpx.AsyncClient:
        """Create a standalone HTTP client (caller manages lifecycle).
        
        Use get_client() context manager when possible. This method is for
        cases where you need to manage the client lifecycle manually.
        
        Args:
            service: Target service to communicate with.
            timeout: Request timeout in seconds.
            **kwargs: Additional arguments passed to httpx.AsyncClient.
        
        Returns:
            Configured httpx.AsyncClient instance.
        
        Warning:
            Caller is responsible for calling `await client.aclose()`.
        """
        base_url = self.get_base_url(service)
        request_timeout = timeout or self._settings.http_timeout_seconds
        
        return httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(request_timeout),
            **kwargs,
        )


# Module-level factory instance (lazy initialization)
_factory: HTTPClientFactory | None = None


def get_http_client_factory() -> HTTPClientFactory:
    """Get the shared HTTP client factory instance.
    
    Returns:
        Shared HTTPClientFactory singleton.
    
    Example:
        ```python
        from src.core.http import get_http_client_factory, ServiceName
        
        factory = get_http_client_factory()
        async with factory.get_client(ServiceName.INFERENCE) as client:
            response = await client.get("/health")
        ```
    """
    global _factory
    if _factory is None:
        _factory = HTTPClientFactory()
    return _factory
