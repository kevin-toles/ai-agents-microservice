"""Unit tests for src/core/http module.

Tests HTTP client factory for Kitchen Brigade service communication.

Reference: WBS-AGT2 AC-2.4
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import httpx

from src.core.http import (
    HTTPClientFactory,
    ServiceName,
    get_http_client_factory,
)


class TestServiceName:
    """Tests for ServiceName enum."""
    
    def test_service_names_defined(self) -> None:
        """Test all expected service names are defined."""
        assert ServiceName.INFERENCE.value == "inference-service"
        assert ServiceName.SEMANTIC_SEARCH.value == "semantic-search-service"
        assert ServiceName.AUDIT.value == "audit-service"
        assert ServiceName.CODE_ORCHESTRATOR.value == "code-orchestrator"
        assert ServiceName.LLM_GATEWAY.value == "llm-gateway"
        assert ServiceName.CODE_REFERENCE.value == "code-reference-engine"
    
    def test_service_name_string_behavior(self) -> None:
        """Test ServiceName behaves as string."""
        # ServiceName inherits from str
        assert isinstance(ServiceName.INFERENCE, str)
        assert ServiceName.INFERENCE == "inference-service"


class TestHTTPClientFactory:
    """Tests for HTTPClientFactory class."""
    
    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings for testing."""
        settings = MagicMock()
        settings.inference_service_url = "http://localhost:8085"
        settings.semantic_search_url = "http://localhost:8081"
        settings.audit_service_url = "http://localhost:8084"
        settings.code_orchestrator_url = "http://localhost:8083"
        settings.llm_gateway_url = "http://localhost:8080"
        settings.code_reference_engine_url = "http://localhost:8086"
        settings.http_timeout_seconds = 30
        return settings
    
    def test_init_with_settings(self, mock_settings: MagicMock) -> None:
        """Test factory initialization with provided settings."""
        factory = HTTPClientFactory(settings=mock_settings)
        
        assert factory._settings == mock_settings
    
    def test_init_without_settings(self) -> None:
        """Test factory initialization uses get_settings()."""
        with patch("src.core.http.get_settings") as mock_get:
            mock_get.return_value = MagicMock(
                inference_service_url="http://localhost:8085",
                semantic_search_url="http://localhost:8081",
                audit_service_url="http://localhost:8084",
                code_orchestrator_url="http://localhost:8083",
                llm_gateway_url="http://localhost:8080",
                code_reference_engine_url="http://localhost:8086",
            )
            
            factory = HTTPClientFactory()
            
            mock_get.assert_called_once()
    
    def test_get_base_url_inference(self, mock_settings: MagicMock) -> None:
        """Test getting base URL for inference service."""
        factory = HTTPClientFactory(settings=mock_settings)
        
        url = factory.get_base_url(ServiceName.INFERENCE)
        
        assert url == "http://localhost:8085"
    
    def test_get_base_url_semantic_search(self, mock_settings: MagicMock) -> None:
        """Test getting base URL for semantic search service."""
        factory = HTTPClientFactory(settings=mock_settings)
        
        url = factory.get_base_url(ServiceName.SEMANTIC_SEARCH)
        
        assert url == "http://localhost:8081"
    
    def test_get_base_url_audit(self, mock_settings: MagicMock) -> None:
        """Test getting base URL for audit service."""
        factory = HTTPClientFactory(settings=mock_settings)
        
        url = factory.get_base_url(ServiceName.AUDIT)
        
        assert url == "http://localhost:8084"
    
    def test_get_base_url_all_services(self, mock_settings: MagicMock) -> None:
        """Test getting base URLs for all services."""
        factory = HTTPClientFactory(settings=mock_settings)
        
        expected = {
            ServiceName.INFERENCE: "http://localhost:8085",
            ServiceName.SEMANTIC_SEARCH: "http://localhost:8081",
            ServiceName.AUDIT: "http://localhost:8084",
            ServiceName.CODE_ORCHESTRATOR: "http://localhost:8083",
            ServiceName.LLM_GATEWAY: "http://localhost:8080",
            ServiceName.CODE_REFERENCE: "http://localhost:8086",
        }
        
        for service, expected_url in expected.items():
            assert factory.get_base_url(service) == expected_url
    
    @pytest.mark.asyncio
    async def test_get_client_context_manager(self, mock_settings: MagicMock) -> None:
        """Test get_client returns async context manager."""
        factory = HTTPClientFactory(settings=mock_settings)
        
        async with factory.get_client(ServiceName.INFERENCE) as client:
            assert isinstance(client, httpx.AsyncClient)
            # httpx may or may not include trailing slash
            assert "localhost:8085" in str(client.base_url)
    
    @pytest.mark.asyncio
    async def test_get_client_with_custom_timeout(self, mock_settings: MagicMock) -> None:
        """Test get_client respects custom timeout."""
        factory = HTTPClientFactory(settings=mock_settings)
        
        async with factory.get_client(ServiceName.INFERENCE, timeout=60.0) as client:
            assert client.timeout.connect == 60.0
    
    def test_create_client_standalone(self, mock_settings: MagicMock) -> None:
        """Test create_client returns standalone client."""
        factory = HTTPClientFactory(settings=mock_settings)
        
        client = factory.create_client(ServiceName.SEMANTIC_SEARCH)
        
        assert isinstance(client, httpx.AsyncClient)
        # httpx may or may not include trailing slash
        assert "localhost:8081" in str(client.base_url)


class TestGetHTTPClientFactory:
    """Tests for get_http_client_factory function."""
    
    def test_returns_factory_instance(self) -> None:
        """Test function returns HTTPClientFactory instance."""
        # Reset the module-level singleton for testing
        import src.core.http as http_module
        http_module._factory = None
        
        with patch("src.core.http.get_settings") as mock_get:
            mock_get.return_value = MagicMock(
                inference_service_url="http://localhost:8085",
                semantic_search_url="http://localhost:8081",
                audit_service_url="http://localhost:8084",
                code_orchestrator_url="http://localhost:8083",
                llm_gateway_url="http://localhost:8080",
                code_reference_engine_url="http://localhost:8086",
            )
            
            factory = get_http_client_factory()
            
            assert isinstance(factory, HTTPClientFactory)
    
    def test_returns_singleton(self) -> None:
        """Test function returns same instance on multiple calls."""
        # Reset the module-level singleton for testing
        import src.core.http as http_module
        http_module._factory = None
        
        with patch("src.core.http.get_settings") as mock_get:
            mock_get.return_value = MagicMock(
                inference_service_url="http://localhost:8085",
                semantic_search_url="http://localhost:8081",
                audit_service_url="http://localhost:8084",
                code_orchestrator_url="http://localhost:8083",
                llm_gateway_url="http://localhost:8080",
                code_reference_engine_url="http://localhost:8086",
            )
            
            factory1 = get_http_client_factory()
            factory2 = get_http_client_factory()
            
            assert factory1 is factory2
