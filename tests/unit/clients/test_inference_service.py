"""Unit tests for InferenceServiceClient.

TDD RED phase for model resolution with preferences.

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Preset Selection
Reference: inference-service/docs/ARCHITECTURE.md → Model Configuration
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.inference_service import (
    InferenceServiceClient,
    ModelInfo,
    ModelResolver,
    create_inference_client,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_models_response() -> dict[str, Any]:
    """Mock response from GET /v1/models.
    
    Reference: inference-service/docs/ARCHITECTURE.md → Models List Response
    """
    return {
        "data": [
            {
                "id": "deepseek-r1-7b",
                "status": "loaded",
                "memory_mb": 4700,
                "context_length": 32768,
                "roles": ["thinker"],
            },
            {
                "id": "qwen2.5-7b",
                "status": "loaded",
                "memory_mb": 4400,
                "context_length": 32768,
                "roles": ["coder", "primary"],
            },
            {
                "id": "llama-3.2-3b",
                "status": "available",  # Not loaded
                "memory_mb": 2000,
                "context_length": 8192,
                "roles": ["fast"],
            },
        ],
        "config": "D4",
        "orchestration_mode": "critique",
    }


@pytest.fixture
def mock_single_model_response() -> dict[str, Any]:
    """Mock response with single loaded model."""
    return {
        "data": [
            {
                "id": "phi-4",
                "status": "loaded",
                "memory_mb": 8400,
                "context_length": 16384,
                "roles": ["primary", "thinker", "coder"],
            },
        ],
        "config": "S1",
        "orchestration_mode": "single",
    }


# =============================================================================
# Test ModelInfo
# =============================================================================


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_model_info_creation(self) -> None:
        """ModelInfo stores model metadata correctly."""
        info = ModelInfo(
            model_id="deepseek-r1-7b",
            status="loaded",
            memory_mb=4700,
            context_length=32768,
            roles=["thinker"],
        )
        assert info.model_id == "deepseek-r1-7b"
        assert info.status == "loaded"
        assert info.is_loaded is True
        assert "thinker" in info.roles

    def test_model_info_is_loaded_false(self) -> None:
        """is_loaded returns False for available but not loaded models."""
        info = ModelInfo(
            model_id="llama-3.2-3b",
            status="available",
            memory_mb=2000,
            context_length=8192,
            roles=["fast"],
        )
        assert info.is_loaded is False

    def test_model_info_has_role(self) -> None:
        """has_role() checks if model supports a role."""
        info = ModelInfo(
            model_id="qwen2.5-7b",
            status="loaded",
            memory_mb=4400,
            context_length=32768,
            roles=["coder", "primary"],
        )
        assert info.has_role("coder") is True
        assert info.has_role("primary") is True
        assert info.has_role("thinker") is False


# =============================================================================
# Test ModelResolver
# =============================================================================


class TestModelResolver:
    """Test ModelResolver for preference-based model selection.
    
    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Preset Selection
    
    Key behavior:
    - Agent functions express preferences, not requirements
    - If preferred model is loaded → use it
    - If not → fallback to role-based lookup or any loaded model
    """

    def test_resolve_with_preferred_model_loaded(
        self, mock_models_response: dict[str, Any]
    ) -> None:
        """When preferred model is loaded, use it."""
        resolver = ModelResolver(mock_models_response)
        
        # deepseek-r1-7b is loaded, so it should be selected
        model = resolver.resolve(preferred="deepseek-r1-7b")
        assert model == "deepseek-r1-7b"

    def test_resolve_with_preferred_model_not_loaded(
        self, mock_models_response: dict[str, Any]
    ) -> None:
        """When preferred model not loaded, fallback to any loaded model."""
        resolver = ModelResolver(mock_models_response)
        
        # llama-3.2-3b is available but not loaded
        model = resolver.resolve(preferred="llama-3.2-3b")
        
        # Should fallback to one of the loaded models
        assert model in ["deepseek-r1-7b", "qwen2.5-7b"]

    def test_resolve_with_role_fallback(
        self, mock_models_response: dict[str, Any]
    ) -> None:
        """When preferred model not loaded, try role-based fallback."""
        resolver = ModelResolver(mock_models_response)
        
        # phi-4 not in list, but we want a "thinker" role
        model = resolver.resolve(
            preferred="phi-4",
            fallback_roles=["thinker"],
        )
        
        # deepseek-r1-7b has thinker role and is loaded
        assert model == "deepseek-r1-7b"

    def test_resolve_with_multiple_role_fallbacks(
        self, mock_models_response: dict[str, Any]
    ) -> None:
        """Fallback roles are tried in order."""
        resolver = ModelResolver(mock_models_response)
        
        # Try roles in order: longctx (none), coder (qwen2.5-7b)
        model = resolver.resolve(
            preferred="phi-3-medium-128k",
            fallback_roles=["longctx", "coder"],
        )
        
        # qwen2.5-7b has coder role
        assert model == "qwen2.5-7b"

    def test_resolve_returns_any_loaded_when_no_match(
        self, mock_models_response: dict[str, Any]
    ) -> None:
        """When no preference or role matches, return any loaded model."""
        resolver = ModelResolver(mock_models_response)
        
        model = resolver.resolve(
            preferred="nonexistent-model",
            fallback_roles=["nonexistent-role"],
        )
        
        # Should return one of the loaded models
        assert model in ["deepseek-r1-7b", "qwen2.5-7b"]

    def test_resolve_with_single_model(
        self, mock_single_model_response: dict[str, Any]
    ) -> None:
        """With single model loaded, always return it regardless of preference."""
        resolver = ModelResolver(mock_single_model_response)
        
        # Even though we prefer deepseek, only phi-4 is loaded
        model = resolver.resolve(
            preferred="deepseek-r1-7b",
            fallback_roles=["thinker"],
        )
        
        assert model == "phi-4"

    def test_get_loaded_models(
        self, mock_models_response: dict[str, Any]
    ) -> None:
        """get_loaded_models() returns only loaded models."""
        resolver = ModelResolver(mock_models_response)
        
        loaded = resolver.get_loaded_models()
        
        assert len(loaded) == 2
        assert all(m.is_loaded for m in loaded)
        model_ids = [m.model_id for m in loaded]
        assert "deepseek-r1-7b" in model_ids
        assert "qwen2.5-7b" in model_ids
        assert "llama-3.2-3b" not in model_ids

    def test_get_current_config(
        self, mock_models_response: dict[str, Any]
    ) -> None:
        """get_current_config() returns the active preset."""
        resolver = ModelResolver(mock_models_response)
        
        assert resolver.get_current_config() == "D4"

    def test_get_orchestration_mode(
        self, mock_models_response: dict[str, Any]
    ) -> None:
        """get_orchestration_mode() returns current mode."""
        resolver = ModelResolver(mock_models_response)
        
        assert resolver.get_orchestration_mode() == "critique"


# =============================================================================
# Test InferenceServiceClient Model Discovery
# =============================================================================


class TestInferenceServiceClientModelDiscovery:
    """Test InferenceServiceClient's model discovery capabilities."""

    @pytest.mark.asyncio
    async def test_get_models_returns_resolver(
        self, mock_models_response: dict[str, Any]
    ) -> None:
        """get_models() queries inference-service and returns ModelResolver."""
        client = InferenceServiceClient("http://localhost:8085")
        
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_models_response
            mock_response.raise_for_status = MagicMock()
            mock_http.get.return_value = mock_response
            mock_get_client.return_value = mock_http
            
            resolver = await client.get_models()
            
            mock_http.get.assert_called_once_with("/v1/models")
            assert isinstance(resolver, ModelResolver)
            assert resolver.get_current_config() == "D4"

    @pytest.mark.asyncio
    async def test_complete_resolves_model_from_preference(
        self, mock_models_response: dict[str, Any]
    ) -> None:
        """complete() uses model preference with fallback."""
        client = InferenceServiceClient("http://localhost:8085")
        
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            
            # First call: get models
            mock_models_resp = MagicMock()
            mock_models_resp.json.return_value = mock_models_response
            mock_models_resp.raise_for_status = MagicMock()
            
            # Second call: chat completion
            mock_completion_resp = MagicMock()
            mock_completion_resp.json.return_value = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1703704800,
                "model": "deepseek-r1-7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Test response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            mock_completion_resp.raise_for_status = MagicMock()
            
            mock_http.get.return_value = mock_models_resp
            mock_http.post.return_value = mock_completion_resp
            mock_get_client.return_value = mock_http
            
            result = await client.complete(
                messages=[{"role": "user", "content": "Hello"}],
                model_preference="deepseek-r1-7b",
            )
            
            assert result == "Test response"
            
            # Verify completion was called with resolved model
            call_args = mock_http.post.call_args
            request_body = call_args[1]["json"]
            assert request_body["model"] == "deepseek-r1-7b"


# =============================================================================
# Test Agent Function Model Preferences
# =============================================================================


class TestAgentFunctionModelPreferences:
    """Test model preferences defined per agent function.
    
    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Quality/Latency Tradeoff Matrix
    
    | Agent Function | Light (Fast) | Standard | High Quality |
    |----------------|--------------|----------|--------------|
    | summarize_content | S4 | D4 | T1 |
    | generate_code | S3 | D4 | T1 |
    | analyze_artifact | S3 | D4 | D3 |
    """

    def test_summarize_content_preference(self) -> None:
        """summarize_content prefers deepseek-r1-7b (thinker role) by default."""
        from src.clients.inference_service import MODEL_PREFERENCES
        
        prefs = MODEL_PREFERENCES["summarize_content"]
        
        # Standard quality preference
        assert prefs["preferred"] == "deepseek-r1-7b"
        assert "thinker" in prefs["fallback_roles"]

    def test_generate_code_preference(self) -> None:
        """generate_code prefers qwen2.5-7b (coder role)."""
        from src.clients.inference_service import MODEL_PREFERENCES
        
        prefs = MODEL_PREFERENCES["generate_code"]
        
        assert prefs["preferred"] == "qwen2.5-7b"
        assert "coder" in prefs["fallback_roles"]

    def test_extract_structure_preference(self) -> None:
        """extract_structure prefers llama-3.2-3b (fast role)."""
        from src.clients.inference_service import MODEL_PREFERENCES
        
        prefs = MODEL_PREFERENCES["extract_structure"]
        
        assert prefs["preferred"] == "llama-3.2-3b"
        assert "fast" in prefs["fallback_roles"]


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateInferenceClient:
    """Test create_inference_client factory."""

    def test_create_with_defaults(self) -> None:
        """Creates client with default URL from env or localhost."""
        with patch.dict("os.environ", {}, clear=True):
            client = create_inference_client()
            assert client.base_url == "http://localhost:8085"

    def test_create_with_env_url(self) -> None:
        """Creates client with URL from environment."""
        with patch.dict(
            "os.environ", 
            {"INFERENCE_SERVICE_URL": "http://inference:8085"},
        ):
            client = create_inference_client()
            assert client.base_url == "http://inference:8085"

    def test_create_with_explicit_url(self) -> None:
        """Creates client with explicitly provided URL."""
        client = create_inference_client(base_url="http://custom:9000")
        assert client.base_url == "http://custom:9000"
