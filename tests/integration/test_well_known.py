"""Integration tests for WBS-PI2: A2A Agent Card & Discovery.

Tests the complete well-known endpoint flow end-to-end with feature flags.

Acceptance Criteria Coverage:
- AC-PI2.4: GET /.well-known/agent-card.json returns 404 when disabled
- AC-PI2.5: GET /.well-known/agent-card.json returns valid card when enabled

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → Agent Card Discovery
"""

import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def clean_env():
    """Clean environment before each test."""
    # Store original values
    orig_a2a = os.environ.get("AGENTS_A2A_ENABLED")
    orig_card = os.environ.get("AGENTS_A2A_AGENT_CARD_ENABLED")
    
    # Clear for test
    os.environ.pop("AGENTS_A2A_ENABLED", None)
    os.environ.pop("AGENTS_A2A_AGENT_CARD_ENABLED", None)
    
    # Clear cache
    from src.config.feature_flags import get_feature_flags
    get_feature_flags.cache_clear()
    
    yield
    
    # Restore
    if orig_a2a:
        os.environ["AGENTS_A2A_ENABLED"] = orig_a2a
    else:
        os.environ.pop("AGENTS_A2A_ENABLED", None)
    
    if orig_card:
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = orig_card
    else:
        os.environ.pop("AGENTS_A2A_AGENT_CARD_ENABLED", None)
    
    get_feature_flags.cache_clear()


class TestWellKnownIntegration:
    """Integration tests for /.well-known/agent-card.json endpoint."""

    def test_agent_card_disabled_by_default(self, clean_env) -> None:
        """Agent Card returns 404 by default (AC-PI2.4)."""
        from src.main import app
        
        client = TestClient(app)
        response = client.get("/.well-known/agent-card.json")
        
        assert response.status_code == 404
        assert "not available" in response.json()["detail"].lower()

    def test_agent_card_enabled_returns_valid_card(self, clean_env) -> None:
        """Agent Card returns valid card when enabled (AC-PI2.5)."""
        os.environ["AGENTS_A2A_ENABLED"] = "true"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "true"
        
        from src.config.feature_flags import get_feature_flags
        get_feature_flags.cache_clear()
        
        from src.main import app
        
        client = TestClient(app)
        response = client.get("/.well-known/agent-card.json")
        
        assert response.status_code == 200
        
        card = response.json()
        assert card["protocolVersion"] == "0.3.0"
        assert card["name"] == "ai-agents-service"
        assert "AI Platform" in card["description"]
        assert card["version"] == "1.0.0"
        assert "capabilities" in card
        assert "skills" in card
        assert len(card["skills"]) == 8

    def test_agent_card_skills_match_function_registry(self, clean_env) -> None:
        """Agent Card skills match FUNCTION_REGISTRY."""
        os.environ["AGENTS_A2A_ENABLED"] = "true"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "true"
        
        from src.config.feature_flags import get_feature_flags
        get_feature_flags.cache_clear()
        
        from src.main import app
        
        client = TestClient(app)
        response = client.get("/.well-known/agent-card.json")
        
        card = response.json()
        skill_ids = {skill["id"] for skill in card["skills"]}
        
        expected_ids = {
            "extract_structure",
            "summarize_content",
            "generate_code",
            "analyze_artifact",
            "validate_against_spec",
            "decompose_task",
            "synthesize_outputs",
            "cross_reference",
        }
        
        assert skill_ids == expected_ids

    def test_agent_card_capabilities_reflect_flags(self, clean_env) -> None:
        """Agent Card capabilities reflect feature flags (AC-PI2.7)."""
        os.environ["AGENTS_A2A_ENABLED"] = "true"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "true"
        os.environ["AGENTS_A2A_STREAMING_ENABLED"] = "true"
        os.environ["AGENTS_A2A_PUSH_NOTIFICATIONS"] = "true"
        
        from src.config.feature_flags import get_feature_flags
        get_feature_flags.cache_clear()
        
        from src.main import app
        
        client = TestClient(app)
        response = client.get("/.well-known/agent-card.json")
        
        card = response.json()
        capabilities = card["capabilities"]
        
        assert capabilities["streaming"] is True
        assert capabilities["pushNotifications"] is True
        assert capabilities["stateTransitionHistory"] is True
