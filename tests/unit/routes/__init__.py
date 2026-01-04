"""Tests for A2A well-known endpoint.

TDD tests for WBS-PI2: A2A Agent Card & Discovery - Well-Known Endpoint.

Acceptance Criteria Coverage:
- AC-PI2.4: GET /.well-known/agent-card.json returns 404 when disabled
- AC-PI2.5: GET /.well-known/agent-card.json returns valid card when enabled

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → Agent Card Discovery
"""

import pytest
from fastapi.testclient import TestClient


# =============================================================================
# AC-PI2.4, AC-PI2.5: Well-Known Endpoint Tests
# =============================================================================


class TestWellKnownEndpoint:
    """Tests for /.well-known/agent-card.json endpoint."""

    def test_well_known_endpoint_exists(self) -> None:
        """Well-known router can be imported."""
        from src.routes.well_known import router

        assert router is not None

    def test_well_known_endpoint_returns_404_when_disabled(self) -> None:
        """GET /.well-known/agent-card.json returns 404 when feature disabled (AC-PI2.4)."""
        from src.main import app
        import os

        # Ensure flags are disabled
        os.environ["AGENTS_A2A_ENABLED"] = "false"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "false"

        client = TestClient(app)
        response = client.get("/.well-known/agent-card.json")

        assert response.status_code == 404

    def test_well_known_endpoint_returns_200_when_enabled(self) -> None:
        """GET /.well-known/agent-card.json returns 200 when feature enabled (AC-PI2.5)."""
        from src.main import app
        import os

        # Enable A2A and agent card
        os.environ["AGENTS_A2A_ENABLED"] = "true"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "true"

        # Need to reload feature flags for test
        from src.config.feature_flags import get_feature_flags
        get_feature_flags.cache_clear()

        client = TestClient(app)
        response = client.get("/.well-known/agent-card.json")

        assert response.status_code == 200

        # Cleanup
        os.environ["AGENTS_A2A_ENABLED"] = "false"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "false"
        get_feature_flags.cache_clear()

    def test_well_known_endpoint_returns_json(self) -> None:
        """GET /.well-known/agent-card.json returns JSON content type."""
        from src.main import app
        import os

        os.environ["AGENTS_A2A_ENABLED"] = "true"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "true"

        from src.config.feature_flags import get_feature_flags
        get_feature_flags.cache_clear()

        client = TestClient(app)
        response = client.get("/.well-known/agent-card.json")

        assert "application/json" in response.headers["content-type"]

        # Cleanup
        os.environ["AGENTS_A2A_ENABLED"] = "false"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "false"
        get_feature_flags.cache_clear()

    def test_well_known_endpoint_returns_valid_agent_card(self) -> None:
        """GET /.well-known/agent-card.json returns valid AgentCard structure (AC-PI2.5)."""
        from src.main import app
        import os

        os.environ["AGENTS_A2A_ENABLED"] = "true"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "true"

        from src.config.feature_flags import get_feature_flags
        get_feature_flags.cache_clear()

        client = TestClient(app)
        response = client.get("/.well-known/agent-card.json")

        assert response.status_code == 200

        data = response.json()
        assert data["protocolVersion"] == "0.3.0"
        assert data["name"] == "ai-agents-service"
        assert "capabilities" in data
        assert "skills" in data
        assert len(data["skills"]) == 8

        # Cleanup
        os.environ["AGENTS_A2A_ENABLED"] = "false"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "false"
        get_feature_flags.cache_clear()

    def test_well_known_endpoint_requires_both_flags_enabled(self) -> None:
        """Endpoint requires both a2a_enabled and a2a_agent_card_enabled."""
        from src.main import app
        import os

        # Test with A2A enabled but agent card disabled
        os.environ["AGENTS_A2A_ENABLED"] = "true"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "false"

        from src.config.feature_flags import get_feature_flags
        get_feature_flags.cache_clear()

        client = TestClient(app)
        response = client.get("/.well-known/agent-card.json")

        assert response.status_code == 404

        # Cleanup
        os.environ["AGENTS_A2A_ENABLED"] = "false"
        get_feature_flags.cache_clear()

    def test_well_known_endpoint_skills_have_required_fields(self) -> None:
        """Skills in returned card have all required fields."""
        from src.main import app
        import os

        os.environ["AGENTS_A2A_ENABLED"] = "true"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "true"

        from src.config.feature_flags import get_feature_flags
        get_feature_flags.cache_clear()

        client = TestClient(app)
        response = client.get("/.well-known/agent-card.json")

        data = response.json()
        skills = data["skills"]

        for skill in skills:
            assert "id" in skill
            assert "name" in skill
            assert "description" in skill
            assert "tags" in skill
            assert "inputModes" in skill
            assert "outputModes" in skill

        # Cleanup
        os.environ["AGENTS_A2A_ENABLED"] = "false"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "false"
        get_feature_flags.cache_clear()
