"""Unit tests for Protocol Feature Flags (WBS-PI1).

TDD Phase: RED - These tests should FAIL until implementation is complete.

Reference: WBS_PROTOCOL_INTEGRATION.md → WBS-PI1: Feature Flags & Configuration
Anti-Patterns Avoided:
- CODING_PATTERNS_ANALYSIS §1: Full type annotations
- CODING_PATTERNS_ANALYSIS §2: No cognitive complexity > 15
- CODING_PATTERNS_ANALYSIS §3: No bare except clauses

Acceptance Criteria Tested:
- AC-PI1.1: ProtocolFeatureFlags loads from environment
- AC-PI1.2: All flags default to False
- AC-PI1.3: A2A master switch gates sub-features
- AC-PI1.4: MCP master switch gates sub-features
- AC-PI1.5: a2a_available() logic
- AC-PI1.6: mcp_available() logic
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Remove all AGENTS_ prefixed environment variables for clean test state."""
    env_vars_to_remove = [k for k in os.environ if k.startswith("AGENTS_")]
    original_values = {k: os.environ.pop(k) for k in env_vars_to_remove}
    yield
    # Restore original values
    os.environ.update(original_values)


@pytest.fixture
def a2a_enabled_env(clean_env: None) -> Generator[None, None, None]:
    """Set A2A enabled environment variables."""
    with patch.dict(
        os.environ,
        {
            "AGENTS_A2A_ENABLED": "true",
            "AGENTS_A2A_AGENT_CARD_ENABLED": "true",
        },
    ):
        yield


@pytest.fixture
def mcp_enabled_env(clean_env: None) -> Generator[None, None, None]:
    """Set MCP enabled environment variables."""
    with patch.dict(
        os.environ,
        {
            "AGENTS_MCP_ENABLED": "true",
            "AGENTS_MCP_SERVER_ENABLED": "true",
        },
    ):
        yield


@pytest.fixture
def all_flags_enabled_env(clean_env: None) -> Generator[None, None, None]:
    """Set all protocol feature flags to enabled."""
    with patch.dict(
        os.environ,
        {
            "AGENTS_A2A_ENABLED": "true",
            "AGENTS_A2A_AGENT_CARD_ENABLED": "true",
            "AGENTS_A2A_STREAMING_ENABLED": "true",
            "AGENTS_A2A_PUSH_NOTIFICATIONS": "true",
            "AGENTS_MCP_ENABLED": "true",
            "AGENTS_MCP_SERVER_ENABLED": "true",
            "AGENTS_MCP_CLIENT_ENABLED": "true",
            "AGENTS_MCP_SEMANTIC_SEARCH": "true",
            "AGENTS_MCP_TOOLBOX_NEO4J": "true",
            "AGENTS_MCP_TOOLBOX_REDIS": "true",
        },
    ):
        yield


# =============================================================================
# AC-PI1.1: ProtocolFeatureFlags loads from environment
# =============================================================================


class TestProtocolFeatureFlagsEnvironmentLoading:
    """Test that ProtocolFeatureFlags loads from environment variables."""

    def test_imports_successfully(self) -> None:
        """ProtocolFeatureFlags can be imported from src.config.feature_flags."""
        from src.config.feature_flags import ProtocolFeatureFlags

        assert ProtocolFeatureFlags is not None

    def test_loads_a2a_enabled_from_env(self, clean_env: None) -> None:
        """A2A enabled flag loads from AGENTS_A2A_ENABLED env var."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(os.environ, {"AGENTS_A2A_ENABLED": "true"}):
            flags = ProtocolFeatureFlags()
            assert flags.a2a_enabled is True

    def test_loads_mcp_enabled_from_env(self, clean_env: None) -> None:
        """MCP enabled flag loads from AGENTS_MCP_ENABLED env var."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(os.environ, {"AGENTS_MCP_ENABLED": "true"}):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_enabled is True

    def test_loads_all_a2a_sub_flags_from_env(
        self, all_flags_enabled_env: None
    ) -> None:
        """All A2A sub-flags load from respective environment variables."""
        from src.config.feature_flags import ProtocolFeatureFlags

        flags = ProtocolFeatureFlags()
        assert flags.a2a_agent_card_enabled is True
        assert flags.a2a_streaming_enabled is True
        assert flags.a2a_push_notifications is True

    def test_loads_all_mcp_sub_flags_from_env(
        self, all_flags_enabled_env: None
    ) -> None:
        """All MCP sub-flags load from respective environment variables."""
        from src.config.feature_flags import ProtocolFeatureFlags

        flags = ProtocolFeatureFlags()
        assert flags.mcp_server_enabled is True
        assert flags.mcp_client_enabled is True
        assert flags.mcp_semantic_search is True
        assert flags.mcp_toolbox_neo4j is True
        assert flags.mcp_toolbox_redis is True


# =============================================================================
# AC-PI1.2: All flags default to False
# =============================================================================


class TestProtocolFeatureFlagsDefaults:
    """Test that all flags default to False for safe rollout."""

    def test_all_a2a_flags_default_false(self, clean_env: None) -> None:
        """All A2A protocol flags default to False."""
        from src.config.feature_flags import ProtocolFeatureFlags

        flags = ProtocolFeatureFlags()
        assert flags.a2a_enabled is False
        assert flags.a2a_agent_card_enabled is False
        assert flags.a2a_streaming_enabled is False
        assert flags.a2a_push_notifications is False

    def test_all_mcp_flags_default_false(self, clean_env: None) -> None:
        """All MCP protocol flags default to False."""
        from src.config.feature_flags import ProtocolFeatureFlags

        flags = ProtocolFeatureFlags()
        assert flags.mcp_enabled is False
        assert flags.mcp_server_enabled is False
        assert flags.mcp_client_enabled is False
        assert flags.mcp_semantic_search is False
        assert flags.mcp_toolbox_neo4j is False
        assert flags.mcp_toolbox_redis is False


# =============================================================================
# AC-PI1.3: A2A master switch gates all A2A features
# =============================================================================


class TestA2AMasterSwitchGating:
    """Test that A2A master switch gates sub-features."""

    def test_a2a_master_disabled_sub_features_ignored(
        self, clean_env: None
    ) -> None:
        """When a2a_enabled=False, sub-features have no effect."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_A2A_ENABLED": "false",
                "AGENTS_A2A_AGENT_CARD_ENABLED": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            # Sub-feature is True but master is False
            assert flags.a2a_enabled is False
            assert flags.a2a_agent_card_enabled is True  # Loaded but gated
            assert flags.a2a_available() is False  # Master gates availability


# =============================================================================
# AC-PI1.4: MCP master switch gates all MCP features
# =============================================================================


class TestMCPMasterSwitchGating:
    """Test that MCP master switch gates sub-features."""

    def test_mcp_master_disabled_sub_features_ignored(
        self, clean_env: None
    ) -> None:
        """When mcp_enabled=False, sub-features have no effect."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_MCP_ENABLED": "false",
                "AGENTS_MCP_SERVER_ENABLED": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            # Sub-feature is True but master is False
            assert flags.mcp_enabled is False
            assert flags.mcp_server_enabled is True  # Loaded but gated
            assert flags.mcp_available() is False  # Master gates availability


# =============================================================================
# AC-PI1.5: a2a_available() returns True only when master + sub-feature enabled
# =============================================================================


class TestA2AAvailableMethod:
    """Test a2a_available() helper method logic."""

    def test_a2a_available_false_when_all_disabled(
        self, clean_env: None
    ) -> None:
        """a2a_available() returns False when all flags disabled."""
        from src.config.feature_flags import ProtocolFeatureFlags

        flags = ProtocolFeatureFlags()
        assert flags.a2a_available() is False

    def test_a2a_available_false_when_master_only(
        self, clean_env: None
    ) -> None:
        """a2a_available() returns False when only master enabled (no sub-features)."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(os.environ, {"AGENTS_A2A_ENABLED": "true"}):
            flags = ProtocolFeatureFlags()
            assert flags.a2a_available() is False

    def test_a2a_available_true_with_agent_card(
        self, clean_env: None
    ) -> None:
        """a2a_available() returns True when master + agent_card enabled."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_A2A_ENABLED": "true",
                "AGENTS_A2A_AGENT_CARD_ENABLED": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.a2a_available() is True

    def test_a2a_available_true_with_streaming(
        self, clean_env: None
    ) -> None:
        """a2a_available() returns True when master + streaming enabled."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_A2A_ENABLED": "true",
                "AGENTS_A2A_STREAMING_ENABLED": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.a2a_available() is True

    def test_a2a_available_true_with_push_notifications(
        self, clean_env: None
    ) -> None:
        """a2a_available() returns True when master + push_notifications enabled."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_A2A_ENABLED": "true",
                "AGENTS_A2A_PUSH_NOTIFICATIONS": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.a2a_available() is True


# =============================================================================
# AC-PI1.6: mcp_available() returns True only when master + sub-feature enabled
# =============================================================================


class TestMCPAvailableMethod:
    """Test mcp_available() helper method logic."""

    def test_mcp_available_false_when_all_disabled(
        self, clean_env: None
    ) -> None:
        """mcp_available() returns False when all flags disabled."""
        from src.config.feature_flags import ProtocolFeatureFlags

        flags = ProtocolFeatureFlags()
        assert flags.mcp_available() is False

    def test_mcp_available_false_when_master_only(
        self, clean_env: None
    ) -> None:
        """mcp_available() returns False when only master enabled (no sub-features)."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(os.environ, {"AGENTS_MCP_ENABLED": "true"}):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_available() is False

    def test_mcp_available_true_with_server(
        self, clean_env: None
    ) -> None:
        """mcp_available() returns True when master + server enabled."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_MCP_ENABLED": "true",
                "AGENTS_MCP_SERVER_ENABLED": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_available() is True

    def test_mcp_available_true_with_client(
        self, clean_env: None
    ) -> None:
        """mcp_available() returns True when master + client enabled."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_MCP_ENABLED": "true",
                "AGENTS_MCP_CLIENT_ENABLED": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_available() is True

    def test_mcp_available_true_with_semantic_search(
        self, clean_env: None
    ) -> None:
        """mcp_available() returns True when master + semantic_search enabled."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_MCP_ENABLED": "true",
                "AGENTS_MCP_SEMANTIC_SEARCH": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_available() is True

    def test_mcp_available_true_with_neo4j_toolbox(
        self, clean_env: None
    ) -> None:
        """mcp_available() returns True when master + neo4j toolbox enabled."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_MCP_ENABLED": "true",
                "AGENTS_MCP_TOOLBOX_NEO4J": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_available() is True

    def test_mcp_available_true_with_redis_toolbox(
        self, clean_env: None
    ) -> None:
        """mcp_available() returns True when master + redis toolbox enabled."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_MCP_ENABLED": "true",
                "AGENTS_MCP_TOOLBOX_REDIS": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_available() is True


# =============================================================================
# Additional Helper Method Tests
# =============================================================================


class TestIndividualFeatureChecks:
    """Test individual feature availability checks."""

    def test_a2a_agent_card_available_property(
        self, a2a_enabled_env: None
    ) -> None:
        """a2a_agent_card_available checks master + specific flag."""
        from src.config.feature_flags import ProtocolFeatureFlags

        flags = ProtocolFeatureFlags()
        assert flags.a2a_agent_card_available is True

    def test_a2a_agent_card_available_false_without_master(
        self, clean_env: None
    ) -> None:
        """a2a_agent_card_available returns False without master."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {"AGENTS_A2A_AGENT_CARD_ENABLED": "true"},
        ):
            flags = ProtocolFeatureFlags()
            assert flags.a2a_agent_card_available is False

    def test_a2a_streaming_available_property(
        self, clean_env: None
    ) -> None:
        """a2a_streaming_available checks master + specific flag."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_A2A_ENABLED": "true",
                "AGENTS_A2A_STREAMING_ENABLED": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.a2a_streaming_available is True

    def test_mcp_server_available_property(
        self, mcp_enabled_env: None
    ) -> None:
        """mcp_server_available checks master + specific flag."""
        from src.config.feature_flags import ProtocolFeatureFlags

        flags = ProtocolFeatureFlags()
        assert flags.mcp_server_available is True

    def test_mcp_server_available_false_without_master(
        self, clean_env: None
    ) -> None:
        """mcp_server_available returns False without master."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {"AGENTS_MCP_SERVER_ENABLED": "true"},
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_server_available is False

    def test_a2a_push_notifications_available_property(
        self, clean_env: None
    ) -> None:
        """a2a_push_notifications_available checks master + specific flag."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_A2A_ENABLED": "true",
                "AGENTS_A2A_PUSH_NOTIFICATIONS": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.a2a_push_notifications_available is True

    def test_a2a_push_notifications_available_false_without_master(
        self, clean_env: None
    ) -> None:
        """a2a_push_notifications_available returns False without master."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {"AGENTS_A2A_PUSH_NOTIFICATIONS": "true"},
        ):
            flags = ProtocolFeatureFlags()
            assert flags.a2a_push_notifications_available is False

    def test_mcp_client_available_property(
        self, clean_env: None
    ) -> None:
        """mcp_client_available checks master + specific flag."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_MCP_ENABLED": "true",
                "AGENTS_MCP_CLIENT_ENABLED": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_client_available is True

    def test_mcp_client_available_false_without_master(
        self, clean_env: None
    ) -> None:
        """mcp_client_available returns False without master."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {"AGENTS_MCP_CLIENT_ENABLED": "true"},
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_client_available is False

    def test_mcp_semantic_search_available_property(
        self, clean_env: None
    ) -> None:
        """mcp_semantic_search_available checks master + specific flag."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_MCP_ENABLED": "true",
                "AGENTS_MCP_SEMANTIC_SEARCH": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_semantic_search_available is True

    def test_mcp_semantic_search_available_false_without_master(
        self, clean_env: None
    ) -> None:
        """mcp_semantic_search_available returns False without master."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {"AGENTS_MCP_SEMANTIC_SEARCH": "true"},
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_semantic_search_available is False

    def test_mcp_toolbox_neo4j_available_property(
        self, clean_env: None
    ) -> None:
        """mcp_toolbox_neo4j_available checks master + specific flag."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_MCP_ENABLED": "true",
                "AGENTS_MCP_TOOLBOX_NEO4J": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_toolbox_neo4j_available is True

    def test_mcp_toolbox_neo4j_available_false_without_master(
        self, clean_env: None
    ) -> None:
        """mcp_toolbox_neo4j_available returns False without master."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {"AGENTS_MCP_TOOLBOX_NEO4J": "true"},
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_toolbox_neo4j_available is False

    def test_mcp_toolbox_redis_available_property(
        self, clean_env: None
    ) -> None:
        """mcp_toolbox_redis_available checks master + specific flag."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {
                "AGENTS_MCP_ENABLED": "true",
                "AGENTS_MCP_TOOLBOX_REDIS": "true",
            },
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_toolbox_redis_available is True

    def test_mcp_toolbox_redis_available_false_without_master(
        self, clean_env: None
    ) -> None:
        """mcp_toolbox_redis_available returns False without master."""
        from src.config.feature_flags import ProtocolFeatureFlags

        with patch.dict(
            os.environ,
            {"AGENTS_MCP_TOOLBOX_REDIS": "true"},
        ):
            flags = ProtocolFeatureFlags()
            assert flags.mcp_toolbox_redis_available is False


# =============================================================================
# Model Configuration Tests
# =============================================================================


class TestProtocolFeatureFlagsConfiguration:
    """Test ProtocolFeatureFlags pydantic configuration."""

    def test_uses_agents_env_prefix(self, clean_env: None) -> None:
        """ProtocolFeatureFlags uses AGENTS_ environment variable prefix."""
        from src.config.feature_flags import ProtocolFeatureFlags

        # Access model config to verify env_prefix
        assert ProtocolFeatureFlags.model_config.get("env_prefix") == "AGENTS_"

    def test_is_pydantic_settings_model(self, clean_env: None) -> None:
        """ProtocolFeatureFlags inherits from BaseSettings."""
        from pydantic_settings import BaseSettings

        from src.config.feature_flags import ProtocolFeatureFlags

        assert issubclass(ProtocolFeatureFlags, BaseSettings)


# =============================================================================
# AC-PI1.7: Feature flags injectable via FastAPI dependency
# =============================================================================


class TestFeatureFlagsDependency:
    """Test FastAPI dependency injection for feature flags."""

    def test_get_feature_flags_importable(self, clean_env: None) -> None:
        """get_feature_flags can be imported from src.config.feature_flags."""
        from src.config.feature_flags import get_feature_flags

        assert callable(get_feature_flags)

    def test_get_feature_flags_returns_protocol_feature_flags(
        self, clean_env: None
    ) -> None:
        """get_feature_flags returns a ProtocolFeatureFlags instance."""
        from src.config.feature_flags import (
            ProtocolFeatureFlags,
            get_feature_flags,
        )

        flags = get_feature_flags()
        assert isinstance(flags, ProtocolFeatureFlags)

    def test_get_feature_flags_cached(self, clean_env: None) -> None:
        """get_feature_flags returns cached singleton instance."""
        from src.config.feature_flags import get_feature_flags

        flags1 = get_feature_flags()
        flags2 = get_feature_flags()
        assert flags1 is flags2

    def test_get_feature_flags_reads_from_env(self, clean_env: None) -> None:
        """get_feature_flags respects environment variables."""
        from src.config.feature_flags import get_feature_flags

        # Clear any cached instance first
        get_feature_flags.cache_clear()

        with patch.dict(
            os.environ,
            {
                "AGENTS_A2A_ENABLED": "true",
                "AGENTS_MCP_ENABLED": "true",
            },
        ):
            flags = get_feature_flags()
            assert flags.a2a_enabled is True
            assert flags.mcp_enabled is True

        # Clear cache for next test
        get_feature_flags.cache_clear()

    def test_fastapi_depends_pattern(self, clean_env: None) -> None:
        """get_feature_flags works with FastAPI Depends pattern."""
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient

        from src.config.feature_flags import (
            ProtocolFeatureFlags,
            get_feature_flags,
        )

        # Clear cache
        get_feature_flags.cache_clear()

        app = FastAPI()

        @app.get("/test-flags")
        async def test_flags(
            flags: ProtocolFeatureFlags = Depends(get_feature_flags),
        ) -> dict[str, bool]:
            return {
                "a2a_enabled": flags.a2a_enabled,
                "mcp_enabled": flags.mcp_enabled,
            }

        client = TestClient(app)
        response = client.get("/test-flags")
        assert response.status_code == 200
        data = response.json()
        assert data["a2a_enabled"] is False  # Default
        assert data["mcp_enabled"] is False  # Default

        # Clear cache
        get_feature_flags.cache_clear()

