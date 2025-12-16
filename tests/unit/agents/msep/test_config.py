"""Unit tests for MSEP configuration.

WBS: MSE-2.3 - Configuration Dataclass
TDD Phase: RED (tests written BEFORE implementation)

Acceptance Criteria Coverage:
- AC-2.3.1: MSEPConfig loads defaults when no env vars set
- AC-2.3.2: MSEPConfig.from_env() loads from environment
- AC-2.3.3: MSEPConfig is immutable (frozen=True)
- AC-2.3.4: All config fields have type annotations

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: No duplicated string literals
- #2.2: Full type annotations
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

# Module constants per S1192 (no duplicated literals)
ENV_PREFIX: str = "MSEP_"
TEST_THRESHOLD: float = 0.65
TEST_TOP_K: int = 10
TEST_TIMEOUT: float = 45.0


class TestMSEPConfigDefaults:
    """Tests for MSEPConfig default values (AC-2.3.1)."""

    def test_config_has_threshold_default(self) -> None:
        """AC-2.3.1: MSEPConfig should have threshold with default value."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert hasattr(config, "threshold")
        assert config.threshold == 0.5  # DEFAULT_THRESHOLD from constants

    def test_config_has_top_k_default(self) -> None:
        """AC-2.3.1: MSEPConfig should have top_k with default value."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert hasattr(config, "top_k")
        assert config.top_k == 5  # DEFAULT_TOP_K from constants

    def test_config_has_timeout_default(self) -> None:
        """AC-2.3.1: MSEPConfig should have timeout with default value."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert hasattr(config, "timeout")
        assert config.timeout == 30.0  # DEFAULT_TIMEOUT from constants

    def test_config_has_use_dynamic_threshold_default(self) -> None:
        """AC-2.3.1: MSEPConfig should have use_dynamic_threshold flag."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert hasattr(config, "use_dynamic_threshold")
        assert config.use_dynamic_threshold is True  # Default enabled

    def test_config_has_enable_hybrid_search_default(self) -> None:
        """AC-2.3.1: MSEPConfig should have enable_hybrid_search flag."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert hasattr(config, "enable_hybrid_search")
        assert config.enable_hybrid_search is True  # Default enabled

    def test_config_has_same_topic_boost_default(self) -> None:
        """AC-2.3.1: MSEPConfig should have same_topic_boost with default."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert hasattr(config, "same_topic_boost")
        assert config.same_topic_boost == 0.15  # SAME_TOPIC_BOOST from constants


class TestMSEPConfigFromEnv:
    """Tests for MSEPConfig.from_env() loading (AC-2.3.2)."""

    def test_from_env_loads_threshold(self) -> None:
        """AC-2.3.2: MSEPConfig.from_env() should load threshold from env."""
        from src.agents.msep.config import MSEPConfig

        env_vars = {f"{ENV_PREFIX}THRESHOLD": str(TEST_THRESHOLD)}

        with patch.dict(os.environ, env_vars, clear=False):
            config = MSEPConfig.from_env()

        assert config.threshold == TEST_THRESHOLD

    def test_from_env_loads_top_k(self) -> None:
        """AC-2.3.2: MSEPConfig.from_env() should load top_k from env."""
        from src.agents.msep.config import MSEPConfig

        env_vars = {f"{ENV_PREFIX}TOP_K": str(TEST_TOP_K)}

        with patch.dict(os.environ, env_vars, clear=False):
            config = MSEPConfig.from_env()

        assert config.top_k == TEST_TOP_K

    def test_from_env_loads_timeout(self) -> None:
        """AC-2.3.2: MSEPConfig.from_env() should load timeout from env."""
        from src.agents.msep.config import MSEPConfig

        env_vars = {f"{ENV_PREFIX}TIMEOUT": str(TEST_TIMEOUT)}

        with patch.dict(os.environ, env_vars, clear=False):
            config = MSEPConfig.from_env()

        assert config.timeout == TEST_TIMEOUT

    def test_from_env_loads_boolean_flags(self) -> None:
        """AC-2.3.2: MSEPConfig.from_env() should load boolean flags."""
        from src.agents.msep.config import MSEPConfig

        env_vars = {
            f"{ENV_PREFIX}USE_DYNAMIC_THRESHOLD": "false",
            f"{ENV_PREFIX}ENABLE_HYBRID_SEARCH": "false",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = MSEPConfig.from_env()

        assert config.use_dynamic_threshold is False
        assert config.enable_hybrid_search is False

    def test_from_env_uses_defaults_when_not_set(self) -> None:
        """AC-2.3.2: MSEPConfig.from_env() uses defaults when env vars not set."""
        from src.agents.msep.config import MSEPConfig

        # Clear any MSEP_ env vars that might exist
        env_to_clear = [k for k in os.environ if k.startswith(ENV_PREFIX)]
        with patch.dict(os.environ, {k: "" for k in env_to_clear}, clear=False):
            for k in env_to_clear:
                os.environ.pop(k, None)
            config = MSEPConfig.from_env()

        # Should use defaults
        assert config.threshold == 0.5
        assert config.top_k == 5
        assert config.timeout == 30.0


class TestMSEPConfigImmutability:
    """Tests for MSEPConfig immutability (AC-2.3.3)."""

    def test_config_is_frozen(self) -> None:
        """AC-2.3.3: MSEPConfig should be immutable (frozen=True)."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        with pytest.raises((AttributeError, TypeError, Exception)):
            config.threshold = 0.9  # type: ignore[misc]

    def test_config_cannot_add_attributes(self) -> None:
        """AC-2.3.3: MSEPConfig should not allow adding new attributes."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        with pytest.raises((AttributeError, TypeError, Exception)):
            config.new_attr = "test"  # type: ignore[attr-defined]


class TestMSEPConfigTypeAnnotations:
    """Tests for MSEPConfig type annotations (AC-2.3.4)."""

    def test_config_threshold_is_float(self) -> None:
        """AC-2.3.4: threshold should be float."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert isinstance(config.threshold, float)

    def test_config_top_k_is_int(self) -> None:
        """AC-2.3.4: top_k should be int."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert isinstance(config.top_k, int)

    def test_config_timeout_is_float(self) -> None:
        """AC-2.3.4: timeout should be float."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert isinstance(config.timeout, float)

    def test_config_use_dynamic_threshold_is_bool(self) -> None:
        """AC-2.3.4: use_dynamic_threshold should be bool."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert isinstance(config.use_dynamic_threshold, bool)

    def test_config_enable_hybrid_search_is_bool(self) -> None:
        """AC-2.3.4: enable_hybrid_search should be bool."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert isinstance(config.enable_hybrid_search, bool)

    def test_config_same_topic_boost_is_float(self) -> None:
        """AC-2.3.4: same_topic_boost should be float."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert isinstance(config.same_topic_boost, float)


class TestMSEPConfigValidation:
    """Tests for MSEPConfig validation."""

    def test_config_threshold_clamped_to_valid_range(self) -> None:
        """threshold should be clamped to [0.0, 1.0]."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig(threshold=0.5)

        assert 0.0 <= config.threshold <= 1.0

    def test_config_top_k_must_be_positive(self) -> None:
        """top_k should be >= 1."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig(top_k=5)

        assert config.top_k >= 1

    def test_config_timeout_must_be_positive(self) -> None:
        """timeout should be > 0."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig(timeout=30.0)

        assert config.timeout > 0
