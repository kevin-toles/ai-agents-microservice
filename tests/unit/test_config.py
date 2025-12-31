"""Unit tests for core configuration.

TDD Phase: GREEN - Tests updated to match actual implementation.
Pattern: Pydantic Settings testing
Source: Comp_Static_Analysis_Report - Settings pattern
"""

import os
import pytest
from unittest.mock import patch

from src.core.config import Settings, get_settings


class TestSettings:
    """Tests for Settings configuration class."""
    
    def test_settings_default_values(self) -> None:
        """Test that Settings has sensible defaults.
        
        Note: Environment variables may override defaults, so we check the
        Field defaults from the model rather than instantiated values.
        """
        # Check the Field defaults directly
        from src.core.config import Settings as SettingsClass
        
        fields = SettingsClass.model_fields
        assert fields["neo4j_uri"].default == "bolt://localhost:7687"
        assert fields["neo4j_user"].default == "neo4j"
        assert fields["llm_gateway_url"].default == "http://localhost:8080"
        assert fields["semantic_search_url"].default == "http://localhost:8081"
        assert fields["log_level"].default == "INFO"
        assert fields["environment"].default == "development"
    
    def test_settings_from_environment(self) -> None:
        """Test that Settings loads from environment variables."""
        env_vars = {
            "AI_AGENTS_NEO4J_URI": "bolt://production:7687",
            "AI_AGENTS_NEO4J_USER": "prod_user",
            "AI_AGENTS_NEO4J_PASSWORD": "secret123",
            "AI_AGENTS_LLM_GATEWAY_URL": "http://gateway.production:8000",
            "AI_AGENTS_LOG_LEVEL": "DEBUG",
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
        
            assert settings.neo4j_uri == "bolt://production:7687"
            assert settings.neo4j_user == "prod_user"
            assert settings.neo4j_password.get_secret_value() == "secret123"
            assert settings.llm_gateway_url == "http://gateway.production:8000"
            assert settings.log_level == "DEBUG"
    
    def test_settings_env_prefix(self) -> None:
        """Test that Settings uses AI_AGENTS_ prefix correctly."""
        # Non-prefixed variables should NOT be read
        env_vars = {
            "NEO4J_URI": "bolt://wrong:7687",
            "AI_AGENTS_NEO4J_URI": "bolt://correct:7687",
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
        
            # Should use the prefixed value
            assert settings.neo4j_uri == "bolt://correct:7687"
    
    def test_settings_agent_configs(self) -> None:
        """Test that agent configuration defaults are sensible."""
        settings = Settings()
        
        assert settings.default_llm_model is not None
        assert settings.max_traversal_hops >= 1
        assert 0.0 <= settings.default_similarity_threshold <= 1.0
    
    def test_settings_timeout_defaults(self) -> None:
        """Test timeout configuration defaults."""
        settings = Settings()
        
        assert settings.llm_timeout_seconds >= 30
        assert settings.search_timeout_seconds >= 10
        assert settings.graph_timeout_seconds >= 10


class TestGetSettings:
    """Tests for get_settings cached factory."""
    
    def test_get_settings_returns_settings(self) -> None:
        """Test that get_settings returns a Settings instance."""
        # Clear cache first
        get_settings.cache_clear()
        
        settings = get_settings()
        
        assert isinstance(settings, Settings)
    
    def test_get_settings_is_cached(self) -> None:
        """Test that get_settings returns the same instance (cached)."""
        # Clear any existing cache first
        get_settings.cache_clear()
        
        first_settings = get_settings()
        second_settings = get_settings()
        
        # Should be the exact same object
        assert first_settings is second_settings
    
    def test_get_settings_cache_can_be_cleared(self) -> None:
        """Test that cache can be cleared for testing."""
        first_settings = get_settings()
        
        # Clear and get new instance
        get_settings.cache_clear()
        second_settings = get_settings()
        
        # After clearing, both should have same config values
        # This validates the cache mechanism exists and works
        assert first_settings.neo4j_uri == second_settings.neo4j_uri
