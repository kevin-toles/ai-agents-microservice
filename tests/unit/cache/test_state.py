"""Unit tests for src/cache/state module.

Tests ADK state prefix constants and cache key builder.

Reference: WBS-AGT3 AC-3.1, AC-3.2
"""

import pytest

from src.cache.state import (
    StatePrefix,
    STATE_PREFIX_TEMP,
    STATE_PREFIX_USER,
    STATE_PREFIX_APP,
    build_cache_key,
    parse_cache_key,
    get_cache_tier,
)


class TestStatePrefixConstants:
    """Tests for state prefix constants (AC-3.1)."""
    
    def test_temp_prefix_value(self) -> None:
        """Test temp: prefix constant value."""
        assert STATE_PREFIX_TEMP == "temp:"
    
    def test_user_prefix_value(self) -> None:
        """Test user: prefix constant value."""
        assert STATE_PREFIX_USER == "user:"
    
    def test_app_prefix_value(self) -> None:
        """Test app: prefix constant value."""
        assert STATE_PREFIX_APP == "app:"
    
    def test_prefixes_end_with_colon(self) -> None:
        """Test all prefixes end with colon for key building."""
        for prefix in (STATE_PREFIX_TEMP, STATE_PREFIX_USER, STATE_PREFIX_APP):
            assert prefix.endswith(":")
    
    def test_state_prefix_enum_values(self) -> None:
        """Test StatePrefix enum has correct values."""
        assert StatePrefix.TEMP.value == "temp:"
        assert StatePrefix.USER.value == "user:"
        assert StatePrefix.APP.value == "app:"


class TestBuildCacheKey:
    """Tests for build_cache_key function (AC-3.2)."""
    
    def test_build_temp_key(self) -> None:
        """Test building temp: prefix key."""
        key = build_cache_key(STATE_PREFIX_TEMP, "extract_structure", "ch1")
        assert key == "temp:extract_structure:ch1"
    
    def test_build_user_key(self) -> None:
        """Test building user: prefix key."""
        key = build_cache_key(STATE_PREFIX_USER, "summarize", "user_123")
        assert key == "user:summarize:user_123"
    
    def test_build_app_key(self) -> None:
        """Test building app: prefix key."""
        key = build_cache_key(STATE_PREFIX_APP, "artifact", "model_v1")
        assert key == "app:artifact:model_v1"
    
    def test_build_key_from_wbs_example(self) -> None:
        """Test exact example from WBS exit criteria."""
        # Exit Criteria: build_cache_key("temp:", "extract_structure", "ch1") → "temp:extract_structure:ch1"
        result = build_cache_key("temp:", "extract_structure", "ch1")
        assert result == "temp:extract_structure:ch1"
    
    def test_key_with_colons_in_value(self) -> None:
        """Test key value can contain colons."""
        key = build_cache_key(STATE_PREFIX_USER, "session", "user:123:abc")
        assert key == "user:session:user:123:abc"
    
    def test_invalid_prefix_raises_error(self) -> None:
        """Test invalid prefix raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            build_cache_key("invalid:", "namespace", "key")
        assert "Invalid prefix" in str(exc_info.value)
    
    def test_empty_namespace_raises_error(self) -> None:
        """Test empty namespace raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            build_cache_key(STATE_PREFIX_TEMP, "", "key")
        assert "namespace cannot be empty" in str(exc_info.value)
    
    def test_empty_key_raises_error(self) -> None:
        """Test empty key raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            build_cache_key(STATE_PREFIX_TEMP, "namespace", "")
        assert "key cannot be empty" in str(exc_info.value)


class TestParseCacheKey:
    """Tests for parse_cache_key function."""
    
    def test_parse_temp_key(self) -> None:
        """Test parsing temp: prefix key."""
        prefix, namespace, key = parse_cache_key("temp:extract_structure:chapter_1")
        assert prefix == "temp:"
        assert namespace == "extract_structure"
        assert key == "chapter_1"
    
    def test_parse_user_key(self) -> None:
        """Test parsing user: prefix key."""
        prefix, namespace, key = parse_cache_key("user:session:user_123")
        assert prefix == "user:"
        assert namespace == "session"
        assert key == "user_123"
    
    def test_parse_app_key(self) -> None:
        """Test parsing app: prefix key."""
        prefix, namespace, key = parse_cache_key("app:artifact:model_v1")
        assert prefix == "app:"
        assert namespace == "artifact"
        assert key == "model_v1"
    
    def test_parse_key_with_colons(self) -> None:
        """Test parsing key with colons in value part."""
        prefix, namespace, key = parse_cache_key("user:session:user:123:abc")
        assert prefix == "user:"
        assert namespace == "session"
        assert key == "user:123:abc"  # Colons preserved in key
    
    def test_invalid_prefix_raises_error(self) -> None:
        """Test invalid prefix raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_cache_key("invalid:namespace:key")
        assert "Invalid cache key prefix" in str(exc_info.value)
    
    def test_missing_key_part_raises_error(self) -> None:
        """Test missing key part raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_cache_key("temp:namespace_only")
        assert "Invalid cache key format" in str(exc_info.value)


class TestGetCacheTier:
    """Tests for get_cache_tier function."""
    
    def test_temp_tier_is_handoff(self) -> None:
        """Test temp: prefix maps to handoff_cache."""
        assert get_cache_tier(STATE_PREFIX_TEMP) == "handoff_cache"
    
    def test_user_tier_is_compression(self) -> None:
        """Test user: prefix maps to compression_cache."""
        assert get_cache_tier(STATE_PREFIX_USER) == "compression_cache"
    
    def test_app_tier_is_artifact(self) -> None:
        """Test app: prefix maps to artifact_store."""
        assert get_cache_tier(STATE_PREFIX_APP) == "artifact_store"
    
    def test_unknown_prefix_returns_unknown(self) -> None:
        """Test unknown prefix returns 'unknown'."""
        assert get_cache_tier("invalid:") == "unknown"
