"""Unit tests for src/cache/handoff module.

Tests HandoffCache for temp: prefix (pipeline-local) state management.

Reference: WBS-AGT3 AC-3.4
"""

import asyncio
import pytest
from datetime import datetime, timezone

from src.cache.handoff import HandoffCache, HandoffEntry
from src.cache.state import STATE_PREFIX_TEMP


class TestHandoffEntry:
    """Tests for HandoffEntry dataclass."""
    
    def test_entry_creation(self) -> None:
        """Test entry can be created with value."""
        entry = HandoffEntry(key="test_key", value={"data": "value"})
        
        assert entry.key == "test_key"
        assert entry.value == {"data": "value"}
        assert entry.source_stage == ""
        assert entry.metadata == {}
    
    def test_entry_with_metadata(self) -> None:
        """Test entry with metadata."""
        entry = HandoffEntry(
            key="key",
            value="value",
            source_stage="stage_1",
            metadata={"custom": "data"},
        )
        
        assert entry.source_stage == "stage_1"
        assert entry.metadata == {"custom": "data"}
    
    def test_entry_metadata_not_shared(self) -> None:
        """Test metadata is not shared between instances (AP-1.5)."""
        entry1 = HandoffEntry(key="k1", value="v1")
        entry2 = HandoffEntry(key="k2", value="v2")
        
        entry1.metadata["key"] = "value"
        assert "key" not in entry2.metadata
    
    def test_entry_created_at_default(self) -> None:
        """Test created_at defaults to current UTC time."""
        before = datetime.now(timezone.utc)
        entry = HandoffEntry(key="key", value="value")
        after = datetime.now(timezone.utc)
        
        assert before <= entry.created_at <= after


class TestHandoffCache:
    """Tests for HandoffCache class (AC-3.4)."""
    
    @pytest.fixture
    def cache(self) -> HandoffCache:
        """Create a HandoffCache instance for testing."""
        return HandoffCache(pipeline_id="test_pipeline_123")
    
    def test_cache_creation(self, cache: HandoffCache) -> None:
        """Test cache can be created."""
        assert cache.pipeline_id == "test_pipeline_123"
        assert cache.prefix == STATE_PREFIX_TEMP
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache: HandoffCache) -> None:
        """Test setting and getting values."""
        await cache.set("stage_output", {"result": "success"})
        
        value = await cache.get("stage_output")
        assert value == {"result": "success"}
    
    @pytest.mark.asyncio
    async def test_get_missing_returns_default(self, cache: HandoffCache) -> None:
        """Test getting missing key returns default."""
        value = await cache.get("nonexistent")
        assert value is None
        
        value = await cache.get("nonexistent", default="fallback")
        assert value == "fallback"
    
    @pytest.mark.asyncio
    async def test_set_returns_full_key(self, cache: HandoffCache) -> None:
        """Test set returns full cache key."""
        full_key = await cache.set("my_key", "my_value")
        
        assert full_key == "temp:test_pipeline_123:my_key"
    
    @pytest.mark.asyncio
    async def test_set_with_source_stage(self, cache: HandoffCache) -> None:
        """Test setting with source stage metadata."""
        await cache.set("output", {"data": "test"}, source_stage="extract_structure")
        
        entry = await cache.get_entry("output")
        assert entry is not None
        assert entry.source_stage == "extract_structure"
    
    @pytest.mark.asyncio
    async def test_get_entry_returns_full_metadata(self, cache: HandoffCache) -> None:
        """Test get_entry returns HandoffEntry with metadata."""
        await cache.set(
            "test",
            "value",
            source_stage="stage_1",
            metadata={"custom": "data"},
        )
        
        entry = await cache.get_entry("test")
        assert entry is not None
        assert entry.value == "value"
        assert entry.source_stage == "stage_1"
        assert entry.metadata == {"custom": "data"}
    
    @pytest.mark.asyncio
    async def test_get_entry_missing_returns_none(self, cache: HandoffCache) -> None:
        """Test get_entry returns None for missing key."""
        entry = await cache.get_entry("nonexistent")
        assert entry is None
    
    @pytest.mark.asyncio
    async def test_delete_existing_key(self, cache: HandoffCache) -> None:
        """Test deleting existing key returns True."""
        await cache.set("to_delete", "value")
        
        result = await cache.delete("to_delete")
        assert result is True
        
        value = await cache.get("to_delete")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, cache: HandoffCache) -> None:
        """Test deleting nonexistent key returns False."""
        result = await cache.delete("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_exists_true(self, cache: HandoffCache) -> None:
        """Test exists returns True for existing key."""
        await cache.set("exists", "value")
        
        assert await cache.exists("exists") is True
    
    @pytest.mark.asyncio
    async def test_exists_false(self, cache: HandoffCache) -> None:
        """Test exists returns False for missing key."""
        assert await cache.exists("nonexistent") is False
    
    @pytest.mark.asyncio
    async def test_keys_returns_short_keys(self, cache: HandoffCache) -> None:
        """Test keys returns keys without prefix."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        keys = await cache.keys()
        assert set(keys) == {"key1", "key2"}
    
    @pytest.mark.asyncio
    async def test_clear_removes_all(self, cache: HandoffCache) -> None:
        """Test clear removes all entries."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        count = await cache.clear()
        assert count == 2
        
        assert await cache.size() == 0
    
    @pytest.mark.asyncio
    async def test_size_returns_count(self, cache: HandoffCache) -> None:
        """Test size returns number of entries."""
        assert await cache.size() == 0
        
        await cache.set("key1", "value1")
        assert await cache.size() == 1
        
        await cache.set("key2", "value2")
        assert await cache.size() == 2
    
    @pytest.mark.asyncio
    async def test_uses_asyncio_lock(self, cache: HandoffCache) -> None:
        """Test cache uses asyncio.Lock (AP-10.1).
        
        Exit Criteria: HandoffCache uses asyncio.Lock
        """
        assert isinstance(cache._lock, asyncio.Lock)
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, cache: HandoffCache) -> None:
        """Test cache is thread-safe for concurrent access."""
        async def writer(key: str, value: str) -> None:
            await cache.set(key, value)
            await asyncio.sleep(0.001)
        
        # Run multiple concurrent writes
        tasks = [writer(f"key_{i}", f"value_{i}") for i in range(10)]
        await asyncio.gather(*tasks)
        
        # Verify all writes succeeded
        assert await cache.size() == 10
        for i in range(10):
            value = await cache.get(f"key_{i}")
            assert value == f"value_{i}"
    
    @pytest.mark.asyncio
    async def test_get_all_entries(self, cache: HandoffCache) -> None:
        """Test get_all_entries returns copy of store."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        entries = await cache.get_all_entries()
        assert len(entries) == 2
    
    def test_repr(self, cache: HandoffCache) -> None:
        """Test repr shows pipeline_id and count."""
        repr_str = repr(cache)
        assert "test_pipeline_123" in repr_str
        assert "HandoffCache" in repr_str
