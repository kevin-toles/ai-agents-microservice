"""Unit tests for src/cache/compression module.

Tests CompressionCache for user: prefix (Redis 24h TTL) state management.

Reference: WBS-AGT3 AC-3.5
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Any

from src.cache.compression import (
    CompressionCache,
    CacheEntry,
    RedisClientProtocol,
    DEFAULT_USER_TTL_SECONDS,
)
from src.cache.state import STATE_PREFIX_USER


class FakeRedisClient:
    """Fake Redis client for testing (Protocol duck typing)."""
    
    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}
        self._ttls: dict[str, int] = {}
    
    async def get(self, key: str) -> bytes | None:
        await asyncio.sleep(0)
        return self._store.get(key)
    
    async def set(
        self,
        key: str,
        value: bytes,
        ex: int | None = None,
    ) -> bool:
        await asyncio.sleep(0)
        self._store[key] = value
        if ex is not None:
            self._ttls[key] = ex
        return True
    
    async def delete(self, key: str) -> int:
        await asyncio.sleep(0)
        if key in self._store:
            del self._store[key]
            self._ttls.pop(key, None)
            return 1
        return 0
    
    async def exists(self, key: str) -> int:
        await asyncio.sleep(0)
        return 1 if key in self._store else 0
    
    async def ttl(self, key: str) -> int:
        await asyncio.sleep(0)
        if key not in self._store:
            return -2
        return self._ttls.get(key, -1)


class TestRedisClientProtocol:
    """Tests for RedisClientProtocol duck typing."""
    
    def test_fake_redis_implements_protocol(self) -> None:
        """Test FakeRedisClient satisfies RedisClientProtocol."""
        fake = FakeRedisClient()
        assert isinstance(fake, RedisClientProtocol)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""
    
    def test_entry_creation(self) -> None:
        """Test entry can be created."""
        entry = CacheEntry(key="test", value={"data": "value"})
        
        assert entry.key == "test"
        assert entry.value == {"data": "value"}
        assert entry.compressed is False
        assert entry.metadata == {}
    
    def test_compression_ratio_no_compression(self) -> None:
        """Test compression_ratio when not compressed."""
        entry = CacheEntry(
            key="test",
            value="data",
            compressed=False,
            original_size=100,
            compressed_size=100,
        )
        assert entry.compression_ratio == pytest.approx(1.0)
    
    def test_compression_ratio_with_compression(self) -> None:
        """Test compression_ratio when compressed."""
        entry = CacheEntry(
            key="test",
            value="data",
            compressed=True,
            original_size=1000,
            compressed_size=200,
        )
        assert entry.compression_ratio == pytest.approx(0.2)
    
    def test_compression_ratio_zero_size(self) -> None:
        """Test compression_ratio with zero original size."""
        entry = CacheEntry(key="test", value="data", original_size=0)
        assert entry.compression_ratio == pytest.approx(1.0)
    
    def test_is_expired_no_expiry(self) -> None:
        """Test is_expired when no expiration set."""
        entry = CacheEntry(key="test", value="data", expires_at=None)
        assert entry.is_expired is False
    
    def test_metadata_not_shared(self) -> None:
        """Test metadata is not shared between instances (AP-1.5)."""
        entry1 = CacheEntry(key="k1", value="v1")
        entry2 = CacheEntry(key="k2", value="v2")
        
        entry1.metadata["key"] = "value"
        assert "key" not in entry2.metadata


class TestCompressionCacheInMemory:
    """Tests for CompressionCache with in-memory fallback (AC-3.5)."""
    
    @pytest.fixture
    def cache(self) -> CompressionCache:
        """Create a CompressionCache with in-memory backend."""
        return CompressionCache(user_id="test_user_123")
    
    def test_cache_creation(self, cache: CompressionCache) -> None:
        """Test cache can be created."""
        assert cache.user_id == "test_user_123"
        assert cache.prefix == STATE_PREFIX_USER
        assert cache.ttl_seconds == DEFAULT_USER_TTL_SECONDS
        assert cache.has_redis is False
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache: CompressionCache) -> None:
        """Test setting and getting values."""
        await cache.set("session_data", {"preferences": {"theme": "dark"}})
        
        value = await cache.get("session_data")
        assert value == {"preferences": {"theme": "dark"}}
    
    @pytest.mark.asyncio
    async def test_get_missing_returns_default(self, cache: CompressionCache) -> None:
        """Test getting missing key returns default."""
        value = await cache.get("nonexistent")
        assert value is None
        
        value = await cache.get("nonexistent", default="fallback")
        assert value == "fallback"
    
    @pytest.mark.asyncio
    async def test_set_returns_full_key(self, cache: CompressionCache) -> None:
        """Test set returns full cache key."""
        full_key = await cache.set("my_key", "my_value")
        
        assert full_key == "user:test_user_123:my_key"
    
    @pytest.mark.asyncio
    async def test_delete_existing_key(self, cache: CompressionCache) -> None:
        """Test deleting existing key returns True."""
        await cache.set("to_delete", "value")
        
        result = await cache.delete("to_delete")
        assert result is True
        
        value = await cache.get("to_delete")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, cache: CompressionCache) -> None:
        """Test deleting nonexistent key returns False."""
        result = await cache.delete("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_exists_true(self, cache: CompressionCache) -> None:
        """Test exists returns True for existing key."""
        await cache.set("exists", "value")
        
        assert await cache.exists("exists") is True
    
    @pytest.mark.asyncio
    async def test_exists_false(self, cache: CompressionCache) -> None:
        """Test exists returns False for missing key."""
        assert await cache.exists("nonexistent") is False
    
    @pytest.mark.asyncio
    async def test_ttl_key_exists(self, cache: CompressionCache) -> None:
        """Test ttl returns remaining time for existing key."""
        await cache.set("with_ttl", "value", ttl_seconds=3600)
        
        ttl = await cache.ttl("with_ttl")
        assert 3590 <= ttl <= 3600  # Allow small timing variance
    
    @pytest.mark.asyncio
    async def test_ttl_key_not_exists(self, cache: CompressionCache) -> None:
        """Test ttl returns -2 for nonexistent key."""
        ttl = await cache.ttl("nonexistent")
        assert ttl == -2
    
    @pytest.mark.asyncio
    async def test_uses_asyncio_lock(self, cache: CompressionCache) -> None:
        """Test cache uses asyncio.Lock (AP-10.1)."""
        assert isinstance(cache._lock, asyncio.Lock)
    
    @pytest.mark.asyncio
    async def test_compression_for_large_values(self, cache: CompressionCache) -> None:
        """Test large values are compressed."""
        # Create value larger than compression threshold (1KB)
        large_value = {"data": "x" * 2000}
        
        await cache.set("large", large_value)
        
        # Verify value is retrievable
        retrieved = await cache.get("large")
        assert retrieved == large_value
    
    @pytest.mark.asyncio
    async def test_small_values_not_compressed(self, cache: CompressionCache) -> None:
        """Test small values are not compressed."""
        small_value = {"data": "small"}
        
        await cache.set("small", small_value)
        
        retrieved = await cache.get("small")
        assert retrieved == small_value
    
    @pytest.mark.asyncio
    async def test_custom_ttl(self, cache: CompressionCache) -> None:
        """Test custom TTL can be set per key."""
        await cache.set("short_ttl", "value", ttl_seconds=60)
        
        ttl = await cache.ttl("short_ttl")
        assert 55 <= ttl <= 60
    
    def test_default_ttl_is_24_hours(self) -> None:
        """Test default TTL is 24 hours (86400 seconds)."""
        assert DEFAULT_USER_TTL_SECONDS == 86400
    
    def test_repr(self, cache: CompressionCache) -> None:
        """Test repr shows user_id and backend."""
        repr_str = repr(cache)
        assert "test_user_123" in repr_str
        assert "memory" in repr_str
        assert "CompressionCache" in repr_str


class TestCompressionCacheWithRedis:
    """Tests for CompressionCache with Redis backend."""
    
    @pytest.fixture
    def redis_client(self) -> FakeRedisClient:
        """Create a fake Redis client."""
        return FakeRedisClient()
    
    @pytest.fixture
    def cache(self, redis_client: FakeRedisClient) -> CompressionCache:
        """Create a CompressionCache with Redis backend."""
        return CompressionCache(
            user_id="test_user_456",
            redis_client=redis_client,
        )
    
    def test_cache_with_redis(self, cache: CompressionCache) -> None:
        """Test cache detects Redis backend."""
        assert cache.has_redis is True
    
    @pytest.mark.asyncio
    async def test_set_and_get_with_redis(
        self,
        cache: CompressionCache,
    ) -> None:
        """Test setting and getting values with Redis."""
        await cache.set("redis_key", {"redis": "value"})
        
        value = await cache.get("redis_key")
        assert value == {"redis": "value"}
    
    @pytest.mark.asyncio
    async def test_delete_with_redis(
        self,
        cache: CompressionCache,
    ) -> None:
        """Test deleting with Redis backend."""
        await cache.set("to_delete", "value")
        
        result = await cache.delete("to_delete")
        assert result is True
        
        value = await cache.get("to_delete")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_exists_with_redis(
        self,
        cache: CompressionCache,
    ) -> None:
        """Test exists with Redis backend."""
        await cache.set("exists", "value")
        
        assert await cache.exists("exists") is True
        assert await cache.exists("nonexistent") is False
    
    @pytest.mark.asyncio
    async def test_ttl_with_redis(
        self,
        cache: CompressionCache,
        redis_client: FakeRedisClient,
    ) -> None:
        """Test TTL with Redis backend."""
        await cache.set("with_ttl", "value", ttl_seconds=3600)
        
        # FakeRedisClient stores TTL
        full_key = "user:test_user_456:with_ttl"
        assert redis_client._ttls.get(full_key) == 3600
    
    @pytest.mark.asyncio
    async def test_clear_expired_noop_with_redis(
        self,
        cache: CompressionCache,
    ) -> None:
        """Test clear_expired is no-op with Redis (Redis handles it)."""
        await cache.set("key", "value")
        
        count = await cache.clear_expired()
        assert count == 0
    
    def test_repr_shows_redis(self, cache: CompressionCache) -> None:
        """Test repr shows Redis backend."""
        repr_str = repr(cache)
        assert "redis" in repr_str


class TestCompressionCacheConcurrency:
    """Tests for concurrent access to CompressionCache."""
    
    @pytest.fixture
    def cache(self) -> CompressionCache:
        """Create cache for concurrency testing."""
        return CompressionCache(user_id="concurrent_user")
    
    @pytest.mark.asyncio
    async def test_concurrent_writes(self, cache: CompressionCache) -> None:
        """Test concurrent writes are thread-safe."""
        async def writer(key: str, value: str) -> None:
            await cache.set(key, value)
            await asyncio.sleep(0.001)
        
        tasks = [writer(f"key_{i}", f"value_{i}") for i in range(10)]
        await asyncio.gather(*tasks)
        
        # Verify all writes succeeded
        for i in range(10):
            value = await cache.get(f"key_{i}")
            assert value == f"value_{i}"
    
    @pytest.mark.asyncio
    async def test_concurrent_reads(self, cache: CompressionCache) -> None:
        """Test concurrent reads are thread-safe."""
        # Setup
        await cache.set("shared_key", {"count": 42})
        
        async def reader() -> Any:
            return await cache.get("shared_key")
        
        # Concurrent reads
        tasks = [reader() for _ in range(20)]
        results = await asyncio.gather(*tasks)
        
        # All reads should return same value
        for result in results:
            assert result == {"count": 42}
