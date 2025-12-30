"""CompressionCache - User session state using user: prefix with Redis.

Implements ADK's user: state prefix for cross-session persistence with
24-hour TTL. Backs to Redis for distributed caching.

Pattern: ADK user: State Prefix
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → State Prefix Conventions

Anti-Pattern Compliance:
- AP-1.5: No mutable default arguments
- AP-10.1: Uses asyncio.Lock for async context
"""

import asyncio
import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable

from src.cache.state import STATE_PREFIX_USER, build_cache_key


# Default TTL for user: prefix (24 hours)
DEFAULT_USER_TTL_SECONDS: int = 86400  # 24 * 60 * 60


@runtime_checkable
class RedisClientProtocol(Protocol):
    """Protocol for Redis client duck typing.
    
    Allows dependency injection of Redis client implementations
    for testing (FakeRedisClient) and production (redis.asyncio.Redis).
    """
    
    async def get(self, key: str) -> bytes | None:
        """Get a value from Redis."""
        ...
    
    async def set(
        self,
        key: str,
        value: bytes,
        ex: int | None = None,
    ) -> bool:
        """Set a value in Redis with optional expiry."""
        ...
    
    async def delete(self, key: str) -> int:
        """Delete a key from Redis."""
        ...
    
    async def exists(self, key: str) -> int:
        """Check if key exists in Redis."""
        ...
    
    async def ttl(self, key: str) -> int:
        """Get TTL of a key in seconds."""
        ...


@dataclass
class CacheEntry:
    """Cached entry with compression metadata.
    
    Attributes:
        key: Cache key (without prefix)
        value: Original value (before compression)
        compressed: Whether data was compressed
        original_size: Size before compression
        compressed_size: Size after compression
        created_at: When entry was created
        expires_at: When entry expires
        metadata: Additional context
    """
    
    key: str
    value: Any
    compressed: bool = False
    original_size: int = 0
    compressed_size: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    # AP-1.5: Use field(default_factory=dict) instead of = {}
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (smaller is better).
        
        Returns:
            Ratio of compressed to original size (0.0-1.0)
        """
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired.
        
        Returns:
            True if entry is past expiration time
        """
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class CompressionCache:
    """User session cache with compression for user: prefix state.
    
    Provides compressed storage for user session data that persists
    across invocations with 24-hour TTL. Backs to Redis when available.
    
    Features:
    - Automatic gzip compression for values > 1KB
    - 24-hour TTL (configurable)
    - Optional Redis backend for distributed caching
    - In-memory fallback when Redis unavailable
    
    Example:
        >>> cache = CompressionCache(user_id="user_123")
        >>> await cache.set("session_data", {"preferences": {...}})
        >>> data = await cache.get("session_data")
        
        # With Redis backend
        >>> redis = redis.asyncio.from_url("redis://localhost")
        >>> cache = CompressionCache(user_id="user_123", redis_client=redis)
    
    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → ADK Pattern Integration
    """
    
    # Compress values larger than this (bytes)
    COMPRESSION_THRESHOLD = 1024  # 1KB
    
    def __init__(
        self,
        user_id: str,
        redis_client: RedisClientProtocol | None = None,
        ttl_seconds: int = DEFAULT_USER_TTL_SECONDS,
    ) -> None:
        """Initialize compression cache for a user session.
        
        Args:
            user_id: Unique identifier for the user
            redis_client: Optional Redis client for distributed caching
            ttl_seconds: Time-to-live for cached entries (default 24h)
        """
        self._user_id = user_id
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds
        # In-memory fallback when Redis unavailable
        self._local_store: dict[str, bytes] = {}
        self._local_ttls: dict[str, datetime] = {}
        # AP-10.1: Use asyncio.Lock for async context
        self._lock = asyncio.Lock()
    
    @property
    def user_id(self) -> str:
        """Return the user ID this cache is associated with."""
        return self._user_id
    
    @property
    def prefix(self) -> str:
        """Return the state prefix for this cache tier."""
        return STATE_PREFIX_USER
    
    @property
    def ttl_seconds(self) -> int:
        """Return the TTL in seconds for this cache."""
        return self._ttl_seconds
    
    @property
    def has_redis(self) -> bool:
        """Check if Redis backend is available."""
        return self._redis is not None
    
    def _build_key(self, key: str) -> str:
        """Build full cache key with prefix and namespace.
        
        Args:
            key: Short key name
        
        Returns:
            Full cache key in ADK format
        """
        return build_cache_key(STATE_PREFIX_USER, self._user_id, key)
    
    def _compress(self, data: bytes) -> tuple[bytes, bool]:
        """Compress data if above threshold.
        
        Args:
            data: Raw bytes to potentially compress
        
        Returns:
            Tuple of (compressed_or_original_data, was_compressed)
        """
        if len(data) < self.COMPRESSION_THRESHOLD:
            return data, False
        
        compressed = gzip.compress(data, compresslevel=6)
        # Only use compressed if it's actually smaller
        if len(compressed) < len(data):
            return compressed, True
        return data, False
    
    def _decompress(self, data: bytes, was_compressed: bool) -> bytes:
        """Decompress data if it was compressed.
        
        Args:
            data: Potentially compressed bytes
            was_compressed: Whether data was compressed
        
        Returns:
            Decompressed data
        """
        if not was_compressed:
            return data
        return gzip.decompress(data)
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes.
        
        Args:
            value: Python object to serialize
        
        Returns:
            JSON-encoded bytes
        """
        return json.dumps(value).encode("utf-8")
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to Python object.
        
        Args:
            data: JSON-encoded bytes
        
        Returns:
            Deserialized Python object
        """
        return json.loads(data.decode("utf-8"))
    
    def _wrap_for_storage(self, value: Any) -> bytes:
        """Wrap value with compression metadata for storage.
        
        Args:
            value: Value to store
        
        Returns:
            Bytes with compression header
        """
        raw_bytes = self._serialize(value)
        compressed_bytes, was_compressed = self._compress(raw_bytes)
        
        # Prepend 1-byte header: 0x00 = not compressed, 0x01 = compressed
        header = bytes([0x01 if was_compressed else 0x00])
        return header + compressed_bytes
    
    def _unwrap_from_storage(self, data: bytes) -> Any:
        """Unwrap stored value with compression handling.
        
        Args:
            data: Stored bytes with compression header
        
        Returns:
            Deserialized Python object
        """
        if len(data) < 1:
            raise ValueError("Invalid cached data: too short")
        
        was_compressed = data[0] == 0x01
        payload = data[1:]
        
        decompressed = self._decompress(payload, was_compressed)
        return self._deserialize(decompressed)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a value in the compression cache.
        
        Args:
            key: Key to store under (without prefix)
            value: Value to store (must be JSON-serializable)
            ttl_seconds: Override default TTL for this entry
            metadata: Optional additional context (stored locally only)
        
        Returns:
            Full cache key that was used
        """
        full_key = self._build_key(key)
        ttl = ttl_seconds or self._ttl_seconds
        wrapped = self._wrap_for_storage(value)
        
        async with self._lock:
            if self._redis is not None:
                await self._redis.set(full_key, wrapped, ex=ttl)
            else:
                # In-memory fallback
                self._local_store[full_key] = wrapped
                self._local_ttls[full_key] = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        
        return full_key
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the compression cache.
        
        Args:
            key: Key to retrieve (without prefix)
            default: Default value if key not found or expired
        
        Returns:
            Stored value or default
        """
        full_key = self._build_key(key)
        
        async with self._lock:
            if self._redis is not None:
                data = await self._redis.get(full_key)
                if data is None:
                    return default
                return self._unwrap_from_storage(data)
            else:
                # Check local store with TTL
                if full_key not in self._local_store:
                    return default
                
                # Check expiration
                expires_at = self._local_ttls.get(full_key)
                if expires_at and datetime.now(timezone.utc) > expires_at:
                    del self._local_store[full_key]
                    del self._local_ttls[full_key]
                    return default
                
                return self._unwrap_from_storage(self._local_store[full_key])
    
    async def delete(self, key: str) -> bool:
        """Delete a key from the cache.
        
        Args:
            key: Key to delete (without prefix)
        
        Returns:
            True if key was deleted, False if not found
        """
        full_key = self._build_key(key)
        
        async with self._lock:
            if self._redis is not None:
                result = await self._redis.delete(full_key)
                return result > 0
            else:
                if full_key in self._local_store:
                    del self._local_store[full_key]
                    self._local_ttls.pop(full_key, None)
                    return True
                return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache (and not expired).
        
        Args:
            key: Key to check (without prefix)
        
        Returns:
            True if key exists and is not expired
        """
        full_key = self._build_key(key)
        
        async with self._lock:
            if self._redis is not None:
                result = await self._redis.exists(full_key)
                return result > 0
            else:
                if full_key not in self._local_store:
                    return False
                
                # Check expiration
                expires_at = self._local_ttls.get(full_key)
                if expires_at and datetime.now(timezone.utc) > expires_at:
                    return False
                return True
    
    async def ttl(self, key: str) -> int:
        """Get remaining TTL for a key in seconds.
        
        Args:
            key: Key to check (without prefix)
        
        Returns:
            Remaining TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        full_key = self._build_key(key)
        
        async with self._lock:
            if self._redis is not None:
                return await self._redis.ttl(full_key)
            else:
                if full_key not in self._local_store:
                    return -2
                
                expires_at = self._local_ttls.get(full_key)
                if expires_at is None:
                    return -1
                
                remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
                return max(0, int(remaining))
    
    async def clear_expired(self) -> int:
        """Clear expired entries from local cache.
        
        Only applicable when using in-memory fallback.
        Redis handles expiration automatically.
        
        Returns:
            Number of entries cleared
        """
        if self._redis is not None:
            return 0  # Redis handles this automatically
        
        now = datetime.now(timezone.utc)
        expired_keys = []
        
        async with self._lock:
            for key, expires_at in self._local_ttls.items():
                if expires_at and now > expires_at:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._local_store.pop(key, None)
                self._local_ttls.pop(key, None)
        
        return len(expired_keys)
    
    def __repr__(self) -> str:
        """Return string representation of cache."""
        backend = "redis" if self._redis else "memory"
        return (
            f"CompressionCache(user_id={self._user_id!r}, "
            f"backend={backend}, ttl={self._ttl_seconds}s)"
        )
