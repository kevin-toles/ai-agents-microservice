"""HandoffCache - Pipeline-local state management using temp: prefix.

Implements ADK's temp: state prefix for pipeline handoff data that is
discarded after pipeline completes. Uses asyncio.Lock for thread safety.

Pattern: ADK temp: State Prefix
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → State Prefix Conventions
Anti-Pattern: AP-10.1 - Uses asyncio.Lock instead of threading.Lock

Anti-Pattern Compliance:
- AP-1.5: No mutable default arguments
- AP-10.1: Uses asyncio.Lock for async context
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar, Generic

from src.cache.state import STATE_PREFIX_TEMP, build_cache_key


T = TypeVar("T")


@dataclass
class HandoffEntry(Generic[T]):
    """Entry in the handoff cache with metadata.
    
    Attributes:
        key: Cache key (without prefix)
        value: Stored value
        created_at: When the entry was created
        source_stage: Pipeline stage that created this entry
        metadata: Additional context
    """
    
    key: str
    value: T
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_stage: str = ""
    # AP-1.5: Use field(default_factory=dict) instead of = {}
    metadata: dict[str, Any] = field(default_factory=dict)


class HandoffCache:
    """Pipeline-local cache for temp: prefix state.
    
    Provides thread-safe async storage for data that flows between
    pipeline stages and is discarded after pipeline completion.
    
    Uses asyncio.Lock per AP-10.1 for async context safety.
    
    Example:
        >>> cache = HandoffCache(pipeline_id="pipeline_123")
        >>> await cache.set("stage_1_output", {"result": "..."})
        >>> data = await cache.get("stage_1_output")
        >>> await cache.clear()  # Called when pipeline completes
    
    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → ADK Pattern Integration
    """
    
    def __init__(self, pipeline_id: str) -> None:
        """Initialize handoff cache for a pipeline execution.
        
        Args:
            pipeline_id: Unique identifier for the pipeline execution
        """
        self._pipeline_id = pipeline_id
        self._store: dict[str, HandoffEntry[Any]] = {}
        # AP-10.1: Use asyncio.Lock for async context
        self._lock = asyncio.Lock()
        self._created_at = datetime.now(timezone.utc)
    
    @property
    def pipeline_id(self) -> str:
        """Return the pipeline ID this cache is associated with."""
        return self._pipeline_id
    
    @property
    def prefix(self) -> str:
        """Return the state prefix for this cache tier."""
        return STATE_PREFIX_TEMP
    
    def _build_key(self, key: str) -> str:
        """Build full cache key with prefix and namespace.
        
        Args:
            key: Short key name
        
        Returns:
            Full cache key in ADK format
        """
        return build_cache_key(STATE_PREFIX_TEMP, self._pipeline_id, key)
    
    async def set(
        self,
        key: str,
        value: T,
        source_stage: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a value in the handoff cache.
        
        Args:
            key: Key to store under (without prefix)
            value: Value to store
            source_stage: Pipeline stage creating this entry
            metadata: Optional additional context
        
        Returns:
            Full cache key that was used
        """
        full_key = self._build_key(key)
        entry = HandoffEntry(
            key=key,
            value=value,
            source_stage=source_stage,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self._store[full_key] = entry
        
        return full_key
    
    async def get(self, key: str, default: T | None = None) -> T | None:
        """Retrieve a value from the handoff cache.
        
        Args:
            key: Key to retrieve (without prefix)
            default: Default value if key not found
        
        Returns:
            Stored value or default
        """
        full_key = self._build_key(key)
        
        async with self._lock:
            entry = self._store.get(full_key)
            if entry is None:
                return default
            return entry.value
    
    async def get_entry(self, key: str) -> HandoffEntry[Any] | None:
        """Retrieve full entry with metadata.
        
        Args:
            key: Key to retrieve (without prefix)
        
        Returns:
            HandoffEntry with metadata or None
        """
        full_key = self._build_key(key)
        
        async with self._lock:
            return self._store.get(full_key)
    
    async def delete(self, key: str) -> bool:
        """Delete a key from the cache.
        
        Args:
            key: Key to delete (without prefix)
        
        Returns:
            True if key was deleted, False if not found
        """
        full_key = self._build_key(key)
        
        async with self._lock:
            if full_key in self._store:
                del self._store[full_key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Key to check (without prefix)
        
        Returns:
            True if key exists
        """
        full_key = self._build_key(key)
        
        async with self._lock:
            return full_key in self._store
    
    async def keys(self) -> list[str]:
        """Return all keys in the cache (without prefix).
        
        Returns:
            List of short key names
        """
        # Build prefix pattern: "temp:{pipeline_id}:"
        prefix_pattern = f"{STATE_PREFIX_TEMP}{self._pipeline_id}:"
        prefix_len = len(prefix_pattern)
        
        async with self._lock:
            return [k[prefix_len:] for k in self._store.keys()]
    
    async def clear(self) -> int:
        """Clear all entries from the cache.
        
        Called when pipeline completes to release temp: state.
        
        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._store)
            self._store.clear()
            return count
    
    async def size(self) -> int:
        """Return number of entries in the cache.
        
        Returns:
            Number of cached entries
        """
        async with self._lock:
            return len(self._store)
    
    async def get_all_entries(self) -> dict[str, HandoffEntry[Any]]:
        """Return all entries for debugging/inspection.
        
        Returns:
            Copy of internal store
        """
        async with self._lock:
            return dict(self._store)
    
    def __repr__(self) -> str:
        """Return string representation of cache."""
        return (
            f"HandoffCache(pipeline_id={self._pipeline_id!r}, "
            f"entries={len(self._store)})"
        )
