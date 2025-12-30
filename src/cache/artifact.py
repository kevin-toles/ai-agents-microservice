"""ADK-aligned Artifact dataclass with version tracking.

Implements ADK's Artifact service pattern for versioned binary storage,
mapped to the platform's cache hierarchy.

Pattern: ADK Artifact Conventions
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Artifact Conventions

Anti-Pattern Compliance:
- AP-1.5: No mutable default arguments (uses field(default_factory=dict))
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import hashlib


@dataclass(frozen=False)
class Artifact:
    """ADK-aligned artifact with version tracking.
    
    Follows ADK's artifact pattern for versioned binary storage
    while integrating with platform's cache hierarchy.
    
    Attributes:
        name: Artifact identifier within namespace
        namespace: Pipeline or agent function ID
        version: Auto-incremented version number
        mime_type: Content type (e.g., "application/json", "text/plain")
        data: Binary content of the artifact
        created_at: Timestamp when artifact was created
        metadata: Additional context (source, checksum, etc.)
    
    Example:
        >>> artifact = Artifact(
        ...     name="chapter_summary",
        ...     namespace="summarize_content",
        ...     version=1,
        ...     mime_type="application/json",
        ...     data=b'{"summary": "..."}',
        ... )
        >>> artifact.qualified_name
        'summarize_content/chapter_summary_v1'
    """
    
    name: str
    namespace: str
    version: int
    mime_type: str
    data: bytes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # AP-1.5: Use field(default_factory=dict) instead of = {}
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate artifact fields after initialization."""
        if not self.name:
            raise ValueError("Artifact name cannot be empty")
        if not self.namespace:
            raise ValueError("Artifact namespace cannot be empty")
        if self.version < 1:
            raise ValueError("Artifact version must be >= 1")
    
    @property
    def qualified_name(self) -> str:
        """Return ADK-style qualified artifact name.
        
        Format: "{namespace}/{name}_v{version}"
        
        Returns:
            Qualified artifact name for identification
        """
        return f"{self.namespace}/{self.name}_v{self.version}"
    
    @property
    def cache_key(self) -> str:
        """Return platform cache key with app: prefix.
        
        Artifacts use the app: prefix for permanent storage.
        
        Returns:
            Cache key in format "app:artifact:{qualified_name}"
        """
        return f"app:artifact:{self.qualified_name}"
    
    @property
    def checksum(self) -> str:
        """Calculate SHA-256 checksum of artifact data.
        
        Returns:
            Hex-encoded SHA-256 hash of the data
        """
        return hashlib.sha256(self.data).hexdigest()
    
    @property
    def size_bytes(self) -> int:
        """Return size of artifact data in bytes.
        
        Returns:
            Length of data in bytes
        """
        return len(self.data)
    
    def with_incremented_version(self, new_data: bytes | None = None) -> "Artifact":
        """Create a new artifact with incremented version.
        
        Args:
            new_data: Optional new data for the artifact. If None, uses existing data.
        
        Returns:
            New Artifact instance with version incremented by 1
        """
        return Artifact(
            name=self.name,
            namespace=self.namespace,
            version=self.version + 1,
            mime_type=self.mime_type,
            data=new_data if new_data is not None else self.data,
            created_at=datetime.now(timezone.utc),
            metadata=dict(self.metadata),  # Copy to avoid sharing
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert artifact to dictionary representation.
        
        Note: data is excluded from dict representation to avoid
        serializing large binary content. Use checksum for verification.
        
        Returns:
            Dictionary with artifact metadata (excluding data)
        """
        return {
            "name": self.name,
            "namespace": self.namespace,
            "version": self.version,
            "qualified_name": self.qualified_name,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ArtifactReference:
    """Immutable reference to an artifact without data.
    
    Used for passing artifact identifiers between pipeline stages
    without copying the actual data.
    
    Attributes:
        namespace: Pipeline or agent function ID
        name: Artifact identifier
        version: Specific version to reference (None for latest)
    """
    
    namespace: str
    name: str
    version: int | None = None
    
    @property
    def qualified_name(self) -> str:
        """Return qualified name, using 'latest' if version is None."""
        if self.version is None:
            return f"{self.namespace}/{self.name}_latest"
        return f"{self.namespace}/{self.name}_v{self.version}"
    
    @property
    def cache_key(self) -> str:
        """Return cache key for this reference."""
        return f"app:artifact:{self.qualified_name}"
