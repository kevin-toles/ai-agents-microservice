"""Unit tests for src/cache/artifact module.

Tests ADK-aligned Artifact dataclass with version tracking.

Reference: WBS-AGT3 AC-3.3, AC-3.6
"""

import pytest
from datetime import datetime, timezone

from src.cache.artifact import Artifact, ArtifactReference


class TestArtifact:
    """Tests for Artifact dataclass (AC-3.3)."""
    
    @pytest.fixture
    def sample_artifact(self) -> Artifact:
        """Create a sample artifact for testing."""
        return Artifact(
            name="chapter_summary",
            namespace="summarize_content",
            version=1,
            mime_type="application/json",
            data=b'{"summary": "This is a test summary"}',
        )
    
    def test_artifact_creation(self, sample_artifact: Artifact) -> None:
        """Test artifact can be created with required fields."""
        assert sample_artifact.name == "chapter_summary"
        assert sample_artifact.namespace == "summarize_content"
        assert sample_artifact.version == 1
        assert sample_artifact.mime_type == "application/json"
        assert b"summary" in sample_artifact.data
    
    def test_qualified_name_format(self, sample_artifact: Artifact) -> None:
        """Test qualified_name returns correct format.
        
        Exit Criteria: Artifact.qualified_name returns "{namespace}/{name}_v{version}"
        """
        assert sample_artifact.qualified_name == "summarize_content/chapter_summary_v1"
    
    def test_qualified_name_with_version(self) -> None:
        """Test qualified_name with different versions."""
        artifact_v2 = Artifact(
            name="output",
            namespace="pipeline_123",
            version=2,
            mime_type="text/plain",
            data=b"test",
        )
        assert artifact_v2.qualified_name == "pipeline_123/output_v2"
    
    def test_cache_key_format(self, sample_artifact: Artifact) -> None:
        """Test cache_key uses app: prefix."""
        expected = "app:artifact:summarize_content/chapter_summary_v1"
        assert sample_artifact.cache_key == expected
    
    def test_checksum_computed(self, sample_artifact: Artifact) -> None:
        """Test checksum is SHA-256 hex string."""
        checksum = sample_artifact.checksum
        assert len(checksum) == 64  # SHA-256 hex is 64 chars
        assert all(c in "0123456789abcdef" for c in checksum)
    
    def test_checksum_deterministic(self, sample_artifact: Artifact) -> None:
        """Test same data produces same checksum."""
        checksum1 = sample_artifact.checksum
        checksum2 = sample_artifact.checksum
        assert checksum1 == checksum2
    
    def test_different_data_different_checksum(self) -> None:
        """Test different data produces different checksum."""
        artifact1 = Artifact(
            name="a", namespace="ns", version=1,
            mime_type="text/plain", data=b"data1",
        )
        artifact2 = Artifact(
            name="a", namespace="ns", version=1,
            mime_type="text/plain", data=b"data2",
        )
        assert artifact1.checksum != artifact2.checksum
    
    def test_size_bytes(self, sample_artifact: Artifact) -> None:
        """Test size_bytes returns data length."""
        assert sample_artifact.size_bytes == len(sample_artifact.data)
    
    def test_created_at_default(self) -> None:
        """Test created_at defaults to current UTC time."""
        before = datetime.now(timezone.utc)
        artifact = Artifact(
            name="test", namespace="ns", version=1,
            mime_type="text/plain", data=b"test",
        )
        after = datetime.now(timezone.utc)
        
        assert before <= artifact.created_at <= after
    
    def test_metadata_default_empty_dict(self) -> None:
        """Test metadata defaults to empty dict (AP-1.5 compliance)."""
        artifact = Artifact(
            name="test", namespace="ns", version=1,
            mime_type="text/plain", data=b"test",
        )
        assert artifact.metadata == {}
        assert isinstance(artifact.metadata, dict)
    
    def test_metadata_not_shared_between_instances(self) -> None:
        """Test metadata is not shared (AP-1.5 compliance)."""
        artifact1 = Artifact(
            name="test", namespace="ns", version=1,
            mime_type="text/plain", data=b"test",
        )
        artifact2 = Artifact(
            name="test2", namespace="ns", version=1,
            mime_type="text/plain", data=b"test",
        )
        
        artifact1.metadata["key"] = "value"
        assert "key" not in artifact2.metadata
    
    def test_with_incremented_version(self, sample_artifact: Artifact) -> None:
        """Test with_incremented_version creates new artifact."""
        new_artifact = sample_artifact.with_incremented_version()
        
        assert new_artifact.version == sample_artifact.version + 1
        assert new_artifact.name == sample_artifact.name
        assert new_artifact.namespace == sample_artifact.namespace
        assert new_artifact.data == sample_artifact.data
        assert new_artifact is not sample_artifact
    
    def test_with_incremented_version_new_data(self, sample_artifact: Artifact) -> None:
        """Test with_incremented_version can use new data."""
        new_data = b"updated content"
        new_artifact = sample_artifact.with_incremented_version(new_data=new_data)
        
        assert new_artifact.version == 2
        assert new_artifact.data == new_data
    
    def test_to_dict_excludes_data(self, sample_artifact: Artifact) -> None:
        """Test to_dict excludes binary data."""
        d = sample_artifact.to_dict()
        
        assert "data" not in d
        assert d["name"] == "chapter_summary"
        assert d["namespace"] == "summarize_content"
        assert d["version"] == 1
        assert d["qualified_name"] == "summarize_content/chapter_summary_v1"
        assert "checksum" in d
        assert "size_bytes" in d
    
    def test_validation_empty_name_raises(self) -> None:
        """Test empty name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Artifact(
                name="", namespace="ns", version=1,
                mime_type="text/plain", data=b"test",
            )
        assert "name cannot be empty" in str(exc_info.value)
    
    def test_validation_empty_namespace_raises(self) -> None:
        """Test empty namespace raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Artifact(
                name="test", namespace="", version=1,
                mime_type="text/plain", data=b"test",
            )
        assert "namespace cannot be empty" in str(exc_info.value)
    
    def test_validation_zero_version_raises(self) -> None:
        """Test version < 1 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Artifact(
                name="test", namespace="ns", version=0,
                mime_type="text/plain", data=b"test",
            )
        assert "version must be >= 1" in str(exc_info.value)


class TestArtifactReference:
    """Tests for ArtifactReference dataclass."""
    
    def test_reference_creation(self) -> None:
        """Test reference can be created."""
        ref = ArtifactReference(
            namespace="summarize_content",
            name="chapter_summary",
            version=1,
        )
        assert ref.namespace == "summarize_content"
        assert ref.name == "chapter_summary"
        assert ref.version == 1
    
    def test_reference_qualified_name_with_version(self) -> None:
        """Test qualified_name with specific version."""
        ref = ArtifactReference(namespace="ns", name="artifact", version=2)
        assert ref.qualified_name == "ns/artifact_v2"
    
    def test_reference_qualified_name_latest(self) -> None:
        """Test qualified_name without version uses 'latest'."""
        ref = ArtifactReference(namespace="ns", name="artifact")
        assert ref.qualified_name == "ns/artifact_latest"
    
    def test_reference_cache_key(self) -> None:
        """Test cache_key format."""
        ref = ArtifactReference(namespace="ns", name="artifact", version=1)
        assert ref.cache_key == "app:artifact:ns/artifact_v1"
    
    def test_reference_is_immutable(self) -> None:
        """Test reference is frozen dataclass."""
        ref = ArtifactReference(namespace="ns", name="artifact")
        with pytest.raises(Exception):  # FrozenInstanceError
            ref.name = "changed"  # type: ignore
