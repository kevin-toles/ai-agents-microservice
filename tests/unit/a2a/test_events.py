"""Unit Tests for A2A Event Models.

WBS-PI4: A2A Streaming (SSE) - Event Models
AC-PI4.2: TaskStatusUpdateEvent emitted on state change
AC-PI4.3: TaskArtifactUpdateEvent emitted on artifact creation
AC-PI4.8: Events include timestamp and sequence number

Test Coverage:
- TaskStatusUpdateEvent model validation
- TaskArtifactUpdateEvent model validation
- Event timestamp and sequence number validation
- Event serialization to SSE format
"""

import unittest
from datetime import datetime

import pytest


class TestTaskStatusUpdateEvent(unittest.TestCase):
    """Test suite for TaskStatusUpdateEvent model (AC-PI4.2, AC-PI4.8)."""

    def test_status_update_event_has_required_fields(self) -> None:
        """TaskStatusUpdateEvent has task_id, status, timestamp, sequence."""
        from src.a2a.events import TaskStatusUpdateEvent
        from src.a2a.models import TaskStatus
        
        event = TaskStatusUpdateEvent(
            task_id="test-task-123",
            status=TaskStatus.WORKING,
            timestamp=datetime.now(),
            sequence=1
        )
        
        assert event.task_id == "test-task-123"
        assert event.status == TaskStatus.WORKING
        assert isinstance(event.timestamp, datetime)
        assert event.sequence == 1

    def test_status_update_event_has_event_type(self) -> None:
        """TaskStatusUpdateEvent has event_type field."""
        from src.a2a.events import TaskStatusUpdateEvent
        from src.a2a.models import TaskStatus
        
        event = TaskStatusUpdateEvent(
            task_id="test-task-123",
            status=TaskStatus.COMPLETED,
            timestamp=datetime.now(),
            sequence=5
        )
        
        assert hasattr(event, "event_type")
        assert event.event_type == "status_update"

    def test_status_update_event_accepts_all_task_statuses(self) -> None:
        """TaskStatusUpdateEvent accepts all TaskStatus values."""
        from src.a2a.events import TaskStatusUpdateEvent
        from src.a2a.models import TaskStatus
        
        for status in TaskStatus:
            event = TaskStatusUpdateEvent(
                task_id="test-task-123",
                status=status,
                timestamp=datetime.now(),
                sequence=1
            )
            assert event.status == status

    def test_status_update_event_to_sse_format(self) -> None:
        """TaskStatusUpdateEvent serializes to SSE format."""
        from src.a2a.events import TaskStatusUpdateEvent
        from src.a2a.models import TaskStatus
        
        event = TaskStatusUpdateEvent(
            task_id="test-task-123",
            status=TaskStatus.WORKING,
            timestamp=datetime.now(),
            sequence=2
        )
        
        sse_data = event.to_sse()
        
        assert "event: status_update" in sse_data
        assert "data:" in sse_data
        assert "test-task-123" in sse_data
        assert sse_data.endswith("\n\n")

    def test_status_update_event_serializes_timestamp_as_iso(self) -> None:
        """TaskStatusUpdateEvent serializes timestamp as ISO 8601."""
        from src.a2a.events import TaskStatusUpdateEvent
        from src.a2a.models import TaskStatus
        
        timestamp = datetime(2026, 1, 3, 12, 30, 45)
        event = TaskStatusUpdateEvent(
            task_id="test-task-123",
            status=TaskStatus.WORKING,
            timestamp=timestamp,
            sequence=1
        )
        
        sse_data = event.to_sse()
        
        assert timestamp.isoformat() in sse_data


class TestTaskArtifactUpdateEvent(unittest.TestCase):
    """Test suite for TaskArtifactUpdateEvent model (AC-PI4.3, AC-PI4.8)."""

    def test_artifact_update_event_has_required_fields(self) -> None:
        """TaskArtifactUpdateEvent has task_id, artifact, timestamp, sequence."""
        from src.a2a.events import TaskArtifactUpdateEvent
        from src.a2a.models import Artifact, Part
        
        artifact = Artifact(parts=[Part(type="text", text="Result")])
        event = TaskArtifactUpdateEvent(
            task_id="test-task-123",
            artifact=artifact,
            timestamp=datetime.now(),
            sequence=3
        )
        
        assert event.task_id == "test-task-123"
        assert event.artifact == artifact
        assert isinstance(event.timestamp, datetime)
        assert event.sequence == 3

    def test_artifact_update_event_has_event_type(self) -> None:
        """TaskArtifactUpdateEvent has event_type field."""
        from src.a2a.events import TaskArtifactUpdateEvent
        from src.a2a.models import Artifact, Part
        
        artifact = Artifact(parts=[Part(type="text", text="Result")])
        event = TaskArtifactUpdateEvent(
            task_id="test-task-123",
            artifact=artifact,
            timestamp=datetime.now(),
            sequence=3
        )
        
        assert hasattr(event, "event_type")
        assert event.event_type == "artifact_update"

    def test_artifact_update_event_to_sse_format(self) -> None:
        """TaskArtifactUpdateEvent serializes to SSE format."""
        from src.a2a.events import TaskArtifactUpdateEvent
        from src.a2a.models import Artifact, Part
        
        artifact = Artifact(parts=[Part(type="text", text="Generated code")])
        event = TaskArtifactUpdateEvent(
            task_id="test-task-123",
            artifact=artifact,
            timestamp=datetime.now(),
            sequence=4
        )
        
        sse_data = event.to_sse()
        
        assert "event: artifact_update" in sse_data
        assert "data:" in sse_data
        assert "test-task-123" in sse_data
        assert "Generated code" in sse_data
        assert sse_data.endswith("\n\n")

    def test_artifact_update_event_includes_artifact_parts(self) -> None:
        """TaskArtifactUpdateEvent includes artifact parts in SSE data."""
        from src.a2a.events import TaskArtifactUpdateEvent
        from src.a2a.models import Artifact, Part
        
        artifact = Artifact(parts=[
            Part(type="text", text="First part"),
            Part(type="text", text="Second part")
        ])
        event = TaskArtifactUpdateEvent(
            task_id="test-task-123",
            artifact=artifact,
            timestamp=datetime.now(),
            sequence=5
        )
        
        sse_data = event.to_sse()
        
        assert "First part" in sse_data
        assert "Second part" in sse_data


class TestEventSequencing(unittest.TestCase):
    """Test suite for event sequencing (AC-PI4.8)."""

    def test_events_have_incrementing_sequence_numbers(self) -> None:
        """Events can be ordered by sequence number."""
        from src.a2a.events import TaskStatusUpdateEvent
        from src.a2a.models import TaskStatus
        
        event1 = TaskStatusUpdateEvent(
            task_id="test-task-123",
            status=TaskStatus.SUBMITTED,
            timestamp=datetime.now(),
            sequence=1
        )
        event2 = TaskStatusUpdateEvent(
            task_id="test-task-123",
            status=TaskStatus.WORKING,
            timestamp=datetime.now(),
            sequence=2
        )
        event3 = TaskStatusUpdateEvent(
            task_id="test-task-123",
            status=TaskStatus.COMPLETED,
            timestamp=datetime.now(),
            sequence=3
        )
        
        assert event1.sequence < event2.sequence < event3.sequence

    def test_event_timestamp_is_required(self) -> None:
        """Event timestamp is required field."""
        from src.a2a.events import TaskStatusUpdateEvent
        from src.a2a.models import TaskStatus
        
        with pytest.raises(Exception):  # Will be ValidationError from pydantic
            TaskStatusUpdateEvent(
                task_id="test-task-123",
                status=TaskStatus.WORKING,
                sequence=1
                # Missing timestamp
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
