"""A2A Protocol Event Models.

WBS-PI4: A2A Streaming (SSE) - Event Models
AC-PI4.2: TaskStatusUpdateEvent on state change
AC-PI4.3: TaskArtifactUpdateEvent on artifact creation
AC-PI4.8: Events include timestamp and sequence number

Implements:
- TaskStatusUpdateEvent: Emitted when task status changes
- TaskArtifactUpdateEvent: Emitted when task produces artifacts
- SSE serialization for event streaming

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → A2A Streaming
"""

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.a2a.models import Artifact, TaskStatus


class TaskStatusUpdateEvent(BaseModel):
    """Task status update event (AC-PI4.2, AC-PI4.8).
    
    Emitted when a task transitions between states (submitted → working → completed).
    
    Attributes:
        event_type: Always "status_update"
        task_id: Task identifier
        status: New task status
        timestamp: Event creation time (ISO 8601)
        sequence: Event sequence number for ordering
    """

    event_type: Literal["status_update"] = Field(
        default="status_update",
        description="Event type identifier"
    )
    task_id: str = Field(..., description="Task identifier")
    status: TaskStatus = Field(..., description="New task status")
    timestamp: datetime = Field(..., description="Event timestamp")
    sequence: int = Field(..., description="Event sequence number", ge=1)

    def to_sse(self) -> str:
        """Serialize event to SSE format.
        
        Returns:
            SSE-formatted string with event type and JSON data
        """
        data = {
            "task_id": self.task_id,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
        }
        
        return f"event: status_update\ndata: {json.dumps(data)}\n\n"


class TaskArtifactUpdateEvent(BaseModel):
    """Task artifact update event (AC-PI4.3, AC-PI4.8).
    
    Emitted when a task produces output artifacts.
    
    Attributes:
        event_type: Always "artifact_update"
        task_id: Task identifier
        artifact: Task output artifact with parts
        timestamp: Event creation time (ISO 8601)
        sequence: Event sequence number for ordering
    """

    event_type: Literal["artifact_update"] = Field(
        default="artifact_update",
        description="Event type identifier"
    )
    task_id: str = Field(..., description="Task identifier")
    artifact: Artifact = Field(..., description="Task output artifact")
    timestamp: datetime = Field(..., description="Event timestamp")
    sequence: int = Field(..., description="Event sequence number", ge=1)

    def to_sse(self) -> str:
        """Serialize event to SSE format.
        
        Returns:
            SSE-formatted string with event type and JSON data
        """
        data = {
            "task_id": self.task_id,
            "artifact": self.artifact.model_dump(),
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
        }
        
        return f"event: artifact_update\ndata: {json.dumps(data)}\n\n"


__all__ = [
    "TaskStatusUpdateEvent",
    "TaskArtifactUpdateEvent",
]
