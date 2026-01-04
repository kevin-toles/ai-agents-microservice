"""A2A Protocol Models.

WBS-PI2: A2A Agent Card & Discovery - Models
WBS-PI3: A2A Task Lifecycle & Endpoints - Task Models

Implements:
- Skill: Agent function capability declaration
- Capability: Service capability flags
- AgentCard: Complete service discovery manifest
- TaskStatus: Task state enum (AC-PI3.2)
- Part: Message/artifact content part
- Artifact: Task output artifact (AC-PI3.1)
- Message: A2A protocol message
- Task: A2A task with lifecycle (AC-PI3.1)
- SendMessageRequest/Response: API request/response models (AC-PI3.3)

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → A2A Protocol Integration
"""

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# =============================================================================
# AC-PI2.2, AC-PI2.8: Skill Model
# =============================================================================


class Skill(BaseModel):
    """A2A Skill model representing an agent function capability.
    
    Maps to agent functions with their metadata for discovery.
    
    Attributes:
        id: Unique identifier (matches agent function name)
        name: Human-readable display name
        description: Detailed capability description
        tags: Categorization tags for filtering/search
        examples: Example use cases
        inputModes: Supported input content types (AC-PI2.8)
        outputModes: Supported output content types (AC-PI2.8)
    """

    id: str = Field(..., description="Unique skill identifier")
    name: str = Field(..., description="Human-readable skill name")
    description: str = Field(..., description="Skill capability description")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    examples: list[str] = Field(default_factory=list, description="Example use cases")
    inputModes: list[str] = Field(
        default_factory=lambda: ["application/json"],
        description="Supported input content types",
    )
    outputModes: list[str] = Field(
        default_factory=lambda: ["application/json"],
        description="Supported output content types",
    )


# =============================================================================
# AC-PI2.7: Capability Model
# =============================================================================


class Capability(BaseModel):
    """A2A Capability model for service feature flags.
    
    Declares optional features supported by the service.
    
    Attributes:
        streaming: Supports SSE streaming (AC-PI2.7)
        pushNotifications: Supports webhook notifications
        stateTransitionHistory: Maintains task state history
    """

    streaming: bool = Field(
        default=False,
        description="Supports Server-Sent Events streaming",
    )
    pushNotifications: bool = Field(
        default=False,
        description="Supports webhook push notifications",
    )
    stateTransitionHistory: bool = Field(
        default=False,
        description="Maintains task state transition history",
    )


# =============================================================================
# AC-PI2.1, AC-PI2.6: AgentCard Model
# =============================================================================


class AgentCard(BaseModel):
    """A2A Agent Card for service discovery.
    
    Complete manifest declaring service capabilities, skills, and metadata.
    Complies with A2A Protocol Specification v0.3.0 (AC-PI2.1).
    
    Attributes:
        protocolVersion: A2A protocol version (AC-PI2.6)
        name: Service identifier
        description: Service description
        version: Service version
        capabilities: Feature capability flags (AC-PI2.7)
        skills: List of available skills
    """

    protocolVersion: str = Field(
        default="0.3.0",
        description="A2A protocol version",
    )
    name: str = Field(..., description="Service name")
    description: str = Field(..., description="Service description")
    version: str = Field(..., description="Service version")
    capabilities: Capability = Field(..., description="Service capabilities")
    skills: list[Skill] = Field(default_factory=list, description="Available skills")

    class Config:
        """Pydantic configuration."""

        # Use camelCase for JSON serialization (A2A spec requirement)
        populate_by_name = True


# =============================================================================
# AC-PI3.2: TaskStatus Enum
# =============================================================================


class TaskStatus(str, Enum):
    """A2A Task status enumeration.
    
    Defines all possible task lifecycle states as per A2A spec.
    
    States:
        SUBMITTED: Task queued, not yet started
        WORKING: Task currently executing
        INPUT_REQUIRED: Task paused, needs additional input
        COMPLETED: Task finished successfully
        FAILED: Task failed with error
        CANCELED: Task canceled by user request
    """

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


# =============================================================================
# AC-PI3.1: Part and Artifact Models
# =============================================================================


class Part(BaseModel):
    """Content part for messages and artifacts.
    
    Represents a piece of content with flexible type and payload.
    
    Attributes:
        type: Content type (e.g., "text", "data", "image")
        text: Text content (for type="text")
        data: Structured data content (for type="data")
    """

    type: str = Field(..., description="Content type")
    text: str | None = Field(default=None, description="Text content")
    data: dict[str, Any] | None = Field(default=None, description="Data content")


class Artifact(BaseModel):
    """Task output artifact.
    
    Contains the results produced by a task execution.
    
    Attributes:
        parts: List of content parts making up the artifact
    """

    parts: list[Part] = Field(default_factory=list, description="Artifact content parts")


# =============================================================================
# AC-PI3.3: Message Models
# =============================================================================


class Message(BaseModel):
    """A2A protocol message.
    
    Represents a message sent to or from an agent.
    
    Attributes:
        parts: Message content parts
        skillId: Optional skill identifier to invoke
    """

    parts: list[Part] = Field(..., description="Message content parts")
    skillId: str | None = Field(default=None, description="Skill to invoke")


class SendMessageRequest(BaseModel):
    """Request model for A2A SendMessage endpoint (AC-PI3.3).
    
    Attributes:
        message: The message to send
        contextId: Optional context identifier for conversation continuity
    """

    message: Message = Field(..., description="Message to send")
    contextId: str | None = Field(default=None, description="Context identifier")


class SendMessageResponse(BaseModel):
    """Response model for A2A SendMessage endpoint.
    
    Attributes:
        taskId: Identifier for the created task
    """

    taskId: str = Field(..., description="Task identifier")


# =============================================================================
# AC-PI3.1: Task Model
# =============================================================================


class Task(BaseModel):
    """A2A Task model with full lifecycle (AC-PI3.1).
    
    Represents an executing or completed task with status, artifacts, and history.
    
    Attributes:
        id: Unique task identifier
        status: Current task status (AC-PI3.2)
        created_at: Task creation timestamp
        updated_at: Last update timestamp
        artifacts: Output artifacts (AC-PI3.1)
        history: State transition history (AC-PI3.1)
        skillId: Optional skill identifier
        error: Error message if status=FAILED
    """

    id: str = Field(..., description="Task identifier")
    status: TaskStatus = Field(..., description="Task status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")
    artifacts: list[Artifact] = Field(
        default_factory=list,
        description="Task output artifacts"
    )
    history: list[str] = Field(
        default_factory=list,
        description="State transition history"
    )
    skillId: str | None = Field(default=None, description="Skill identifier")
    error: str | None = Field(default=None, description="Error message")


__all__ = [
    "AgentCard",
    "Artifact",
    "Capability",
    "Message",
    "Part",
    "SendMessageRequest",
    "SendMessageResponse",
    "Skill",
    "Task",
    "TaskStatus",
]
