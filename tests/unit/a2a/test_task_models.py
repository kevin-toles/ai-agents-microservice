"""Tests for A2A task models.

TDD tests for WBS-PI3: A2A Task Lifecycle & Endpoints - Task Models.

Acceptance Criteria Coverage:
- AC-PI3.1: Task model includes id, status, artifacts, history
- AC-PI3.2: TaskStatus enum with all states
- AC-PI3.3: SendMessageRequest accepts message with parts
- AC-PI3.8: Task state maps to pipeline stage

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → A2A Task Lifecycle Mapping
"""

import pytest
from datetime import datetime
from pydantic import ValidationError


# =============================================================================
# AC-PI3.2: TaskStatus Enum
# =============================================================================


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_task_status_enum_exists(self) -> None:
        """TaskStatus enum can be imported."""
        from src.a2a.models import TaskStatus

        assert TaskStatus is not None

    def test_task_status_has_submitted(self) -> None:
        """TaskStatus has submitted state."""
        from src.a2a.models import TaskStatus

        assert hasattr(TaskStatus, "SUBMITTED")
        assert TaskStatus.SUBMITTED.value == "submitted"

    def test_task_status_has_working(self) -> None:
        """TaskStatus has working state."""
        from src.a2a.models import TaskStatus

        assert hasattr(TaskStatus, "WORKING")
        assert TaskStatus.WORKING.value == "working"

    def test_task_status_has_input_required(self) -> None:
        """TaskStatus has input-required state."""
        from src.a2a.models import TaskStatus

        assert hasattr(TaskStatus, "INPUT_REQUIRED")
        assert TaskStatus.INPUT_REQUIRED.value == "input-required"

    def test_task_status_has_completed(self) -> None:
        """TaskStatus has completed state."""
        from src.a2a.models import TaskStatus

        assert hasattr(TaskStatus, "COMPLETED")
        assert TaskStatus.COMPLETED.value == "completed"

    def test_task_status_has_failed(self) -> None:
        """TaskStatus has failed state."""
        from src.a2a.models import TaskStatus

        assert hasattr(TaskStatus, "FAILED")
        assert TaskStatus.FAILED.value == "failed"

    def test_task_status_has_canceled(self) -> None:
        """TaskStatus has canceled state."""
        from src.a2a.models import TaskStatus

        assert hasattr(TaskStatus, "CANCELED")
        assert TaskStatus.CANCELED.value == "canceled"


# =============================================================================
# AC-PI3.1: Part and Artifact Models
# =============================================================================


class TestPartModel:
    """Tests for Part pydantic model."""

    def test_part_model_exists(self) -> None:
        """Part model can be imported."""
        from src.a2a.models import Part

        assert isinstance(Part, type)

    def test_part_text_content(self) -> None:
        """Part can represent text content."""
        from src.a2a.models import Part

        part = Part(type="text", text="Hello, world!")

        assert part.type == "text"
        assert part.text == "Hello, world!"

    def test_part_data_content(self) -> None:
        """Part can represent data content."""
        from src.a2a.models import Part

        part = Part(type="data", data={"key": "value"})

        assert part.type == "data"
        assert part.data == {"key": "value"}


class TestArtifactModel:
    """Tests for Artifact pydantic model."""

    def test_artifact_model_exists(self) -> None:
        """Artifact model can be imported."""
        from src.a2a.models import Artifact

        assert isinstance(Artifact, type)

    def test_artifact_has_parts(self) -> None:
        """Artifact has parts list."""
        from src.a2a.models import Artifact, Part

        artifact = Artifact(
            parts=[Part(type="text", text="Result")]
        )

        assert len(artifact.parts) == 1
        assert artifact.parts[0].text == "Result"


# =============================================================================
# AC-PI3.3: Message and SendMessageRequest Models
# =============================================================================


class TestMessageModel:
    """Tests for Message pydantic model."""

    def test_message_model_exists(self) -> None:
        """Message model can be imported."""
        from src.a2a.models import Message

        assert isinstance(Message, type)

    def test_message_has_parts(self) -> None:
        """Message has parts list."""
        from src.a2a.models import Message, Part

        msg = Message(parts=[Part(type="text", text="Test message")])

        assert len(msg.parts) == 1
        assert msg.parts[0].text == "Test message"

    def test_message_has_optional_skill_id(self) -> None:
        """Message has optional skillId field."""
        from src.a2a.models import Message, Part

        msg = Message(
            parts=[Part(type="text", text="Test")],
            skillId="extract_structure"
        )

        assert msg.skillId == "extract_structure"


class TestSendMessageRequest:
    """Tests for SendMessageRequest model (AC-PI3.3)."""

    def test_send_message_request_exists(self) -> None:
        """SendMessageRequest model can be imported."""
        from src.a2a.models import SendMessageRequest

        assert isinstance(SendMessageRequest, type)

    def test_send_message_request_has_message(self) -> None:
        """SendMessageRequest has message field."""
        from src.a2a.models import SendMessageRequest, Message, Part

        request = SendMessageRequest(
            message=Message(parts=[Part(type="text", text="Execute task")])
        )

        assert request.message is not None
        assert len(request.message.parts) == 1

    def test_send_message_request_has_optional_context_id(self) -> None:
        """SendMessageRequest has optional contextId."""
        from src.a2a.models import SendMessageRequest, Message, Part

        request = SendMessageRequest(
            message=Message(parts=[Part(type="text", text="Test")]),
            contextId="ctx_123"
        )

        assert request.contextId == "ctx_123"


class TestSendMessageResponse:
    """Tests for SendMessageResponse model."""

    def test_send_message_response_exists(self) -> None:
        """SendMessageResponse model can be imported."""
        from src.a2a.models import SendMessageResponse

        assert isinstance(SendMessageResponse, type)

    def test_send_message_response_has_task_id(self) -> None:
        """SendMessageResponse has taskId field."""
        from src.a2a.models import SendMessageResponse

        response = SendMessageResponse(taskId="task_123")

        assert response.taskId == "task_123"


# =============================================================================
# AC-PI3.1: Task Model
# =============================================================================


class TestTaskModel:
    """Tests for Task pydantic model (AC-PI3.1)."""

    def test_task_model_exists(self) -> None:
        """Task model can be imported."""
        from src.a2a.models import Task

        assert isinstance(Task, type)

    def test_task_has_id(self) -> None:
        """Task has id field."""
        from src.a2a.models import Task, TaskStatus

        task = Task(
            id="task_123",
            status=TaskStatus.SUBMITTED,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        assert task.id == "task_123"

    def test_task_has_status(self) -> None:
        """Task has status field."""
        from src.a2a.models import Task, TaskStatus

        task = Task(
            id="task_123",
            status=TaskStatus.WORKING,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        assert task.status == TaskStatus.WORKING

    def test_task_has_artifacts(self) -> None:
        """Task has artifacts list (AC-PI3.1)."""
        from src.a2a.models import Task, TaskStatus, Artifact, Part

        task = Task(
            id="task_123",
            status=TaskStatus.COMPLETED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            artifacts=[Artifact(parts=[Part(type="text", text="Result")])]
        )

        assert len(task.artifacts) == 1
        assert task.artifacts[0].parts[0].text == "Result"

    def test_task_has_history(self) -> None:
        """Task has history list (AC-PI3.1)."""
        from src.a2a.models import Task, TaskStatus

        task = Task(
            id="task_123",
            status=TaskStatus.COMPLETED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            history=["submitted", "working", "completed"]
        )

        assert len(task.history) == 3
        assert "working" in task.history

    def test_task_has_timestamps(self) -> None:
        """Task has created_at and updated_at timestamps."""
        from src.a2a.models import Task, TaskStatus

        now = datetime.now()
        task = Task(
            id="task_123",
            status=TaskStatus.SUBMITTED,
            created_at=now,
            updated_at=now
        )

        assert task.created_at == now
        assert task.updated_at == now

    def test_task_has_optional_skill_id(self) -> None:
        """Task has optional skillId field."""
        from src.a2a.models import Task, TaskStatus

        task = Task(
            id="task_123",
            status=TaskStatus.SUBMITTED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            skillId="extract_structure"
        )

        assert task.skillId == "extract_structure"

    def test_task_has_optional_error(self) -> None:
        """Task has optional error field for failed tasks."""
        from src.a2a.models import Task, TaskStatus

        task = Task(
            id="task_123",
            status=TaskStatus.FAILED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            error="Execution failed: Invalid input"
        )

        assert task.error == "Execution failed: Invalid input"

    def test_task_artifacts_defaults_to_empty(self) -> None:
        """Task artifacts defaults to empty list."""
        from src.a2a.models import Task, TaskStatus

        task = Task(
            id="task_123",
            status=TaskStatus.SUBMITTED,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        assert task.artifacts == []

    def test_task_history_defaults_to_empty(self) -> None:
        """Task history defaults to empty list."""
        from src.a2a.models import Task, TaskStatus

        task = Task(
            id="task_123",
            status=TaskStatus.SUBMITTED,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        assert task.history == []
