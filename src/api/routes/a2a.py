"""A2A Protocol Router.

WBS-PI3: A2A Task Lifecycle & Endpoints - A2A Router
AC-PI3.4: POST /a2a/v1/message:send endpoint
AC-PI3.5: GET /a2a/v1/tasks/{id} endpoint
AC-PI3.6: POST /a2a/v1/tasks/{id}:cancel endpoint
AC-PI3.7: Feature flag guards return 501 when disabled

WBS-PI4: A2A Streaming (SSE)
AC-PI4.1: POST /a2a/v1/message:stream endpoint
AC-PI4.2: Stream emits TaskStatusUpdateEvent
AC-PI4.3: Stream emits TaskArtifactUpdateEvent
AC-PI4.4: Stream closes cleanly
AC-PI4.6: Returns 501 when streaming disabled

Implements:
- POST /a2a/v1/message:send - Create task from message
- GET /a2a/v1/tasks/{id} - Retrieve task by ID
- POST /a2a/v1/tasks/{id}:cancel - Cancel running task
- POST /a2a/v1/message:stream - Stream task events via SSE
- Feature flag guards for a2a_enabled, a2a_task_lifecycle_enabled, a2a_streaming_enabled

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → A2A Endpoints
"""

from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.a2a.events import TaskStatusUpdateEvent
from src.a2a.models import SendMessageRequest, SendMessageResponse, Task, TaskStatus
from src.a2a.task_store import TaskStore
from src.config.feature_flags import ProtocolFeatureFlags, get_feature_flags

router = APIRouter(
    prefix="/a2a/v1",
    tags=["A2A Protocol"],
)


# =============================================================================
# Dependency Injection
# =============================================================================


# Global TaskStore singleton
_task_store: TaskStore | None = None


def get_task_store() -> TaskStore:
    """Get TaskStore singleton instance.
    
    Returns:
        TaskStore instance
    """
    global _task_store
    if _task_store is None:
        _task_store = TaskStore()
    return _task_store


# =============================================================================
# Feature Flag Guard
# =============================================================================


def check_a2a_enabled(
    flags: ProtocolFeatureFlags = Depends(get_feature_flags)
) -> None:
    """Check if A2A Protocol is enabled (AC-PI3.7).
    
    Args:
        flags: Feature flags instance
        
    Raises:
        HTTPException: 501 if A2A or task lifecycle disabled
    """
    if not flags.a2a_enabled or not flags.a2a_task_lifecycle_enabled:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="A2A Protocol task lifecycle not enabled"
        )


# =============================================================================
# AC-PI3.4: POST /a2a/v1/message:send
# =============================================================================


@router.post(
    "/message:send",
    response_model=SendMessageResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_a2a_enabled)],
)
def send_message(
    request: SendMessageRequest,
    store: TaskStore = Depends(get_task_store),
) -> SendMessageResponse:
    """Create task from A2A message (AC-PI3.4).
    
    Args:
        request: SendMessageRequest with message and optional contextId
        store: TaskStore instance
        
    Returns:
        SendMessageResponse with taskId
    """
    # Create task in SUBMITTED state
    task = store.create_task(
        status=TaskStatus.SUBMITTED,
        skill_id=request.message.skillId,
    )
    
    # TODO: Queue task for background processing (WBS-PI4)
    # For now, task remains in SUBMITTED state
    
    return SendMessageResponse(taskId=task.id)


# =============================================================================
# AC-PI3.5: GET /a2a/v1/tasks/{id}
# =============================================================================


@router.get(
    "/tasks/{task_id}",
    response_model=Task,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_a2a_enabled)],
)
def get_task(
    task_id: str,
    store: TaskStore = Depends(get_task_store),
) -> Task:
    """Retrieve task by ID (AC-PI3.5).
    
    Args:
        task_id: Task identifier
        store: TaskStore instance
        
    Returns:
        Task instance
        
    Raises:
        HTTPException: 404 if task not found
    """
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    return task


# =============================================================================
# AC-PI3.6: POST /a2a/v1/tasks/{id}:cancel
# =============================================================================


@router.post(
    "/tasks/{task_id}:cancel",
    response_model=Task,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_a2a_enabled)],
)
def cancel_task(
    task_id: str,
    store: TaskStore = Depends(get_task_store),
) -> Task:
    """Cancel running task (AC-PI3.6).
    
    Args:
        task_id: Task identifier
        store: TaskStore instance
        
    Returns:
        Updated Task with status=CANCELED
        
    Raises:
        HTTPException: 404 if task not found
    """
    task = store.update_task(
        task_id=task_id,
        status=TaskStatus.CANCELED,
        history_entry=f"Task canceled at {TaskStatus.CANCELED.value}"
    )
    
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    return task


# =============================================================================
# AC-PI4.1: POST /a2a/v1/message:stream
# =============================================================================


def check_a2a_streaming_enabled(
    flags: ProtocolFeatureFlags = Depends(get_feature_flags)
) -> None:
    """Check if A2A Streaming is enabled (AC-PI4.6).
    
    Args:
        flags: Feature flags instance
        
    Raises:
        HTTPException: 501 if A2A or streaming disabled
    """
    if not flags.a2a_enabled or not flags.a2a_streaming_enabled:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="A2A streaming not enabled"
        )


@router.post(
    "/message:stream",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_a2a_streaming_enabled)],
)
async def stream_message(
    request: SendMessageRequest,
    store: TaskStore = Depends(get_task_store),
) -> StreamingResponse:
    """Stream task events via SSE (AC-PI4.1).
    
    Args:
        request: SendMessageRequest with message
        store: TaskStore instance
        
    Returns:
        StreamingResponse with text/event-stream content type
    """
    def create_status_event(task_id: str, status: TaskStatus, sequence: int) -> TaskStatusUpdateEvent:
        """Create a TaskStatusUpdateEvent with current timestamp.
        
        Args:
            task_id: Task identifier
            status: Task status
            sequence: Event sequence number
            
        Returns:
            TaskStatusUpdateEvent instance
        """
        return TaskStatusUpdateEvent(
            task_id=task_id,
            status=status,
            timestamp=datetime.now(timezone.utc),
            sequence=sequence
        )
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for task lifecycle (AC-PI4.2, AC-PI4.4).
        
        Yields:
            SSE formatted event strings
        """
        # Create task in SUBMITTED state
        task = store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id=request.message.skillId,
        )
        
        # AC-PI4.2: Emit initial status update event
        sequence = 1
        yield create_status_event(task.id, TaskStatus.SUBMITTED, sequence).to_sse()
        
        # TODO: Async task execution will be added in future WBS
        # For WBS-PI4, we emit WORKING → COMPLETED immediately
        
        # Emit WORKING status
        sequence += 1
        store.update_task(
            task_id=task.id,
            status=TaskStatus.WORKING,
            history_entry="Task processing started"
        )
        yield create_status_event(task.id, TaskStatus.WORKING, sequence).to_sse()
        
        # Emit COMPLETED status (AC-PI4.4: Stream closes cleanly)
        sequence += 1
        store.update_task(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            history_entry="Task completed successfully"
        )
        yield create_status_event(task.id, TaskStatus.COMPLETED, sequence).to_sse()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


__all__ = ["router"]
