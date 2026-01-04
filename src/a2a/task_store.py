"""TaskStore - In-memory A2A task storage.

WBS-PI3: A2A Task Lifecycle & Endpoints - TaskStore
AC-PI3.9: In-memory task store with CRUD operations

Implements:
- Task creation with unique IDs
- Task retrieval by ID
- Task updates (status, artifacts, history, error)
- Task deletion
- Task listing
- Thread-safe operations using threading.Lock

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → Task State Management
"""

import threading
import uuid
from datetime import datetime

from src.a2a.models import Artifact, Task, TaskStatus


class TaskStore:
    """In-memory task storage with thread-safe CRUD operations (AC-PI3.9).
    
    Provides create, read, update, delete, and list operations for A2A tasks.
    Uses a threading lock to ensure thread-safe concurrent access.
    
    Attributes:
        _tasks: Internal dictionary mapping task IDs to Task objects
        _lock: Threading lock for thread-safe operations
    """

    def __init__(self) -> None:
        """Initialize empty task store with thread lock."""
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def create_task(
        self,
        status: TaskStatus,
        skill_id: str | None = None,
    ) -> Task:
        """Create a new task with unique ID.
        
        Args:
            status: Initial task status (typically SUBMITTED)
            skill_id: Optional skill identifier
            
        Returns:
            Created Task instance with unique ID and timestamps
        """
        with self._lock:
            task_id = str(uuid.uuid4())
            now = datetime.now()
            
            task = Task(
                id=task_id,
                status=status,
                created_at=now,
                updated_at=now,
                artifacts=[],
                history=[],
                skillId=skill_id,  # Map to model's camelCase field
                error=None,
            )
            
            self._tasks[task_id] = task
            return task

    def get_task(self, task_id: str) -> Task | None:
        """Retrieve task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task if found, None otherwise
        """
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(
        self,
        task_id: str,
        status: TaskStatus | None = None,
        artifacts: list[Artifact] | None = None,
        history_entry: str | None = None,
        error: str | None = None,
    ) -> Task | None:
        """Update task fields.
        
        Args:
            task_id: Task identifier
            status: New task status (if provided)
            artifacts: New artifacts list (if provided)
            history_entry: History entry to append (if provided)
            error: Error message (if provided)
            
        Returns:
            Updated Task if found, None otherwise
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            
            # Update fields
            if status is not None:
                task.status = status
            
            if artifacts is not None:
                task.artifacts = artifacts
            
            if history_entry is not None:
                task.history.append(history_entry)
            
            if error is not None:
                task.error = error
            
            # Update timestamp
            task.updated_at = datetime.now()
            
            return task

    def delete_task(self, task_id: str) -> bool:
        """Delete task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was deleted, False if not found
        """
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
            return False

    def list_tasks(self) -> list[Task]:
        """List all tasks.
        
        Returns:
            List of all tasks in the store
        """
        with self._lock:
            return list(self._tasks.values())


__all__ = ["TaskStore"]
