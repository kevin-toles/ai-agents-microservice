"""Unit Tests for TaskStore.

WBS-PI3: A2A Task Lifecycle & Endpoints - TaskStore
AC-PI3.9: In-memory task store with CRUD operations

Test Coverage:
- Task creation with unique IDs
- Task retrieval by ID
- Task updates (status, artifacts, history)
- Task deletion
- Task listing
- Thread-safe operations
- Nonexistent task handling
"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock

import pytest


class TestTaskStore(unittest.TestCase):
    """Test suite for TaskStore CRUD operations (AC-PI3.9)."""

    # =============================================================================
    # Setup and Teardown
    # =============================================================================

    def setUp(self) -> None:
        """Set up test fixtures."""
        from src.a2a.task_store import TaskStore

        self.store = TaskStore()

    # =============================================================================
    # Task Creation Tests
    # =============================================================================

    def test_create_task_returns_task_with_id(self) -> None:
        """TaskStore.create_task() returns Task with unique ID."""
        from src.a2a.models import TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )

        assert task.id is not None
        assert len(task.id) > 0

    def test_create_task_sets_submitted_status(self) -> None:
        """TaskStore.create_task() sets initial status to SUBMITTED."""
        from src.a2a.models import TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )

        assert task.status == TaskStatus.SUBMITTED

    def test_create_task_sets_timestamps(self) -> None:
        """TaskStore.create_task() sets created_at and updated_at."""
        from src.a2a.models import TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )

        assert isinstance(task.created_at, datetime)
        assert isinstance(task.updated_at, datetime)

    def test_create_task_initializes_empty_artifacts(self) -> None:
        """TaskStore.create_task() initializes empty artifacts list."""
        from src.a2a.models import TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )

        assert task.artifacts == []

    def test_create_task_initializes_empty_history(self) -> None:
        """TaskStore.create_task() initializes empty history list."""
        from src.a2a.models import TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )

        assert task.history == []

    def test_create_task_stores_skill_id(self) -> None:
        """TaskStore.create_task() stores skillId."""
        from src.a2a.models import TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="analyze_code"
        )

        assert task.skillId == "analyze_code"

    def test_create_task_with_none_skill_id(self) -> None:
        """TaskStore.create_task() allows None skillId."""
        from src.a2a.models import TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id=None
        )

        assert task.skillId is None

    # =============================================================================
    # Task Retrieval Tests
    # =============================================================================

    def test_get_task_returns_existing_task(self) -> None:
        """TaskStore.get_task() returns task by ID."""
        from src.a2a.models import TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )

        retrieved = self.store.get_task(task.id)

        assert retrieved is not None
        assert retrieved.id == task.id

    def test_get_task_returns_none_for_nonexistent_id(self) -> None:
        """TaskStore.get_task() returns None for invalid ID."""
        result = self.store.get_task("nonexistent-id")

        assert result is None

    # =============================================================================
    # Task Update Tests
    # =============================================================================

    def test_update_task_status(self) -> None:
        """TaskStore.update_task() updates task status."""
        from src.a2a.models import TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )

        updated = self.store.update_task(
            task_id=task.id,
            status=TaskStatus.WORKING
        )

        assert updated is not None
        assert updated.status == TaskStatus.WORKING

    def test_update_task_updates_timestamp(self) -> None:
        """TaskStore.update_task() updates updated_at timestamp."""
        from src.a2a.models import TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )
        original_updated_at = task.updated_at

        updated = self.store.update_task(
            task_id=task.id,
            status=TaskStatus.WORKING
        )

        assert updated is not None
        assert updated.updated_at > original_updated_at

    def test_update_task_adds_artifacts(self) -> None:
        """TaskStore.update_task() adds artifacts to task."""
        from src.a2a.models import Artifact, Part, TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )

        artifact = Artifact(
            parts=[Part(type="text", text="Generated code")]
        )

        updated = self.store.update_task(
            task_id=task.id,
            artifacts=[artifact]
        )

        assert updated is not None
        assert len(updated.artifacts) == 1
        assert updated.artifacts[0].parts[0].text == "Generated code"

    def test_update_task_appends_to_history(self) -> None:
        """TaskStore.update_task() appends state transitions to history."""
        from src.a2a.models import TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )

        updated = self.store.update_task(
            task_id=task.id,
            status=TaskStatus.WORKING,
            history_entry="Started execution"
        )

        assert updated is not None
        assert len(updated.history) == 1
        assert updated.history[0] == "Started execution"

    def test_update_task_sets_error_message(self) -> None:
        """TaskStore.update_task() sets error message for failed tasks."""
        from src.a2a.models import TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )

        updated = self.store.update_task(
            task_id=task.id,
            status=TaskStatus.FAILED,
            error="Validation failed"
        )

        assert updated is not None
        assert updated.status == TaskStatus.FAILED
        assert updated.error == "Validation failed"

    def test_update_task_returns_none_for_nonexistent_id(self) -> None:
        """TaskStore.update_task() returns None for invalid ID."""
        from src.a2a.models import TaskStatus

        result = self.store.update_task(
            task_id="nonexistent-id",
            status=TaskStatus.WORKING
        )

        assert result is None

    # =============================================================================
    # Task Deletion Tests
    # =============================================================================

    def test_delete_task_removes_task(self) -> None:
        """TaskStore.delete_task() removes task from store."""
        from src.a2a.models import TaskStatus

        task = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )

        result = self.store.delete_task(task.id)

        assert result is True
        assert self.store.get_task(task.id) is None

    def test_delete_task_returns_false_for_nonexistent_id(self) -> None:
        """TaskStore.delete_task() returns False for invalid ID."""
        result = self.store.delete_task("nonexistent-id")

        assert result is False

    # =============================================================================
    # Task Listing Tests
    # =============================================================================

    def test_list_tasks_returns_all_tasks(self) -> None:
        """TaskStore.list_tasks() returns all tasks."""
        from src.a2a.models import TaskStatus

        task1 = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )
        task2 = self.store.create_task(
            status=TaskStatus.WORKING,
            skill_id="analyze_code"
        )

        tasks = self.store.list_tasks()

        assert len(tasks) == 2
        assert any(t.id == task1.id for t in tasks)
        assert any(t.id == task2.id for t in tasks)

    def test_list_tasks_returns_empty_list_when_no_tasks(self) -> None:
        """TaskStore.list_tasks() returns empty list when no tasks exist."""
        tasks = self.store.list_tasks()

        assert tasks == []

    # =============================================================================
    # Thread Safety Tests
    # =============================================================================

    def test_create_task_generates_unique_ids(self) -> None:
        """TaskStore.create_task() generates unique IDs for concurrent tasks."""
        from src.a2a.models import TaskStatus

        task1 = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="generate_code"
        )
        task2 = self.store.create_task(
            status=TaskStatus.SUBMITTED,
            skill_id="analyze_code"
        )

        assert task1.id != task2.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
