"""Unit Tests for A2A Router Endpoints.

WBS-PI3: A2A Task Lifecycle & Endpoints - A2A Router
AC-PI3.4: POST /a2a/v1/message:send endpoint
AC-PI3.5: GET /a2a/v1/tasks/{id} endpoint
AC-PI3.6: POST /a2a/v1/tasks/{id}:cancel endpoint
AC-PI3.7: Feature flag guards return 501 when disabled

Test Coverage:
- Feature flag guards (501 when disabled, 404/200 when enabled)
- SendMessage endpoint creates task
- GetTask endpoint retrieves task by ID
- CancelTask endpoint cancels running task
- Error handling (404 for nonexistent tasks)
- Request/response model validation
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestA2ARouter(unittest.TestCase):
    """Test suite for A2A Router endpoints (AC-PI3.4-7)."""

    # =============================================================================
    # Setup and Teardown
    # =============================================================================

    def setUp(self) -> None:
        """Set up test fixtures."""
        from src.main import create_app
        from src.api.routes.a2a import get_feature_flags, get_task_store
        
        self.app = create_app()
        self.client = TestClient(self.app)
        
        # Store original dependencies for restoration
        self._original_get_flags = get_feature_flags
        self._original_get_store = get_task_store
    
    def tearDown(self) -> None:
        """Clean up test fixtures."""
        # Restore original dependencies
        from src.api.routes import a2a
        a2a.get_feature_flags = self._original_get_flags
        a2a.get_task_store = self._original_get_store

    # =============================================================================
    # Feature Flag Guard Tests (AC-PI3.7)
    # =============================================================================

    def test_send_message_returns_501_when_a2a_disabled(
        self
    ) -> None:
        """POST /a2a/v1/message:send returns 501 when A2A disabled."""
        from src.api.routes import a2a
        
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = False
        mock_flags.a2a_task_lifecycle_enabled = True
        
        # Override dependency
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.post(
            "/a2a/v1/message:send",
            json={
                "message": {"parts": [{"type": "text", "text": "Hello"}]}
            }
        )

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        
        # Cleanup
        self.app.dependency_overrides.clear()

    def test_send_message_returns_501_when_task_lifecycle_disabled(
        self
    ) -> None:
        """POST /a2a/v1/message:send returns 501 when task lifecycle disabled."""
        from src.api.routes import a2a
        
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = True
        mock_flags.a2a_task_lifecycle_enabled = False
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.post(
            "/a2a/v1/message:send",
            json={
                "message": {"parts": [{"type": "text", "text": "Hello"}]}
            }
        )

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        self.app.dependency_overrides.clear()

    def test_get_task_returns_501_when_a2a_disabled(
        self
    ) -> None:
        """GET /a2a/v1/tasks/{id} returns 501 when A2A disabled."""
        from src.api.routes import a2a
        
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = False
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.get("/a2a/v1/tasks/test-id")

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        self.app.dependency_overrides.clear()

    def test_cancel_task_returns_501_when_a2a_disabled(
        self
    ) -> None:
        """POST /a2a/v1/tasks/{id}:cancel returns 501 when A2A disabled."""
        from src.api.routes import a2a
        
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = False
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.post("/a2a/v1/tasks/test-id:cancel")

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        self.app.dependency_overrides.clear()

    # =============================================================================
    # SendMessage Endpoint Tests (AC-PI3.4)
    # =============================================================================

    def test_send_message_creates_task(
        self
    ) -> None:
        """POST /a2a/v1/message:send creates task and returns taskId."""
        from src.api.routes import a2a
        
        # Setup mocks
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = True
        mock_flags.a2a_task_lifecycle_enabled = True

        mock_store = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "test-task-id"
        mock_store.create_task.return_value = mock_task
        
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags
        self.app.dependency_overrides[a2a.get_task_store] = lambda: mock_store

        # Make request
        response = self.client.post(
            "/a2a/v1/message:send",
            json={
                "message": {"parts": [{"type": "text", "text": "Generate code"}]}
            }
        )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "taskId" in data
        assert data["taskId"] == "test-task-id"
        
        self.app.dependency_overrides.clear()

    def test_send_message_with_skill_id(
        self
    ) -> None:
        """POST /a2a/v1/message:send with skillId creates task with skill."""
        from src.api.routes import a2a
        
        # Setup mocks
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = True
        mock_flags.a2a_task_lifecycle_enabled = True

        mock_store = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "test-task-id"
        mock_store.create_task.return_value = mock_task
        
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags
        self.app.dependency_overrides[a2a.get_task_store] = lambda: mock_store

        # Make request
        response = self.client.post(
            "/a2a/v1/message:send",
            json={
                "message": {
                    "parts": [{"type": "text", "text": "Generate code"}],
                    "skillId": "generate_code"
                }
            }
        )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        self.app.dependency_overrides.clear()

    def test_send_message_with_context_id(
        self
    ) -> None:
        """POST /a2a/v1/message:send with contextId passes context."""
        from src.api.routes import a2a
        
        # Setup mocks
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = True
        mock_flags.a2a_task_lifecycle_enabled = True

        mock_store = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "test-task-id"
        mock_store.create_task.return_value = mock_task
        
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags
        self.app.dependency_overrides[a2a.get_task_store] = lambda: mock_store

        # Make request
        response = self.client.post(
            "/a2a/v1/message:send",
            json={
                "message": {"parts": [{"type": "text", "text": "Continue"}]},
                "contextId": "previous-task-id"
            }
        )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        self.app.dependency_overrides.clear()

    # =============================================================================
    # GetTask Endpoint Tests (AC-PI3.5)
    # =============================================================================

    def test_get_task_returns_task(
        self
    ) -> None:
        """GET /a2a/v1/tasks/{id} returns task by ID."""
        from src.a2a.models import TaskStatus
        from src.api.routes import a2a
        from datetime import datetime

        # Setup mocks
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = True
        mock_flags.a2a_task_lifecycle_enabled = True

        mock_store = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "test-task-id"
        mock_task.status = TaskStatus.WORKING
        mock_task.created_at = datetime.now()
        mock_task.updated_at = datetime.now()
        mock_task.artifacts = []
        mock_task.history = []
        mock_task.skillId = None
        mock_task.error = None
        mock_task.model_dump.return_value = {
            "id": "test-task-id",
            "status": "working",
            "created_at": mock_task.created_at.isoformat(),
            "updated_at": mock_task.updated_at.isoformat(),
            "artifacts": [],
            "history": [],
            "skillId": None,
            "error": None,
        }
        mock_store.get_task.return_value = mock_task
        
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags
        self.app.dependency_overrides[a2a.get_task_store] = lambda: mock_store

        # Make request
        response = self.client.get("/a2a/v1/tasks/test-task-id")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "test-task-id"
        assert data["status"] == "working"
        
        self.app.dependency_overrides.clear()

    def test_get_task_returns_404_for_nonexistent_id(
        self
    ) -> None:
        """GET /a2a/v1/tasks/{id} returns 404 for nonexistent ID."""
        from src.api.routes import a2a
        
        # Setup mocks
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = True
        mock_flags.a2a_task_lifecycle_enabled = True

        mock_store = MagicMock()
        mock_store.get_task.return_value = None
        
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags
        self.app.dependency_overrides[a2a.get_task_store] = lambda: mock_store

        # Make request
        response = self.client.get("/a2a/v1/tasks/nonexistent-id")

        # Verify response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        self.app.dependency_overrides.clear()

    # =============================================================================
    # CancelTask Endpoint Tests (AC-PI3.6)
    # =============================================================================

    def test_cancel_task_updates_status_to_canceled(
        self
    ) -> None:
        """POST /a2a/v1/tasks/{id}:cancel updates task status to CANCELED."""
        from src.a2a.models import Task, TaskStatus
        from src.api.routes import a2a
        from datetime import datetime

        # Setup mocks
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = True
        mock_flags.a2a_task_lifecycle_enabled = True

        mock_store = MagicMock()
        # Return a real Task object instead of MagicMock
        mock_task = Task(
            id="test-task-id",
            status=TaskStatus.CANCELED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            artifacts=[],
            history=["Task canceled at canceled"],
            skillId=None,
            error=None,
        )
        mock_store.update_task.return_value = mock_task
        
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags
        self.app.dependency_overrides[a2a.get_task_store] = lambda: mock_store

        # Make request
        response = self.client.post("/a2a/v1/tasks/test-task-id:cancel")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "canceled"
        
        self.app.dependency_overrides.clear()

    def test_cancel_task_returns_404_for_nonexistent_id(
        self
    ) -> None:
        """POST /a2a/v1/tasks/{id}:cancel returns 404 for nonexistent ID."""
        from src.api.routes import a2a
        
        # Setup mocks
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = True
        mock_flags.a2a_task_lifecycle_enabled = True

        mock_store = MagicMock()
        mock_store.update_task.return_value = None
        
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags
        self.app.dependency_overrides[a2a.get_task_store] = lambda: mock_store

        # Make request
        response = self.client.post("/a2a/v1/tasks/nonexistent-id:cancel")

        # Verify response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        self.app.dependency_overrides.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
