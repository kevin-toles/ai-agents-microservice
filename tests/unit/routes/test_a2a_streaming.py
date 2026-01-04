"""Unit Tests for A2A SSE Streaming Endpoint.

WBS-PI4: A2A Streaming (SSE) - SSE Endpoint
AC-PI4.1: POST /a2a/v1/message:stream returns SSE stream
AC-PI4.2: Stream emits TaskStatusUpdateEvent on state change
AC-PI4.3: Stream emits TaskArtifactUpdateEvent on artifact creation
AC-PI4.4: Stream closes cleanly on task completion
AC-PI4.5: Stream respects client disconnection
AC-PI4.6: Streaming disabled when a2a_streaming_enabled=false

Test Coverage:
- SSE endpoint returns StreamingResponse
- Feature flag guard returns 501 when disabled
- Stream emits events in SSE format
- Stream lifecycle (start, events, close)
- Client disconnection handling
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestA2AStreamingEndpoint(unittest.TestCase):
    """Test suite for A2A SSE streaming endpoint (AC-PI4.1-6)."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from src.main import create_app
        
        self.app = create_app()
        self.client = TestClient(self.app)

    # =============================================================================
    # Feature Flag Guard Tests (AC-PI4.6)
    # =============================================================================

    def test_stream_message_returns_501_when_a2a_disabled(self) -> None:
        """POST /a2a/v1/message:stream returns 501 when A2A disabled."""
        from src.api.routes import a2a
        
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = False
        mock_flags.a2a_task_lifecycle_enabled = True
        mock_flags.a2a_streaming_enabled = True
        
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Hello"}]}
            }
        )

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        self.app.dependency_overrides.clear()

    def test_stream_message_returns_501_when_streaming_disabled(self) -> None:
        """POST /a2a/v1/message:stream returns 501 when streaming disabled."""
        from src.api.routes import a2a
        
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = True
        mock_flags.a2a_task_lifecycle_enabled = True
        mock_flags.a2a_streaming_enabled = False
        
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Hello"}]}
            }
        )

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        self.app.dependency_overrides.clear()

    # =============================================================================
    # SSE Stream Tests (AC-PI4.1)
    # =============================================================================

    def test_stream_message_returns_streaming_response(self) -> None:
        """POST /a2a/v1/message:stream returns StreamingResponse."""
        from src.api.routes import a2a
        
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = True
        mock_flags.a2a_task_lifecycle_enabled = True
        mock_flags.a2a_streaming_enabled = True
        
        mock_store = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "test-task-id"
        mock_store.create_task.return_value = mock_task
        mock_store.update_task.return_value = mock_task  # update_task also returns task with same ID
        
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags
        self.app.dependency_overrides[a2a.get_task_store] = lambda: mock_store

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Generate code"}]}
            }
        )

        # Verify response type
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        self.app.dependency_overrides.clear()

    def test_stream_message_emits_sse_formatted_events(self) -> None:
        """POST /a2a/v1/message:stream emits SSE-formatted events."""
        from src.api.routes import a2a
        
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = True
        mock_flags.a2a_task_lifecycle_enabled = True
        mock_flags.a2a_streaming_enabled = True
        
        mock_store = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "test-task-id"
        mock_store.create_task.return_value = mock_task
        mock_store.update_task.return_value = mock_task
        
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags
        self.app.dependency_overrides[a2a.get_task_store] = lambda: mock_store

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Generate code"}]}
            }
        )

        # Read SSE stream
        chunks = []
        for chunk in response.iter_text():
            chunks.append(chunk)
        
        content = "".join(chunks)
        
        # SSE events should have "event:" and "data:" lines
        assert "event:" in content
        assert "data:" in content
        self.app.dependency_overrides.clear()

    # =============================================================================
    # Event Emission Tests (AC-PI4.2, AC-PI4.3)
    # =============================================================================

    def test_stream_emits_status_update_event(self) -> None:
        """Stream emits TaskStatusUpdateEvent on state change (AC-PI4.2)."""
        from src.api.routes import a2a
        
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = True
        mock_flags.a2a_task_lifecycle_enabled = True
        mock_flags.a2a_streaming_enabled = True
        
        mock_store = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "test-task-id"
        mock_store.create_task.return_value = mock_task
        mock_store.update_task.return_value = mock_task
        
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags
        self.app.dependency_overrides[a2a.get_task_store] = lambda: mock_store

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Generate code"}]}
            }
        )

        content = "".join(response.iter_text())
        
        # Should contain status update event
        assert "event: status_update" in content or "event:status_update" in content
        self.app.dependency_overrides.clear()

    # =============================================================================
    # Stream Lifecycle Tests (AC-PI4.4, AC-PI4.5)
    # =============================================================================

    def test_stream_closes_on_completion(self) -> None:
        """Stream closes cleanly when task completes (AC-PI4.4)."""
        from src.api.routes import a2a
        
        mock_flags = MagicMock()
        mock_flags.a2a_enabled = True
        mock_flags.a2a_task_lifecycle_enabled = True
        mock_flags.a2a_streaming_enabled = True
        
        mock_store = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "test-task-id"
        mock_store.create_task.return_value = mock_task
        mock_store.update_task.return_value = mock_task
        
        self.app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags
        self.app.dependency_overrides[a2a.get_task_store] = lambda: mock_store

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Generate code"}]}
            }
        )

        # Consume entire stream
        chunks = list(response.iter_text())
        
        # Stream should close (no exception raised)
        assert len(chunks) > 0
        self.app.dependency_overrides.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
