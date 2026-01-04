"""Integration tests for A2A Streaming (SSE).

WBS-PI4: A2A Streaming (SSE) - Integration Tests

Tests end-to-end SSE streaming from HTTP request through task lifecycle
to event emission. Verifies:
- Feature flag enforcement (501 when disabled)
- SSE stream format compliance
- Event ordering and sequencing
- Clean stream closure

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → A2A Streaming
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi import status
from fastapi.testclient import TestClient

from src.main import app


class TestA2AStreamingIntegration(unittest.TestCase):
    """Integration tests for A2A streaming endpoint."""

    def setUp(self) -> None:
        """Set up test client."""
        self.client = TestClient(app)

    def tearDown(self) -> None:
        """Clean up test resources."""
        app.dependency_overrides.clear()

    # =============================================================================
    # Integration Tests: Feature Flag Enforcement
    # =============================================================================

    def test_streaming_returns_501_when_a2a_disabled(self) -> None:
        """Streaming returns 501 when A2A protocol disabled."""
        from src.api.routes import a2a
        from src.config.feature_flags import ProtocolFeatureFlags

        # Override feature flags to disable A2A
        mock_flags = ProtocolFeatureFlags(a2a_enabled=False)
        app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Generate code"}]}
            }
        )
        
        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        assert "not enabled" in response.json()["detail"].lower()

    def test_streaming_returns_501_when_streaming_disabled(self) -> None:
        """Streaming returns 501 when streaming feature disabled."""
        from src.api.routes import a2a
        from src.config.feature_flags import ProtocolFeatureFlags

        # Override feature flags to disable streaming
        mock_flags = ProtocolFeatureFlags(
            a2a_enabled=True,
            a2a_task_lifecycle_enabled=True,
            a2a_streaming_enabled=False  # Disable streaming
        )
        app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Generate code"}]}
            }
        )
        
        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        assert "streaming not enabled" in response.json()["detail"].lower()

    # =============================================================================
    # Integration Tests: End-to-End SSE Streaming
    # =============================================================================

    def test_stream_emits_valid_sse_events(self) -> None:
        """Stream emits valid SSE-formatted events."""
        from src.api.routes import a2a
        from src.config.feature_flags import ProtocolFeatureFlags

        # Enable all A2A flags
        mock_flags = ProtocolFeatureFlags(
            a2a_enabled=True,
            a2a_task_lifecycle_enabled=True,
            a2a_streaming_enabled=True
        )
        app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Generate code"}]}
            }
        )
        
        # Verify response format
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        
        # Read and parse SSE stream
        events = []
        for chunk in response.iter_text():
            if chunk.strip():
                events.append(chunk)
        
        # Verify SSE format compliance
        assert len(events) > 0
        for event_str in events:
            assert "event: status_update" in event_str
            assert "data:" in event_str

    def test_stream_emits_lifecycle_events_in_order(self) -> None:
        """Stream emits SUBMITTED → WORKING → COMPLETED events."""
        from src.api.routes import a2a
        from src.config.feature_flags import ProtocolFeatureFlags

        # Enable all A2A flags
        mock_flags = ProtocolFeatureFlags(
            a2a_enabled=True,
            a2a_task_lifecycle_enabled=True,
            a2a_streaming_enabled=True
        )
        app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Generate code"}]}
            }
        )
        
        # Collect all events
        raw_events = []
        for chunk in response.iter_text():
            if "data:" in chunk:
                # Extract JSON data from SSE format
                lines = chunk.strip().split("\n")
                for line in lines:
                    if line.startswith("data:"):
                        data_json = line[5:].strip()  # Remove "data:" prefix
                        raw_events.append(json.loads(data_json))
        
        # Verify event sequence
        assert len(raw_events) >= 3  # At least SUBMITTED, WORKING, COMPLETED
        
        # Check first 3 events (basic lifecycle)
        assert raw_events[0]["status"] == "submitted"
        assert raw_events[1]["status"] == "working"
        assert raw_events[2]["status"] == "completed"
        
        # Verify sequence numbers increment
        assert raw_events[0]["sequence"] == 1
        assert raw_events[1]["sequence"] == 2
        assert raw_events[2]["sequence"] == 3

    def test_stream_events_have_consistent_task_id(self) -> None:
        """All events in stream reference the same task_id."""
        from src.api.routes import a2a
        from src.config.feature_flags import ProtocolFeatureFlags

        # Enable all A2A flags
        mock_flags = ProtocolFeatureFlags(
            a2a_enabled=True,
            a2a_task_lifecycle_enabled=True,
            a2a_streaming_enabled=True
        )
        app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Generate code"}]}
            }
        )
        
        # Collect all events
        raw_events = []
        for chunk in response.iter_text():
            if "data:" in chunk:
                lines = chunk.strip().split("\n")
                for line in lines:
                    if line.startswith("data:"):
                        data_json = line[5:].strip()
                        raw_events.append(json.loads(data_json))
        
        # Verify all events have same task_id
        assert len(raw_events) > 0
        task_ids = {event["task_id"] for event in raw_events}
        assert len(task_ids) == 1  # Only one unique task_id

    def test_stream_events_have_iso_timestamps(self) -> None:
        """All events include valid ISO 8601 timestamps."""
        from src.api.routes import a2a
        from src.config.feature_flags import ProtocolFeatureFlags

        # Enable all A2A flags
        mock_flags = ProtocolFeatureFlags(
            a2a_enabled=True,
            a2a_task_lifecycle_enabled=True,
            a2a_streaming_enabled=True
        )
        app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Generate code"}]}
            }
        )
        
        # Collect all events
        raw_events = []
        for chunk in response.iter_text():
            if "data:" in chunk:
                lines = chunk.strip().split("\n")
                for line in lines:
                    if line.startswith("data:"):
                        data_json = line[5:].strip()
                        raw_events.append(json.loads(data_json))
        
        # Verify timestamps
        assert len(raw_events) > 0
        for event in raw_events:
            assert "timestamp" in event
            # Verify ISO 8601 format (basic check for 'T' and 'Z')
            assert "T" in event["timestamp"]
            assert event["timestamp"].endswith("Z") or "+" in event["timestamp"]

    def test_stream_closes_cleanly_without_error(self) -> None:
        """Stream completes without exceptions."""
        from src.api.routes import a2a
        from src.config.feature_flags import ProtocolFeatureFlags

        # Enable all A2A flags
        mock_flags = ProtocolFeatureFlags(
            a2a_enabled=True,
            a2a_task_lifecycle_enabled=True,
            a2a_streaming_enabled=True
        )
        app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Generate code"}]}
            }
        )
        
        # Consume entire stream (should not raise exception)
        all_chunks = list(response.iter_text())
        
        # Stream should have content
        assert len(all_chunks) > 0
        
        # Verify last event is COMPLETED
        completed_found = False
        for chunk in all_chunks:
            if "completed" in chunk.lower():
                completed_found = True
        assert completed_found

    # =============================================================================
    # Integration Tests: SSE Response Headers
    # =============================================================================

    def test_stream_response_has_sse_headers(self) -> None:
        """Stream response includes proper SSE headers."""
        from src.api.routes import a2a
        from src.config.feature_flags import ProtocolFeatureFlags

        # Enable all A2A flags
        mock_flags = ProtocolFeatureFlags(
            a2a_enabled=True,
            a2a_task_lifecycle_enabled=True,
            a2a_streaming_enabled=True
        )
        app.dependency_overrides[a2a.get_feature_flags] = lambda: mock_flags

        response = self.client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {"parts": [{"type": "text", "text": "Generate code"}]}
            }
        )
        
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["connection"] == "keep-alive"


if __name__ == "__main__":
    unittest.main()
