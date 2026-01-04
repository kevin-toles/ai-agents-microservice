"""Integration Tests for A2A Protocol (WBS-PI3).

Tests complete A2A task lifecycle end-to-end:
- Agent card discovery
- Message submission
- Task retrieval
- Task cancellation
- Feature flag guards

Reference: WBS_PROTOCOL_INTEGRATION.md → WBS-PI3
"""

import unittest

import pytest
from fastapi.testclient import TestClient


class TestA2AIntegration(unittest.TestCase):
    """End-to-end integration tests for A2A Protocol (WBS-PI3)."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        import os
        from src.config.feature_flags import get_feature_flags
        from src.main import create_app
        
        # Clear cache and enable A2A for integration tests
        get_feature_flags.cache_clear()
        os.environ["AGENTS_A2A_ENABLED"] = "true"
        os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "true"
        os.environ["AGENTS_A2A_TASK_LIFECYCLE_ENABLED"] = "true"
        
        self.app = create_app()
        self.client = TestClient(self.app)
    
    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import os
        from src.config.feature_flags import get_feature_flags
        
        os.environ.pop("AGENTS_A2A_ENABLED", None)
        os.environ.pop("AGENTS_A2A_AGENT_CARD_ENABLED", None)
        os.environ.pop("AGENTS_A2A_TASK_LIFECYCLE_ENABLED", None)
        get_feature_flags.cache_clear()

    def test_complete_task_lifecycle(self) -> None:
        """Test complete A2A task lifecycle: submit → retrieve → cancel."""
        # 1. Discover agent capabilities
        card_response = self.client.get("/.well-known/agent-card.json")
        assert card_response.status_code == 200
        card = card_response.json()
        assert card["protocolVersion"] == "0.3.0"
        assert len(card["skills"]) > 0
        
        # 2. Submit message to create task
        message_response = self.client.post(
            "/a2a/v1/message:send",
            json={
                "message": {
                    "parts": [
                        {"type": "text", "text": "Generate Python code for quicksort"}
                    ],
                    "skillId": "generate_code"
                }
            }
        )
        assert message_response.status_code == 200
        message_data = message_response.json()
        assert "taskId" in message_data
        task_id = message_data["taskId"]
        
        # 3. Retrieve task status
        get_response = self.client.get(f"/a2a/v1/tasks/{task_id}")
        assert get_response.status_code == 200
        task_data = get_response.json()
        assert task_data["id"] == task_id
        assert task_data["status"] in ["submitted", "working"]
        
        # 4. Cancel task
        cancel_response = self.client.post(f"/a2a/v1/tasks/{task_id}:cancel")
        assert cancel_response.status_code == 200
        canceled_data = cancel_response.json()
        assert canceled_data["status"] == "canceled"
        
        # 5. Verify canceled task still retrievable
        final_response = self.client.get(f"/a2a/v1/tasks/{task_id}")
        assert final_response.status_code == 200
        assert final_response.json()["status"] == "canceled"

    def test_feature_flags_disabled_returns_501(self) -> None:
        """Test endpoints return 501 when A2A disabled."""
        import os
        from src.config.feature_flags import get_feature_flags
        
        # Clear cache and disable A2A
        get_feature_flags.cache_clear()
        os.environ["AGENTS_A2A_ENABLED"] = "false"
        os.environ["AGENTS_A2A_TASK_LIFECYCLE_ENABLED"] = "false"
        
        # Create new app instance with disabled flags
        from src.main import create_app
        app = create_app()
        client = TestClient(app)
        
        # Agent card should return 404
        card_response = client.get("/.well-known/agent-card.json")
        assert card_response.status_code == 404
        
        # Task endpoints should return 501
        message_response = client.post(
            "/a2a/v1/message:send",
            json={"message": {"parts": [{"type": "text", "text": "test"}]}}
        )
        assert message_response.status_code == 501
        
        get_response = client.get("/a2a/v1/tasks/test-id")
        assert get_response.status_code == 501
        
        cancel_response = client.post("/a2a/v1/tasks/test-id:cancel")
        assert cancel_response.status_code == 501
        
        # Cleanup and restore
        os.environ["AGENTS_A2A_ENABLED"] = "true"
        os.environ["AGENTS_A2A_TASK_LIFECYCLE_ENABLED"] = "true"
        get_feature_flags.cache_clear()

    def test_nonexistent_task_returns_404(self) -> None:
        """Test retrieving nonexistent task returns 404."""
        response = self.client.get("/a2a/v1/tasks/nonexistent-task-id")
        assert response.status_code == 404

    def test_message_with_context_id(self) -> None:
        """Test message submission with contextId for conversation continuity."""
        # First message
        first_response = self.client.post(
            "/a2a/v1/message:send",
            json={
                "message": {
                    "parts": [{"type": "text", "text": "Analyze this code"}]
                }
            }
        )
        assert first_response.status_code == 200
        first_task_id = first_response.json()["taskId"]
        
        # Follow-up message with context
        second_response = self.client.post(
            "/a2a/v1/message:send",
            json={
                "message": {
                    "parts": [{"type": "text", "text": "Now improve it"}]
                },
                "contextId": first_task_id
            }
        )
        assert second_response.status_code == 200
        second_task_id = second_response.json()["taskId"]
        assert second_task_id != first_task_id  # Different tasks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
