"""Integration tests for MCP Server Lifecycle.

WBS-PI5b: MCP Server Lifecycle Integration
Test Coverage:
- AC-PI5b.1: MCP server integrates with FastAPI startup/shutdown
- AC-PI5b.2: MCP server disabled when `mcp_server_enabled=false`
- AC-PI5b.3: Feature flag guard prevents MCP exposure when disabled
- AC-PI5b.4: Server starts/stops cleanly with application

TDD Phase: RED - Tests written first, implementation to follow
"""

import pytest
from fastapi.testclient import TestClient

@pytest.mark.asyncio
class TestMCPLifecycleIntegration:
    """Test AC-PI5b.1, AC-PI5b.4: MCP server lifecycle with FastAPI."""
    
    async def test_mcp_server_starts_on_fastapi_startup_when_enabled(self, monkeypatch):
        """MCP server should start during FastAPI startup when flag enabled."""
        monkeypatch.setenv("AGENTS_MCP_ENABLED", "true")
        monkeypatch.setenv("AGENTS_MCP_SERVER_ENABLED", "true")
        
        # Import after setting env
        from src.main import create_app
        
        app = create_app()
        with TestClient(app) as client:
            # Server should be attached to app.state
            assert hasattr(app.state, "mcp_server")
            assert app.state.mcp_server is not None
            assert app.state.mcp_server["name"] == "ai-platform-agent-functions"
    
    async def test_mcp_server_not_started_when_disabled(self, monkeypatch):
        """MCP server should not start when flag disabled."""
        monkeypatch.setenv("AGENTS_MCP_ENABLED", "false")
        monkeypatch.setenv("AGENTS_MCP_SERVER_ENABLED", "false")
        
        from src.main import create_app
        
        app = create_app()
        with TestClient(app) as client:
            # Server should not be attached
            assert not hasattr(app.state, "mcp_server") or app.state.mcp_server is None
    
    async def test_mcp_server_stops_cleanly_on_shutdown(self, monkeypatch):
        """MCP server should clean up during FastAPI shutdown."""
        monkeypatch.setenv("AGENTS_MCP_ENABLED", "true")
        monkeypatch.setenv("AGENTS_MCP_SERVER_ENABLED", "true")
        
        from src.main import create_app
        
        app = create_app()
        with TestClient(app) as client:
            # Startup complete
            assert hasattr(app.state, "mcp_server")
        
        # After context exit, shutdown should have occurred without errors
        # No assertion needed - if shutdown raises, test fails


class TestMCPFeatureFlagGuard:
    """Test AC-PI5b.2, AC-PI5b.3: Feature flag guards."""
    
    def test_mcp_not_exposed_when_master_switch_disabled(self, monkeypatch):
        """MCP server should not be created when mcp_enabled=false."""
        monkeypatch.setenv("AGENTS_MCP_ENABLED", "false")
        monkeypatch.setenv("AGENTS_MCP_SERVER_ENABLED", "true")  # Sub-feature enabled
        
        from src.main import create_app
        
        app = create_app()
        with TestClient(app) as client:
            # Even though server flag is true, master switch prevents creation
            assert not hasattr(app.state, "mcp_server") or app.state.mcp_server is None
    
    def test_mcp_not_exposed_when_server_flag_disabled(self, monkeypatch):
        """MCP server should not be created when mcp_server_enabled=false."""
        monkeypatch.setenv("AGENTS_MCP_ENABLED", "true")  # Master enabled
        monkeypatch.setenv("AGENTS_MCP_SERVER_ENABLED", "false")  # Server disabled
        
        from src.main import create_app
        
        app = create_app()
        with TestClient(app) as client:
            # Master switch on but server flag off prevents creation
            assert not hasattr(app.state, "mcp_server") or app.state.mcp_server is None
