"""Unit tests for MCP Server scaffold.

WBS Reference: WBS-KB8 - VS Code MCP Server
Tasks: KB8.1 - Create MCP server scaffold

Acceptance Criteria:
- AC-KB8.1: MCP server starts on configured port with stdio transport
- AC-KB8.2: Server exposes tools via tools/list handler
- AC-KB8.3: Server executes tools via tools/call handler
- AC-KB8.4: Server handles feature flags for safe rollout

Exit Criteria:
- pytest tests/unit/mcp/test_server.py passes
- MCP server responds to tools/list with available tools
- MCP server responds to tools/call with tool results

TDD: RED Phase - These tests should FAIL initially

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → MCP Server Implementation
"""

from __future__ import annotations

import pytest
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from src.mcp.server import MCPServer


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

_CONST_SERVER_NAME = "ai-platform-agent-functions"
_CONST_DEFAULT_PORT = 8765
_CONST_TOOL_CROSS_REFERENCE = "cross_reference"
_CONST_TOOL_ANALYZE_CODE = "analyze_code"
_CONST_TOOL_GENERATE_CODE = "generate_code"
_CONST_TOOL_EXPLAIN_CODE = "explain_code"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Create mock configuration for MCP server."""
    return {
        "server_name": _CONST_SERVER_NAME,
        "port": _CONST_DEFAULT_PORT,
        "enabled": True,
        "feature_flags": {
            "mcp_enabled": True,
            "mcp_server_enabled": True,
        },
    }


@pytest.fixture
def mock_pipeline():
    """Create mock CrossReferencePipeline."""
    pipeline = AsyncMock()
    pipeline.run.return_value = MagicMock(
        content="Test response with citation [^1]",
        citations=[{"id": "1", "source": "test.py", "line": 10}],
        confidence=0.95,
        metadata={"cycles_used": 1},
    )
    return pipeline


@pytest.fixture
def mock_code_validation_tool():
    """Create mock CodeValidationTool."""
    tool = AsyncMock()
    tool.validate_code.return_value = MagicMock(
        success=True,
        findings=[],
        metrics={"quality_score": 0.9},
    )
    return tool


# =============================================================================
# MCPServerConfig Tests
# =============================================================================


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass."""

    def test_default_config_values(self):
        """AC-KB8.1: MCP server has sensible defaults."""
        from src.mcp.server import MCPServerConfig
        
        config = MCPServerConfig()
        
        assert config.server_name == _CONST_SERVER_NAME
        assert config.port == _CONST_DEFAULT_PORT
        assert config.enabled is True

    def test_config_from_dict(self):
        """MCPServerConfig can be created from dictionary."""
        from src.mcp.server import MCPServerConfig
        
        config = MCPServerConfig.from_dict({
            "server_name": "custom-server",
            "port": 9000,
            "enabled": False,
        })
        
        assert config.server_name == "custom-server"
        assert config.port == 9000
        assert config.enabled is False

    def test_config_feature_flag_check(self):
        """AC-KB8.4: Config respects feature flags."""
        from src.mcp.server import MCPServerConfig
        
        config = MCPServerConfig(
            feature_flags={"mcp_enabled": False}
        )
        
        assert config.is_enabled() is False


# =============================================================================
# MCPServer Initialization Tests
# =============================================================================


class TestMCPServerInit:
    """Tests for MCPServer initialization."""

    def test_server_creation_with_defaults(self):
        """MCPServer can be created with default configuration."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        
        assert server.config == config
        assert server._tools is not None

    def test_server_creation_disabled_by_feature_flag(self, mock_config):
        """AC-KB8.4: Server creation respects feature flags."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        mock_config["feature_flags"]["mcp_enabled"] = False
        config = MCPServerConfig.from_dict(mock_config)
        
        with pytest.raises(RuntimeError, match="MCP server disabled"):
            MCPServer(config)

    def test_server_registers_tools_on_init(self):
        """MCPServer registers all tools during initialization."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        
        tool_names = server.get_tool_names()
        
        assert _CONST_TOOL_CROSS_REFERENCE in tool_names
        assert _CONST_TOOL_ANALYZE_CODE in tool_names
        assert _CONST_TOOL_GENERATE_CODE in tool_names
        assert _CONST_TOOL_EXPLAIN_CODE in tool_names


# =============================================================================
# tools/list Handler Tests
# =============================================================================


class TestToolsListHandler:
    """Tests for MCP tools/list handler."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_all_tools(self):
        """AC-KB8.2: tools/list returns all registered tools."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        
        tools = await server.handle_list_tools()
        
        assert len(tools) >= 4
        tool_names = [t.name for t in tools]
        assert _CONST_TOOL_CROSS_REFERENCE in tool_names
        assert _CONST_TOOL_ANALYZE_CODE in tool_names

    @pytest.mark.asyncio
    async def test_list_tools_includes_schemas(self):
        """tools/list includes JSON schema for each tool."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        
        tools = await server.handle_list_tools()
        
        for tool in tools:
            assert tool.name is not None
            assert tool.description is not None
            assert tool.inputSchema is not None

    @pytest.mark.asyncio
    async def test_cross_reference_tool_schema(self):
        """cross_reference tool has proper schema."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        
        tools = await server.handle_list_tools()
        cr_tool = next(t for t in tools if t.name == _CONST_TOOL_CROSS_REFERENCE)
        
        assert "query" in cr_tool.inputSchema.get("properties", {})
        assert "sources" in cr_tool.inputSchema.get("properties", {})

    @pytest.mark.asyncio
    async def test_analyze_code_tool_schema(self):
        """analyze_code tool has proper schema."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        
        tools = await server.handle_list_tools()
        ac_tool = next(t for t in tools if t.name == _CONST_TOOL_ANALYZE_CODE)
        
        assert "code" in ac_tool.inputSchema.get("properties", {})
        assert "language" in ac_tool.inputSchema.get("properties", {})


# =============================================================================
# tools/call Handler Tests
# =============================================================================


class TestToolsCallHandler:
    """Tests for MCP tools/call handler."""

    @pytest.mark.asyncio
    async def test_call_unknown_tool_raises_error(self):
        """tools/call raises error for unknown tool."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        
        with pytest.raises(ValueError, match="Unknown tool"):
            await server.handle_call_tool("nonexistent_tool", {})

    @pytest.mark.asyncio
    async def test_call_cross_reference_tool(self, mock_pipeline):
        """AC-KB8.3: tools/call executes cross_reference tool."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        server._pipeline = mock_pipeline
        
        result = await server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "How does the discussion loop work?"}
        )
        
        assert result is not None
        assert len(result) > 0
        mock_pipeline.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_returns_text_content(self, mock_pipeline):
        """tools/call returns TextContent with citations."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        server._pipeline = mock_pipeline
        
        result = await server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Test query"}
        )
        
        # Result should be a list of TextContent
        assert isinstance(result, list)
        assert len(result) > 0
        # First item should be TextContent with text attribute
        assert hasattr(result[0], "text") or hasattr(result[0], "type")

    @pytest.mark.asyncio
    async def test_call_tool_validates_arguments(self):
        """tools/call validates required arguments."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        
        with pytest.raises(ValueError, match="Missing required"):
            await server.handle_call_tool(
                _CONST_TOOL_CROSS_REFERENCE,
                {}  # Missing required 'query' argument
            )


# =============================================================================
# Citation Formatting Tests
# =============================================================================


class TestCitationFormatting:
    """Tests for citation formatting in MCP responses."""

    @pytest.mark.asyncio
    async def test_response_includes_citations(self, mock_pipeline):
        """Responses include citation markers [^N]."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        server._pipeline = mock_pipeline
        
        result = await server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Test query"}
        )
        
        # Response should include citation marker
        text = result[0].text if hasattr(result[0], "text") else str(result[0])
        assert "[^" in text or "citation" in text.lower()

    @pytest.mark.asyncio
    async def test_response_includes_sources_section(self, mock_pipeline):
        """Responses include sources section at end."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        server._pipeline = mock_pipeline
        
        result = await server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Test query"}
        )
        
        text = result[0].text if hasattr(result[0], "text") else str(result[0])
        # Should have a Sources section with citation details
        assert "Sources:" in text or "[^1]:" in text or "References:" in text


# =============================================================================
# Session State Tests
# =============================================================================


class TestSessionState:
    """Tests for session state management in MCP server."""

    @pytest.mark.asyncio
    async def test_session_tracks_conversation(self, mock_pipeline):
        """Session tracks conversation for follow-ups."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        server._pipeline = mock_pipeline
        
        session_id = "test-session-123"
        
        # First query
        await server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "How does caching work?"},
            session_id=session_id
        )
        
        # Verify session state was stored
        assert session_id in server._sessions
        session = server._sessions[session_id]
        assert len(session.history) == 1

    @pytest.mark.asyncio
    async def test_follow_up_uses_context(self, mock_pipeline):
        """Follow-up questions use prior context."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        server._pipeline = mock_pipeline
        
        session_id = "test-session-456"
        
        # First query
        await server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Explain the discussion loop"},
            session_id=session_id
        )
        
        # Follow-up query
        await server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "What are its termination conditions?"},
            session_id=session_id
        )
        
        # Second call should include context from first
        assert mock_pipeline.run.call_count == 2
        # Context should be passed to pipeline
        second_call = mock_pipeline.run.call_args_list[1]
        assert "context" in str(second_call) or len(server._sessions[session_id].history) == 2


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for MCP server error handling."""

    @pytest.mark.asyncio
    async def test_pipeline_error_returns_error_content(self, mock_pipeline):
        """Pipeline errors are returned as error content."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        
        mock_pipeline.run.side_effect = RuntimeError("Pipeline failed")
        server._pipeline = mock_pipeline
        
        result = await server.handle_call_tool(
            _CONST_TOOL_CROSS_REFERENCE,
            {"query": "Test query"}
        )
        
        # Should return error content, not raise
        assert len(result) > 0
        text = result[0].text if hasattr(result[0], "text") else str(result[0])
        assert "error" in text.lower() or "failed" in text.lower()

    @pytest.mark.asyncio
    async def test_invalid_json_schema_returns_error(self):
        """Invalid input schema returns validation error."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        
        with pytest.raises(ValueError):
            await server.handle_call_tool(
                _CONST_TOOL_CROSS_REFERENCE,
                {"query": 12345}  # Should be string
            )


# =============================================================================
# Server Lifecycle Tests
# =============================================================================


class TestServerLifecycle:
    """Tests for MCP server lifecycle management."""

    @pytest.mark.asyncio
    async def test_server_start_configures_transport(self):
        """AC-KB8.1: Server starts with configured transport."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig(port=8765)
        server = MCPServer(config)
        
        # Server should have start method
        assert hasattr(server, "start")
        assert hasattr(server, "stop")

    @pytest.mark.asyncio
    async def test_server_cleanup_on_stop(self, mock_pipeline):
        """Server cleanup releases resources on stop."""
        from src.mcp.server import MCPServer, MCPServerConfig
        
        config = MCPServerConfig()
        server = MCPServer(config)
        server._pipeline = mock_pipeline
        
        # Should be able to stop without error
        await server.stop()
        
        # Sessions should be cleared
        assert len(server._sessions) == 0
