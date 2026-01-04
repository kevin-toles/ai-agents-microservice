"""Tests for MCP Agent Functions Server.

TDD tests for WBS-PI5: MCP Server - Agent Functions Exposure.

Acceptance Criteria Coverage:
- AC-PI5.1: MCP server initializes from FUNCTION_REGISTRY
- AC-PI5.2: list_tools() returns all 8 agent functions as MCP tools
- AC-PI5.3: Tool inputSchema generated from function input model
- AC-PI5.4: call_tool() executes agent function and returns result
- AC-PI5.5: Result formatted as MCP TextContent
- AC-PI5.7: Server handles tool errors gracefully

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → MCP Server Implementation

Note: This is separate from the KB8 MCP server (VS Code integration).
WBS-PI5 exposes agent functions for external MCP clients (Claude Desktop, etc.)
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel


# Mock model for testing
class MockOutput(BaseModel):
    result: str = "test"


# Mock model for testing
class MockOutput(BaseModel):
    result: str = "test"


# =============================================================================
# AC-PI5.1: MCP Server Initialization from FUNCTION_REGISTRY
# =============================================================================


class TestAgentFunctionsMCPServer:
    """Tests for agent functions MCP server creation."""

    def test_create_agent_functions_mcp_server_exists(self) -> None:
        """create_agent_functions_mcp_server() function can be imported."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        assert callable(create_agent_functions_mcp_server)

    @pytest.mark.asyncio
    async def test_create_server_returns_server_dict(self) -> None:
        """create_agent_functions_mcp_server() returns server configuration."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        
        # Server should have name and tool handlers
        assert isinstance(server, dict)
        assert "name" in server
        assert server["name"] == "ai-platform-agent-functions"

    @pytest.mark.asyncio
    async def test_server_has_list_tools_handler(self) -> None:
        """Server has list_tools handler function."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        
        assert "list_tools" in server
        assert callable(server["list_tools"])

    @pytest.mark.asyncio
    async def test_server_has_call_tool_handler(self) -> None:
        """Server has call_tool handler function."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        
        assert "call_tool" in server
        assert callable(server["call_tool"])


# =============================================================================
# AC-PI5.2: list_tools() Returns All 8 Agent Functions
# =============================================================================


class TestListToolsHandler:
    """Tests for list_tools() returning agent function tools."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_8_tools(self) -> None:
        """list_tools() returns exactly 8 tools from FUNCTION_REGISTRY."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        list_tools_fn = server["list_tools"]
        
        tools = await list_tools_fn()
        
        # Should return exactly 8 tools (matching FUNCTION_REGISTRY)
        assert len(tools) == 8

    @pytest.mark.asyncio
    async def test_list_tools_includes_extract_structure(self) -> None:
        """list_tools() includes extract_structure tool."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        list_tools_fn = server["list_tools"]
        
        tools = await list_tools_fn()
        tool_names = [tool["name"] for tool in tools]
        
        assert "extract_structure" in tool_names

    @pytest.mark.asyncio
    async def test_list_tools_includes_all_functions(self) -> None:
        """list_tools() includes all 8 agent functions."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        list_tools_fn = server["list_tools"]
        
        tools = await list_tools_fn()
        
        expected_tools = {
            "extract_structure",
            "summarize_content",
            "generate_code",
            "analyze_artifact",
            "validate_against_spec",
            "decompose_task",
            "synthesize_outputs",
            "cross_reference",
        }
        
        actual_tools = {tool["name"] for tool in tools}
        assert actual_tools == expected_tools

    @pytest.mark.asyncio
    async def test_tool_has_description(self) -> None:
        """Each tool has a description field."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        list_tools_fn = server["list_tools"]
        
        tools = await list_tools_fn()
        
        for tool in tools:
            assert "description" in tool
            assert isinstance(tool["description"], str)
            assert len(tool["description"]) > 0


# =============================================================================
# AC-PI5.3: Tool Input Schema from Pydantic Models
# =============================================================================


class TestToolInputSchema:
    """Tests for JSON schema generation from Pydantic input models."""

    @pytest.mark.asyncio
    async def test_extract_structure_has_input_schema(self) -> None:
        """extract_structure tool has valid inputSchema."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        list_tools_fn = server["list_tools"]
        
        tools = await list_tools_fn()
        extract_tool = next(t for t in tools if t["name"] == "extract_structure")
        
        assert "inputSchema" in extract_tool
        assert extract_tool["inputSchema"] is not None

    @pytest.mark.asyncio
    async def test_input_schema_is_valid_json_schema(self) -> None:
        """inputSchema is valid JSON Schema dict."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        list_tools_fn = server["list_tools"]
        
        tools = await list_tools_fn()
        extract_tool = next(t for t in tools if t["name"] == "extract_structure")
        
        schema = extract_tool["inputSchema"]
        
        # Should be a dict with standard JSON Schema properties
        assert isinstance(schema, dict)
        assert "type" in schema or "properties" in schema

    @pytest.mark.asyncio
    async def test_schema_includes_required_fields(self) -> None:
        """Input schema includes required field declarations."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        list_tools_fn = server["list_tools"]
        
        tools = await list_tools_fn()
        extract_tool = next(t for t in tools if t["name"] == "extract_structure")
        
        schema = extract_tool["inputSchema"]
        
        # extract_structure requires 'content' and 'extraction_type'
        if "properties" in schema:
            assert "content" in schema["properties"]
            # May have 'required' array listing required fields
            if "required" in schema:
                assert isinstance(schema["required"], list)


# =============================================================================
# AC-PI5.4: call_tool() Executes Agent Functions
# =============================================================================


class TestCallToolExecution:
    """Tests for call_tool() executing agent functions."""

    @pytest.mark.asyncio
    async def test_call_tool_executes_function(self) -> None:
        """call_tool() executes the specified agent function."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        call_tool_fn = server["call_tool"]
        
        # Call extract_structure with valid arguments
        result = await call_tool_fn(
            name="extract_structure",
            arguments={
                "content": "# Test Heading\n\nSome content.",
                "extraction_type": "headings",
            },
        )
        
        # Should return result (not raise exception)
        assert result is not None

    @pytest.mark.asyncio
    async def test_call_tool_with_mock_function(self) -> None:
        """call_tool() invokes function class and calls run()."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        # Mock the function execution
        mock_output = MockOutput(result="test")
        
        with patch("src.mcp.agent_functions_server.FUNCTION_REGISTRY") as mock_registry, \
             patch("src.mcp.agent_functions_server.INPUT_SCHEMA_REGISTRY", {}):
            mock_func_class = MagicMock()
            mock_func_instance = AsyncMock()
            mock_func_instance.run.return_value = mock_output
            mock_func_class.return_value = mock_func_instance
            
            mock_registry.__getitem__.return_value = mock_func_class
            mock_registry.keys.return_value = ["extract-structure"]
            mock_registry.__contains__.return_value = True
            
            server = await create_agent_functions_mcp_server()
            call_tool_fn = server["call_tool"]
            
            await call_tool_fn(
                name="extract_structure",
                arguments={"content": "test"},
            )
            
            # Verify function was instantiated and run() was called
            mock_func_class.assert_called_once()
            mock_func_instance.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_passes_arguments_correctly(self) -> None:
        """call_tool() passes arguments dict to function.run()."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        mock_output = MockOutput(result="test")
        
        with patch("src.mcp.agent_functions_server.FUNCTION_REGISTRY") as mock_registry, \
             patch("src.mcp.agent_functions_server.INPUT_SCHEMA_REGISTRY", {}):
            mock_func_class = MagicMock()
            mock_func_instance = AsyncMock()
            mock_func_instance.run.return_value = mock_output
            mock_func_class.return_value = mock_func_instance
            
            mock_registry.__getitem__.return_value = mock_func_class
            mock_registry.keys.return_value = ["extract-structure"]
            mock_registry.__contains__.return_value = True
            
            server = await create_agent_functions_mcp_server()
            call_tool_fn = server["call_tool"]
            
            test_args = {"content": "test", "extraction_type": "keywords"}
            await call_tool_fn(
                name="extract_structure",
                arguments=test_args,
            )
            
            # Verify arguments were passed as kwargs
            call_args = mock_func_instance.run.call_args
            assert call_args.kwargs == test_args


# =============================================================================
# AC-PI5.5: Result Formatted as MCP TextContent
# =============================================================================


class TestResultFormatting:
    """Tests for MCP TextContent result formatting."""

    @pytest.mark.asyncio
    async def test_call_tool_returns_text_content_list(self) -> None:
        """call_tool() returns list with TextContent structure."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        call_tool_fn = server["call_tool"]
        
        result = await call_tool_fn(
            name="extract_structure",
            arguments={
                "content": "# Test",
                "extraction_type": "headings",
            },
        )
        
        # Result should be a list
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_text_content_has_correct_structure(self) -> None:
        """TextContent dict has type='text' and text field."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        call_tool_fn = server["call_tool"]
        
        result = await call_tool_fn(
            name="extract_structure",
            arguments={
                "content": "# Test",
                "extraction_type": "headings",
            },
        )
        
        # First item should be TextContent dict
        text_content = result[0]
        assert isinstance(text_content, dict)
        assert "type" in text_content
        assert text_content["type"] == "text"
        assert "text" in text_content

    @pytest.mark.asyncio
    async def test_result_text_is_json_serialized(self) -> None:
        """Result text field contains JSON-serialized function output."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        call_tool_fn = server["call_tool"]
        
        result = await call_tool_fn(
            name="extract_structure",
            arguments={
                "content": "# Test Heading",
                "extraction_type": "headings",
            },
        )
        
        text_content = result[0]
        
        # Should be valid JSON
        output_data = json.loads(text_content["text"])
        assert isinstance(output_data, dict)


# =============================================================================
# AC-PI5.7: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for graceful error handling."""

    @pytest.mark.asyncio
    async def test_call_tool_with_invalid_tool_name(self) -> None:
        """call_tool() handles invalid tool name gracefully."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        call_tool_fn = server["call_tool"]
        
        # Should return error content, not raise exception
        result = await call_tool_fn(
            name="nonexistent_function",
            arguments={},
        )
        
        assert isinstance(result, list)
        assert len(result) > 0
        # Error should be in text content
        assert "error" in result[0]["text"].lower() or "unknown" in result[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_call_tool_with_invalid_arguments(self) -> None:
        """call_tool() handles validation errors gracefully."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        server = await create_agent_functions_mcp_server()
        call_tool_fn = server["call_tool"]
        
        # Missing required arguments - should return error, not raise
        result = await call_tool_fn(
            name="extract_structure",
            arguments={},  # Missing required 'content' and 'extraction_type'
        )
        
        assert isinstance(result, list)
        assert len(result) > 0
        # Should contain error information
        text = result[0]["text"]
        assert "error" in text.lower() or "required" in text.lower()

    @pytest.mark.asyncio
    async def test_call_tool_handles_function_execution_errors(self) -> None:
        """call_tool() handles function execution errors gracefully."""
        from src.mcp.agent_functions_server import create_agent_functions_mcp_server
        
        with patch("src.mcp.agent_functions_server.FUNCTION_REGISTRY") as mock_registry, \
             patch("src.mcp.agent_functions_server.INPUT_SCHEMA_REGISTRY", {}):
            mock_func_class = MagicMock()
            mock_func_instance = AsyncMock()
            # Simulate function execution error
            mock_func_instance.run.side_effect = RuntimeError("Execution failed")
            mock_func_class.return_value = mock_func_instance
            
            mock_registry.__getitem__.return_value = mock_func_class
            mock_registry.keys.return_value = ["extract-structure"]
            mock_registry.__contains__.return_value = True
            
            server = await create_agent_functions_mcp_server()
            call_tool_fn = server["call_tool"]
            
            # Should return error content, not raise
            result = await call_tool_fn(
                name="extract_structure",
                arguments={"content": "test", "extraction_type": "keywords"},
            )
            
            assert isinstance(result, list)
            assert "error" in result[0]["text"].lower()
