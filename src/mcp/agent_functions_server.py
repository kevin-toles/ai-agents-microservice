"""MCP Server for Agent Functions.

WBS-PI5: MCP Server Implementation
Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → MCP Server Implementation

Exposes all agent functions as MCP tools, enabling external MCP clients
(Claude Desktop, VS Code, other ADK agents) to discover and invoke our
agent functions via the standardized MCP protocol.

Architecture Pattern:
- Dict-based approach for testing compatibility
- For production: Use FastMCP's stdio transport with decorators
- Reference: https://google.github.io/adk-docs/tools-custom/mcp-tools/

Acceptance Criteria:
- AC-PI5.1: MCP server initializes from FUNCTION_REGISTRY
- AC-PI5.2: list_tools() returns all 8 agent functions as MCP tools
- AC-PI5.3: Tool inputSchema generated from function input model
- AC-PI5.4: call_tool() executes agent function and returns result
- AC-PI5.5: Result formatted as MCP TextContent
- AC-PI5.7: Server handles tool errors gracefully
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.api.routes.functions import FUNCTION_REGISTRY
from src.schemas.functions.extract_structure import ExtractStructureInput
from src.schemas.functions.summarize_content import SummarizeContentInput
from src.schemas.functions.generate_code import GenerateCodeInput
from src.schemas.functions.analyze_artifact import AnalyzeArtifactInput
from src.schemas.functions.validate_against_spec import ValidateAgainstSpecInput
from src.schemas.functions.decompose_task import DecomposeTaskInput
from src.schemas.functions.cross_reference import CrossReferenceInput


logger = logging.getLogger(__name__)


# Map function names to their input schemas
INPUT_SCHEMA_REGISTRY: dict[str, type] = {
    "extract-structure": ExtractStructureInput,
    "summarize-content": SummarizeContentInput,
    "generate-code": GenerateCodeInput,
    "analyze-artifact": AnalyzeArtifactInput,
    "validate-against-spec": ValidateAgainstSpecInput,
    "decompose-task": DecomposeTaskInput,
    "synthesize-outputs": SummarizeContentInput,  # Reuses SummarizeContentInput
    "cross-reference": CrossReferenceInput,
}


async def create_agent_functions_mcp_server() -> dict:
    """Create MCP server exposing agent functions as tools.
    
    Returns dict-based server for testing compatibility.
    For production, use FastMCP's stdio transport.
    
    Returns:
        Dict with server name and handler functions:
        - name: Server identifier
        - list_tools: Async function returning list of MCP tools
        - call_tool: Async function executing tools
    
    Example:
        >>> server = await create_agent_functions_mcp_server()
        >>> tools = await server["list_tools"]()
        >>> len(tools)
        8
        >>> result = await server["call_tool"]("extract_structure", {...})
    
    Reference: https://google.github.io/adk-docs/tools-custom/mcp-tools/
    """
    
    async def list_tools() -> list[dict[str, Any]]:
        """List all agent functions as MCP tools.
        
        Returns:
            List of MCP tool definitions with name, description, and inputSchema.
            
        AC-PI5.2: Returns all 8 agent functions as MCP tools
        AC-PI5.3: Tool inputSchema generated from function input model
        """
        tools = []
        
        for func_name, func_class in FUNCTION_REGISTRY.items():
            try:
                # Instantiate function to get schema
                func_instance = func_class()
                
                # Convert hyphenated name to underscored for MCP compatibility
                tool_name = func_name.replace("-", "_")
                
                # Get description from class docstring
                description = func_class.__doc__ or f"{func_name} agent function"
                description = description.strip().split("\n")[0]  # First line only
                
                # Generate input schema from Pydantic Input model
                input_schema = {}
                if func_name in INPUT_SCHEMA_REGISTRY:
                    input_model = INPUT_SCHEMA_REGISTRY[func_name]
                    # Use Pydantic's model_json_schema to generate valid JSON Schema
                    input_schema = input_model.model_json_schema()
                
                tools.append({
                    "name": tool_name,
                    "description": description,
                    "inputSchema": input_schema
                })
                
                logger.debug(f"Registered MCP tool: {tool_name}")
                
            except Exception as e:
                logger.error(f"Failed to register tool {func_name}: {e}")
                continue
        
        logger.info(f"MCP server initialized with {len(tools)} tools")
        return tools
    
    async def call_tool(name: str, arguments: dict) -> list[dict[str, str]]:
        """Execute agent function and return result.
        
        Args:
            name: Tool name (underscored format, e.g., "extract_structure")
            arguments: Tool arguments matching inputSchema
            
        Returns:
            List of MCP TextContent objects with execution results.
            
        AC-PI5.4: call_tool() executes agent function and returns result
        AC-PI5.5: Result formatted as MCP TextContent
        AC-PI5.7: Server handles tool errors gracefully
        """
        # Convert underscored name back to hyphenated for registry lookup
        func_name = name.replace("_", "-")
        
        # Validate tool exists
        if func_name not in FUNCTION_REGISTRY:
            error_response = {
                "error": f"Unknown tool: {name}",
                "available_tools": list(FUNCTION_REGISTRY.keys())
            }
            return [{"type": "text", "text": json.dumps(error_response, indent=2)}]
        
        try:
            # Validate arguments using Input model if available
            if func_name in INPUT_SCHEMA_REGISTRY:
                try:
                    input_model = INPUT_SCHEMA_REGISTRY[func_name]
                    validated_input = input_model(**arguments)
                    # Convert to dict for function execution
                    arguments = validated_input.model_dump()
                except Exception as validation_error:
                    error_response = {
                        "error": f"Invalid arguments for {name}: {str(validation_error)}",
                        "tool": name,
                        "arguments_provided": arguments
                    }
                    logger.error(f"Argument validation failed for {name}: {validation_error}")
                    return [{"type": "text", "text": json.dumps(error_response, indent=2)}]
            
            # Get function class and instantiate
            func_class = FUNCTION_REGISTRY[func_name]
            func_instance = func_class()
            
            logger.info(f"Executing MCP tool: {name} with arguments: {arguments}")
            
            # Execute function - handle both sync and async
            result = await func_instance.run(**arguments)
            
            # Format result as MCP TextContent
            result_data = result.model_dump() if hasattr(result, 'model_dump') else result
            
            return [{"type": "text", "text": json.dumps(result_data, indent=2)}]
            
        except TypeError as e:
            # Invalid arguments (shouldn't reach here with validation above)
            error_response = {
                "error": f"Invalid arguments for {name}: {str(e)}",
                "tool": name,
                "arguments_provided": arguments
            }
            logger.error(f"Invalid arguments for {name}: {e}")
            return [{"type": "text", "text": json.dumps(error_response, indent=2)}]
            
        except Exception as e:
            # Execution error
            error_response = {
                "error": f"Execution failed for {name}: {str(e)}",
                "tool": name,
                "error_type": type(e).__name__
            }
            logger.error(f"Tool execution failed for {name}: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps(error_response, indent=2)}]
    
    # Return dict-based server for testing
    return {
        "name": "ai-platform-agent-functions",
        "list_tools": list_tools,
        "call_tool": call_tool
    }


# =============================================================================
# Production FastMCP Implementation (Future Enhancement)
# =============================================================================
# 
# For production deployment with stdio transport, use FastMCP decorators:
#
# from fastmcp import FastMCP
# 
# mcp = FastMCP("ai-platform-agent-functions")
# 
# @mcp.tool()
# async def extract_structure(content: str, extraction_type: str):
#     """Extract structured data from content."""
#     func = ExtractStructureFunction()
#     return await func.run(content=content, extraction_type=extraction_type)
# 
# @mcp.tool()
# async def summarize_content(content: str, target_length: str):
#     """Generate summary of content."""
#     func = SummarizeContentFunction()
#     return await func.run(content=content, target_length=target_length)
#
# ... (repeat for all 8 functions)
#
# if __name__ == "__main__":
#     mcp.run()
