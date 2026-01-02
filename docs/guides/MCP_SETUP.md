# MCP Server Setup Guide

## WBS Reference
**WBS-KB8** - VS Code MCP Server Integration

## Overview

The AI Platform MCP (Model Context Protocol) Server exposes agent functions as standardized tools that VS Code and other MCP-compatible clients can consume. This enables seamless integration of cross-reference queries, code analysis, and code generation directly within the development environment.

## Available Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `cross_reference` | Cross-reference questions across code, docs, and textbooks | Research, understanding architecture |
| `analyze_code` | Analyze code for quality, security, patterns | Code review, refactoring |
| `generate_code` | Generate code with pattern references | Scaffolding, implementation |
| `explain_code` | Explain code with design pattern analysis | Learning, documentation |

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/kevintoles/POC/ai-agents
pip install -r requirements.txt
```

### 2. Configure VS Code

Copy the MCP configuration template to your VS Code settings:

```bash
cp config/mcp.json.template ~/.vscode/mcp.json
```

Or add to your workspace's `.vscode/mcp.json`:

```json
{
  "mcpServers": {
    "ai-platform-agent-functions": {
      "command": "python3",
      "args": ["-m", "src.mcp.server"],
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "AGENTS_MCP_ENABLED": "true",
        "AGENTS_MCP_SERVER_ENABLED": "true"
      }
    }
  }
}
```

### 3. Verify Installation

Run the MCP server tests:

```bash
python3 -m pytest tests/unit/mcp/ tests/integration/test_mcp_server.py -v
```

Expected output: All tests pass (22 unit + 16 integration = 38 tests)

## Usage Examples

### Cross-Reference Query

Ask questions that span code, documentation, and textbooks:

```
Tool: cross_reference
Arguments: {"query": "How does the discussion loop handle participant disagreement?"}
```

Response includes:
- Answer grounded in sources
- Citations `[^1]`, `[^2]` pointing to specific files/lines
- Sources section with full references
- Confidence score

### Code Analysis

Analyze code for quality and patterns:

```
Tool: analyze_code
Arguments: {
  "code": "async def process(data):\n    return await transform(data)",
  "language": "python",
  "checks": ["quality", "security", "patterns"]
}
```

Response includes:
- Quality score
- Security findings
- Pattern analysis
- References to best practices

### Code Generation

Generate code based on requirements:

```
Tool: generate_code
Arguments: {
  "requirement": "Create a retry decorator with exponential backoff",
  "language": "python",
  "style": "async"
}
```

Response includes:
- Generated code
- References to similar patterns in codebase
- Usage examples

### Code Explanation

Explain existing code with references:

```
Tool: explain_code
Arguments: {
  "code": "<paste code here>",
  "detail_level": "detailed",
  "include_patterns": true
}
```

Response includes:
- Functionality explanation
- Design patterns detected
- Related documentation references

## Session Management

The MCP server maintains session state for follow-up questions. Pass a `session_id` to track conversation context:

```python
# First query
await server.handle_call_tool(
    "cross_reference",
    {"query": "Explain the pipeline architecture"},
    session_id="my-session"
)

# Follow-up uses context from first query
await server.handle_call_tool(
    "cross_reference",
    {"query": "What are its termination conditions?"},
    session_id="my-session"
)
```

## Feature Flags

Control MCP server behavior via environment variables:

| Flag | Default | Description |
|------|---------|-------------|
| `AGENTS_MCP_ENABLED` | `true` | Master switch for MCP functionality |
| `AGENTS_MCP_SERVER_ENABLED` | `true` | Enable MCP server specifically |

To disable:

```bash
export AGENTS_MCP_ENABLED=false
```

## Architecture

```
VS Code (MCP Host)
    │
    ├── MCP Client ─── tools/list ──► MCPServer.handle_list_tools()
    │                                      │
    │                                      ▼
    │                               ToolDefinition[]
    │
    └── MCP Client ─── tools/call ──► MCPServer.handle_call_tool()
                                           │
                       ┌───────────────────┼───────────────────┐
                       ▼                   ▼                   ▼
              CrossReferencePipeline  CodeValidationTool   Session
                       │                   │                   │
                       ▼                   ▼                   ▼
                  TextContent[]       TextContent[]      History[]
```

## Testing

### Unit Tests

```bash
python3 -m pytest tests/unit/mcp/test_server.py -v
```

Tests cover:
- MCPServerConfig initialization
- Tool registration
- tools/list handler
- tools/call handler
- Session management
- Error handling

### Integration Tests

```bash
python3 -m pytest tests/integration/test_mcp_server.py -v
```

Tests cover:
- Full cross_reference flow
- Full analyze_code flow
- Session continuity
- Citation formatting
- Error recovery
- Server lifecycle

## Troubleshooting

### Server Won't Start

1. Check feature flags are enabled
2. Verify Python path is correct
3. Check dependencies installed

### Tools Not Appearing

1. Verify mcp.json configuration
2. Check server logs for errors
3. Restart VS Code

### Citations Missing

1. Ensure pipeline is connected
2. Check source retrieval is working
3. Verify citation formatter output

## Related Documentation

- [KITCHEN_BRIGADE_ARCHITECTURE.md](../docs/KITCHEN_BRIGADE_ARCHITECTURE.md) - System architecture
- [PROTOCOL_INTEGRATION_ARCHITECTURE.md](../docs/PROTOCOL_INTEGRATION_ARCHITECTURE.md) - Protocol details
- [Cross-Reference Pipeline](../src/pipelines/cross_reference_pipeline.py) - Pipeline implementation

## Exit Criteria Status

| Criteria | Status |
|----------|--------|
| MCP server starts on configured port | ✅ |
| VS Code detects tools via mcp.json | ✅ |
| Tools return grounded answers with citations [^N] | ✅ |
| Follow-up questions work within session | ✅ |
| Unit tests pass (22) | ✅ |
| Integration tests pass (16) | ✅ |

**Total: 38 tests passing**
