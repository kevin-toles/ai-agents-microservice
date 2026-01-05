# AI Platform MCP Server

Production-grade MCP (Model Context Protocol) server exposing Kitchen Brigade agent functions to VS Code, Claude Desktop, and other MCP-compatible clients.

## Quick Start

### Test the server
```bash
cd /Users/kevintoles/POC/ai-agents
source .venv/bin/activate
fastmcp inspect src/mcp/stdio_server.py
```

### Run interactively (requires Node.js)
```bash
fastmcp dev src/mcp/stdio_server.py
```

## Tools Exposed

| Tool | Description |
|------|-------------|
| `extract_structure` | Parse code/document structure (functions, classes, headings) |
| `summarize_content` | Generate summaries using LLM |
| `generate_code` | Generate code from specifications |
| `analyze_artifact` | Deep analysis of code artifacts |
| `validate_against_spec` | Validate code against specifications |
| `decompose_task` | Break tasks into subtasks (WBS generation) |
| `synthesize_outputs` | Combine multiple agent outputs |
| `cross_reference` | Search across code and book knowledge sources |
| `llm_complete` | **Tiered LLM fallback** (local → cloud → passthrough) |

## Tiered LLM Fallback

The `llm_complete` tool implements a resilient tiered architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    llm_complete()                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Tier 1: Local Inference (inference-service :8085)      │
│     └─ Uses Metal GPU acceleration                       │
│     └─ Supports: Qwen3, Phi-4, LLaMA, etc.              │
│                                                          │
│           ↓ if unavailable or fails                      │
│                                                          │
│  Tier 2: Cloud LLM (llm-gateway :8081)                  │
│     └─ Routes to OpenAI, Anthropic, etc.                │
│     └─ Handles API key management                        │
│                                                          │
│           ↓ if unavailable or fails                      │
│                                                          │
│  Tier 3: Passthrough (work_package)                      │
│     └─ Returns structured request for client             │
│     └─ Client (Copilot/Claude) handles completion        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## VS Code Integration

### Option 1: User Settings (all workspaces)
Add to `~/Library/Application Support/Code/User/settings.json`:
```json
{
  "github.copilot.chat.mcp.servers": {
    "ai-platform": {
      "command": "/Users/kevintoles/POC/ai-agents/.venv/bin/python",
      "args": ["-m", "src.mcp.stdio_server"],
      "cwd": "/Users/kevintoles/POC/ai-agents",
      "env": {
        "PYTHONPATH": "/Users/kevintoles/POC/ai-agents",
        "LLM_GATEWAY_URL": "http://localhost:8081"
      }
    }
  }
}
```

### Option 2: Workspace Settings (this project only)
Already configured in `.vscode/settings.json`

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        MCP Client                                 │
│              (VS Code Copilot / Claude Desktop)                   │
└──────────────────────────────┬───────────────────────────────────┘
                               │ stdio (JSON-RPC)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    stdio_server.py                                │
│                  (FastMCP Server)                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 9 @mcp.tool() decorated functions                          │  │
│  │   - extract_structure, summarize_content, generate_code    │  │
│  │   - analyze_artifact, validate_against_spec, decompose_task│  │
│  │   - synthesize_outputs, cross_reference, llm_complete      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────┘
                               │ HTTP
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Kitchen Brigade Services                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │
│  │ai-agents    │  │llm-gateway  │  │inference-service        │   │
│  │:8082        │  │:8081        │  │:8085 (local GPU)        │   │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │
│  │Qdrant       │  │Neo4j        │  │semantic-search-service  │   │
│  │:6333        │  │:7687        │  │:8083                    │   │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Interview Talking Points

### 1. Production MCP Implementation
- FastMCP with stdio transport for native VS Code/Claude Desktop integration
- 9 specialized tools following Google ADK patterns
- Type-safe with full JSON Schema validation

### 2. Tiered LLM Fallback
- Cost optimization: local first, cloud fallback
- Resilience: graceful degradation when services unavailable
- Flexibility: client can handle final tier if all else fails

### 3. Kitchen Brigade Architecture
- Microservices with clear separation of concerns
- Unified protocol layer (WBS-PI7 integration)
- Cross-referencing across code and documentation knowledge bases

### 4. Developer Experience
- Single config file for VS Code integration
- `fastmcp inspect` for introspection
- Full async/await support

## Testing

```python
# Test server internals
cd /Users/kevintoles/POC/ai-agents
source .venv/bin/activate

python -c "
from src.mcp.stdio_server import mcp
import asyncio

async def test():
    tool = mcp._tool_manager._tools['llm_complete']
    result = await tool.fn(prompt='What is 2+2?', model_preference='cloud')
    print(result)

asyncio.run(test())
"
```

## Dependencies

- `fastmcp>=2.14.0` - MCP server framework
- `httpx` - Async HTTP client
- `pydantic` - Data validation

## Status

✅ **Production Ready**
- All 9 tools registered and functional
- Tiered fallback tested with cloud tier
- VS Code configs in place
- Ready for interview demo
