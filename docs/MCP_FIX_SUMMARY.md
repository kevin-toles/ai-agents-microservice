MCP Fix Summary (Copilot + VS Code)

What was done
- Added a Python FastMCP fallback server to Copilot MCP settings so tools are always available even if the NodeJS dynamic server is down.
- Verified the ai-agents API endpoints used for dynamic tools.

Global Copilot MCP settings (applies to every VS Code window)
File: /Users/kevintoles/Library/Application Support/Code/User/settings.json

Added server:
name: ai-kitchen-brigade-fallback
command: python3
args: -m src.mcp.stdio_server
cwd: /Users/kevintoles/POC/ai-agents
env: PYTHONPATH=/Users/kevintoles/POC/ai-agents

Existing dynamic server (already present)
name: ai-kitchen-brigade
command: node
args: /Users/kevintoles/POC/ai-agents-mcp-server/dist/index.js
env includes AI_AGENTS_URL and related service URLs

Current API status
- http://localhost:8082/v1/functions returns 8 functions.
- http://localhost:8082/v1/protocols returns 0 protocols.

Why “dynamic tools” may not update
- Functions list is static and comes from FUNCTION_REGISTRY in:
  /Users/kevintoles/POC/ai-agents/src/api/routes/functions.py
  Add new functions there and restart ai-agents.
- Protocols list is dynamic and comes from JSON files in:
  /Users/kevintoles/POC/ai-agents/config/protocols
  Add a new JSON file and it should show up without restart.

If tools do not appear in Copilot
1) Restart VS Code to reload MCP servers.
2) Ensure ai-agents service is running and /v1/functions is reachable.
3) Check Copilot > MCP tools list.
