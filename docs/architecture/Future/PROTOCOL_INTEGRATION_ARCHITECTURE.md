# Protocol Integration Architecture (Phase 2)

> **Version:** 1.2.0  
> **Created:** 2025-12-29  
> **Updated:** 2026-01-03  
> **Status:** Implementation Complete (8 of 8 blocks complete)  
> **Phase:** Phase 2 of Agent Architecture Evolution  
> **Reference:** [AGENT_FUNCTIONS_ARCHITECTURE.md](AGENT_FUNCTIONS_ARCHITECTURE.md), [ADK_MIGRATION_GUIDE.md](ADK_MIGRATION_GUIDE.md)

## Overview

This document defines the **Protocol Integration Architecture** for enabling Agent-to-Agent (A2A) communication and Model Context Protocol (MCP) tool standardization within the AI Platform. This is **Phase 2** of the architecture evolution.

### Architecture Evolution Phases

| Phase | Focus | Status | Document |
|-------|-------|--------|----------|
| **Phase 1** | Agent Functions + ADK Patterns | ✅ Complete | [AGENT_FUNCTIONS_ARCHITECTURE.md](AGENT_FUNCTIONS_ARCHITECTURE.md) |
| **Phase 2** | Protocol Integration (A2A + MCP) | 🚧 In Progress (75%) | [WBS_PROTOCOL_INTEGRATION.md](../../../Platform%20Documentation/ai-agents/WBS_PROTOCOL_INTEGRATION.md) |
| **Phase 3** | Full ADK Migration | 📋 Planned | [ADK_MIGRATION_GUIDE.md](ADK_MIGRATION_GUIDE.md) |

### Design Philosophy

> "Protocols are the language agents speak; tools are the actions they take."

| Protocol | Purpose | Integration Layer |
|----------|---------|-------------------|
| **A2A** | Agent-to-agent communication & discovery | Inter-service communication |
| **MCP** | Tool standardization & external integrations | Tool layer abstraction |

---

## Table of Contents

1. [No-Conflict Analysis](#no-conflict-analysis)
2. [Feature Flags](#feature-flags)
3. [A2A Protocol Integration](#a2a-protocol-integration)
4. [MCP Tool Integration](#mcp-tool-integration)
5. [Platform Service Mapping](#platform-service-mapping)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Testing Strategy](#testing-strategy)
8. [Rollback Plan](#rollback-plan)

---

## No-Conflict Analysis

A2A and MCP integration is **fully compatible** with existing Phase 1 architecture:

### Compatibility Matrix

| Component | Phase 1 (Current) | Phase 2 (A2A/MCP) | Relationship |
|-----------|-------------------|-------------------|--------------|
| **Endpoints** | REST `/v1/functions/{name}/run` | A2A JSON-RPC `/v1/message:send` | Parallel—both active |
| **State Prefixes** | `temp:`, `user:`, `app:` | A2A `contextId`, `taskId` | Complementary—A2A tracks tasks |
| **Tools** | `cross_reference`, `semantic_search` | MCP Toolsets | MCP wraps existing tools |
| **Artifacts** | Platform Artifact conventions | A2A Artifact parts | Same semantics |
| **Discovery** | Hardcoded service registry | A2A Agent Cards | A2A adds dynamic discovery |
| **Orchestration** | Pipeline DAGs | A2A Task lifecycle | A2A adds external visibility |

### Why No Conflicts?

1. **A2A is a protocol layer**, not a replacement for agent functions
2. **MCP standardizes tool interfaces**, wrapping our existing tools
3. **Both are opt-in** via feature flags—zero impact on current paths
4. **Agent functions remain the execution unit**—protocols provide communication

---

## Feature Flags

All Phase 2 features are behind feature flags for safe experimentation.

### Environment Variables

```bash
# A2A Protocol Features
export AGENTS_A2A_ENABLED=false           # Master switch for A2A
export AGENTS_A2A_AGENT_CARD_ENABLED=false # Expose /.well-known/agent-card.json
export AGENTS_A2A_TASK_LIFECYCLE_ENABLED=false # A2A task endpoints (message:send, tasks)
export AGENTS_A2A_STREAMING_ENABLED=false  # SSE streaming for task updates
export AGENTS_A2A_PUSH_NOTIFICATIONS=false # Webhook notifications

# MCP Features
export AGENTS_MCP_ENABLED=false           # Master switch for MCP
export AGENTS_MCP_SERVER_ENABLED=false    # Expose agent functions as MCP tools
export AGENTS_MCP_CLIENT_ENABLED=false    # Consume external MCP servers
export AGENTS_MCP_SEMANTIC_SEARCH=false # MCP wrapper for semantic-search-service (hybrid RAG)
export AGENTS_MCP_TOOLBOX_NEO4J=false     # MCP Toolbox for Neo4j
export AGENTS_MCP_TOOLBOX_REDIS=false     # MCP Toolbox for Redis
```

### Feature Flag Configuration

```python
# config/feature_flags.py
from pydantic_settings import BaseSettings
from typing import Literal

class ProtocolFeatureFlags(BaseSettings):
    """Phase 2 Protocol Integration feature flags.
    
    All flags default to False for safe rollout.
    Enable incrementally in development → staging → production.
    """
    
    # A2A Protocol
    a2a_enabled: bool = False
    a2a_agent_card_enabled: bool = False
    a2a_task_lifecycle_enabled: bool = False
    a2a_streaming_enabled: bool = False
    a2a_push_notifications: bool = False
    
    # MCP Protocol
    mcp_enabled: bool = False
    mcp_server_enabled: bool = False
    mcp_client_enabled: bool = False
    mcp_semantic_search: bool = False  # semantic-search-service hybrid RAG
    mcp_toolbox_neo4j: bool = False
    mcp_toolbox_redis: bool = False
    
    class Config:
        env_prefix = "AGENTS_"
        
    def a2a_available(self) -> bool:
        """Check if any A2A feature is enabled."""
        return self.a2a_enabled and any([
            self.a2a_agent_card_enabled,
            self.a2a_task_lifecycle_enabled,
            self.a2a_streaming_enabled,
            self.a2a_push_notifications,
        ])
    
    def mcp_available(self) -> bool:
        """Check if any MCP feature is enabled."""
        return self.mcp_enabled and any([
            self.mcp_server_enabled,
            self.mcp_client_enabled,
            self.mcp_semantic_search,
            self.mcp_toolbox_neo4j,
            self.mcp_toolbox_redis,
        ])
```

---

## A2A Protocol Integration

### Overview

The Agent-to-Agent (A2A) Protocol enables standardized communication between agents across platforms. Key concepts:

| A2A Concept | Description | Platform Mapping |
|-------------|-------------|------------------|
| **Agent Card** | JSON manifest declaring capabilities | Service discovery at `/.well-known/agent-card.json` |
| **Task** | Unit of work with lifecycle | Pipeline execution instance |
| **Message** | Communication unit with Parts | Request/response payloads |
| **Artifact** | Task output (document, code, data) | Agent function output |
| **Skill** | Declared capability | Agent function |

### Agent Card Schema

Each Kitchen Brigade service exposes an Agent Card when A2A is enabled:

```json
{
  "protocolVersion": "0.3.0",
  "name": "ai-agents-service",
  "description": "AI Platform Agent Functions Service - Pipeline orchestration and agent function execution",
  "supportedInterfaces": [
    {
      "url": "http://localhost:8082/a2a/v1",
      "protocolBinding": "HTTP+JSON"
    }
  ],
  "provider": {
    "organization": "AI Platform",
    "url": "http://localhost:8082"
  },
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "stateTransitionHistory": true
  },
  "defaultInputModes": ["application/json", "text/plain"],
  "defaultOutputModes": ["application/json", "text/plain"],
  "skills": [
    {
      "id": "extract_structure",
      "name": "Extract Structure",
      "description": "Extract structured data (keywords, concepts, entities, outline) from unstructured content",
      "tags": ["extraction", "nlp", "structure"],
      "examples": [
        "Extract keywords from this chapter",
        "Identify the main concepts in this document"
      ],
      "inputModes": ["text/plain", "application/json"],
      "outputModes": ["application/json"]
    },
    {
      "id": "summarize_content",
      "name": "Summarize Content",
      "description": "Compress content while preserving key invariants",
      "tags": ["summarization", "compression", "nlp"],
      "examples": [
        "Summarize this chapter in 500 words",
        "Create an executive summary preserving these key points"
      ]
    },
    {
      "id": "generate_code",
      "name": "Generate Code",
      "description": "Generate code from specification with context awareness",
      "tags": ["code", "generation", "development"],
      "examples": [
        "Generate a Python class implementing the Repository pattern",
        "Create a FastAPI endpoint for user authentication"
      ]
    },
    {
      "id": "analyze_artifact",
      "name": "Analyze Artifact",
      "description": "Analyze code or documents for quality, security, and patterns",
      "tags": ["analysis", "quality", "security"],
      "examples": [
        "Analyze this code for security vulnerabilities",
        "Check code quality and complexity metrics"
      ]
    },
    {
      "id": "validate_against_spec",
      "name": "Validate Against Spec",
      "description": "Validate artifact meets specification and acceptance criteria",
      "tags": ["validation", "verification", "qa"],
      "examples": [
        "Verify this code implements the specification",
        "Check if the summary preserves all required invariants"
      ]
    },
    {
      "id": "synthesize_outputs",
      "name": "Synthesize Outputs",
      "description": "Combine multiple artifacts into coherent result",
      "tags": ["synthesis", "merge", "reconciliation"],
      "examples": [
        "Merge these code snippets into a single module",
        "Reconcile conflicting analysis results"
      ]
    },
    {
      "id": "decompose_task",
      "name": "Decompose Task",
      "description": "Break complex task into executable subtasks",
      "tags": ["planning", "decomposition", "workflow"],
      "examples": [
        "Break down this feature request into implementation steps",
        "Create a task plan for refactoring this module"
      ]
    },
    {
      "id": "cross_reference",
      "name": "Cross Reference",
      "description": "Find related content across knowledge bases via semantic search",
      "tags": ["search", "retrieval", "reference"],
      "examples": [
        "Find similar implementations in our codebase",
        "Search for related patterns in the reference library"
      ]
    }
  ]
}
```

### A2A Task Lifecycle Mapping

A2A tasks map to platform pipeline executions:

```
A2A Task States          Platform Pipeline States
─────────────────        ─────────────────────────
submitted       ◀─────── Pipeline queued
working         ◀─────── Stage executing
input-required  ◀─────── Validation failed (retry needed)
completed       ◀─────── All stages successful
failed          ◀─────── Unrecoverable error
canceled        ◀─────── User cancellation
```

### A2A Endpoint Routes

When `AGENTS_A2A_ENABLED=true`:

```python
# src/routes/a2a.py
from fastapi import APIRouter, Depends
from src.config.feature_flags import ProtocolFeatureFlags

router = APIRouter(prefix="/a2a/v1", tags=["A2A Protocol"])

@router.post("/message:send")
async def send_message(
    request: A2ASendMessageRequest,
    flags: ProtocolFeatureFlags = Depends(get_feature_flags)
):
    """A2A SendMessage - initiates task execution."""
    if not flags.a2a_enabled:
        raise HTTPException(501, "A2A protocol not enabled")
    # Map to agent function execution
    ...

@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """A2A GetTask - retrieve task status."""
    ...

@router.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str):
    """A2A CancelTask - request task cancellation."""
    ...
```

### Agent Card Discovery

```python
# src/routes/well_known.py
@router.get("/.well-known/agent-card.json")
async def get_agent_card(
    flags: ProtocolFeatureFlags = Depends(get_feature_flags)
):
    """Expose A2A Agent Card for service discovery."""
    if not flags.a2a_agent_card_enabled:
        raise HTTPException(404, "Agent Card not available")
    return generate_agent_card()
```

---

## MCP Tool Integration

### Overview

Model Context Protocol (MCP) standardizes how LLMs interact with external tools. Two integration patterns:

| Pattern | Direction | Use Case |
|---------|-----------|----------|
| **MCP Client** | Platform → External MCP Servers | Consume third-party tools (Google Maps, GitHub, etc.) |
| **MCP Server** | External → Platform | Expose agent functions to MCP clients (Claude, other ADK agents) |

### MCP Integration Architecture

> **Critical Design Decision**: semantic-search-service is a **hybrid RAG layer**, not just a Qdrant proxy.
> It combines vector search + graph traversal + domain taxonomy scoring to reduce LLM hallucination.
> Direct Qdrant access would bypass these hallucination-reduction mechanisms.

#### Integration Patterns

| Integration | Platform Service | MCP Pattern | Hallucination Reduction |
|-------------|------------------|-------------|-------------------------|
| **semantic-search-service** | ai-agents → semantic-search:8081 | MCP wrapper (Phase 2) | ✅ Hybrid RAG: α=0.7 vector + 0.3 graph fusion |
| **Neo4j** | Graph relationships | genai-toolbox (Phase 2) | ✅ Cypher queries, schema inspection |
| **Redis** | State caching | genai-toolbox (Phase 2) | N/A - operational caching |

#### Why semantic-search-service, Not Direct Qdrant?

1. **Hybrid Score Fusion**: `final_score = 0.7 * vector_score + 0.3 * graph_score`
2. **Domain Taxonomy**: Boosts results from whitelisted books/tiers, penalizes irrelevant sources
3. **Multi-Collection**: Searches code-reference-engine + ai-platform-data/books simultaneously
4. **Graceful Degradation**: Falls back if Neo4j unavailable

> **Note**: genai-toolbox does NOT support Qdrant. Direct Qdrant access (if needed for Phase 3 ADK)
> should use `google.adk.tools.third_party.qdrant` for administrative operations only.

### MCP Server Implementation

Expose agent functions as MCP tools using FastMCP:

```python
# src/mcp/agent_functions_server.py
from fastmcp import FastMCP
from src.api.routes.functions import FUNCTION_REGISTRY
import json

async def create_agent_functions_mcp_server() -> dict:
    """Create MCP server exposing agent functions as tools.
    
    Uses dict-based approach for testing. For production, use FastMCP's
    stdio transport with decorator pattern.
    
    Reference:
    - https://google.github.io/adk-docs/tools-custom/mcp-tools/
    - https://github.com/jlowin/fastmcp
    """
    
    async def list_tools():
        """Advertise all agent functions as MCP tools."""
        tools = []
        for func_name, func_class in FUNCTION_REGISTRY.items():
            func_instance = func_class()
            tools.append({
                "name": func_name.replace("-", "_"),
                "description": func_class.__doc__ or f"{func_name} agent function",
                "inputSchema": {"type": "object", "properties": {}}
            })
        return tools
    
    async def call_tool(name: str, arguments: dict):
        """Execute agent function and return MCP-formatted result."""
        func_name = name.replace("_", "-")
        if func_name not in FUNCTION_REGISTRY:
            return [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {name}"})}]
        
        try:
            func_class = FUNCTION_REGISTRY[func_name]
            func_instance = func_class()
            result = await func_instance.run(**arguments)
            return [{"type": "text", "text": json.dumps(result.model_dump())}]
        except Exception as e:
            return [{"type": "text", "text": json.dumps({"error": str(e)})}]
    
    return {
        "name": "ai-platform-agent-functions",
        "list_tools": list_tools,
        "call_tool": call_tool
    }
```

**Production FastMCP Pattern** (for stdio transport):
```python
from fastmcp import FastMCP

mcp = FastMCP("ai-platform-agent-functions")

@mcp.tool()
async def extract_structure(content: str, extraction_type: str) -> dict:
    """Extract structured data from content."""
    func = FUNCTION_REGISTRY["extract-structure"]()
    result = await func.run(content=content, extraction_type=extraction_type)
    return result.model_dump()

# Register all 8 agent functions similarly
```

### MCP Client Integration

#### Semantic Search MCP Wrapper (Phase 2)

Wrap semantic-search-service REST API as MCP tools for standardized agent access:

```python
# src/mcp/semantic_search_wrapper.py
from httpx import AsyncClient
from src.config.feature_flags import ProtocolFeatureFlags

class SemanticSearchMcpWrapper:
    """MCP wrapper for semantic-search-service hybrid RAG.
    
    This wraps our existing semantic-search-service REST API as MCP tools,
    preserving the hybrid search (vector + graph) and domain taxonomy scoring
    that reduces LLM hallucination.
    
    Architecture:
    - semantic-search-service: Python service on port 8081
    - Provides: hybrid_search, graph_relationships, domain_taxonomy
    - NOT raw Qdrant: semantic-search-service adds RAG intelligence
    
    Why MCP wrapper instead of genai-toolbox?
    - genai-toolbox does NOT support Qdrant
    - semantic-search-service provides hybrid RAG, not just vector search
    - Preserves score fusion and domain taxonomy
    """
    
    def __init__(
        self,
        flags: ProtocolFeatureFlags,
        semantic_search_url: str = "http://localhost:8081",
    ):
        self.flags = flags
        self.semantic_search_url = semantic_search_url
        self._client: AsyncClient | None = None
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        search_scope: list[str] | None = None,
    ) -> dict:
        """Hybrid search across knowledge bases.
        
        Combines vector similarity with graph relationships using
        configurable score fusion (default: α=0.7 vector, 0.3 graph).
        """
        if not self.flags.mcp_semantic_search:
            return {"error": "MCP semantic search not enabled"}
        
        if self._client is None:
            self._client = AsyncClient(base_url=self.semantic_search_url)
        
        response = await self._client.post(
            "/v1/search/hybrid",
            json={"query": query, "top_k": top_k, "scope": search_scope or []},
        )
        return response.json()
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
```

#### genai-toolbox Integration (Neo4j, Redis)

For databases supported by genai-toolbox:

```python
# src/mcp/toolbox_client.py
from toolbox_core import ToolboxClient
from src.config.feature_flags import ProtocolFeatureFlags

class McpToolboxManager:
    """Manage connections to external genai-toolbox MCP server.
    
    genai-toolbox is a Go-based MCP server that exposes database
    operations as standardized tools.
    
    Supported databases: Neo4j, Redis, PostgreSQL, MySQL, MongoDB, BigQuery
    NOT supported: Qdrant (use SemanticSearchMcpWrapper instead)
    
    Reference: https://github.com/googleapis/genai-toolbox
    """
    
    def __init__(
        self,
        flags: ProtocolFeatureFlags,
        toolbox_url: str = "http://127.0.0.1:5000",
    ):
        self.flags = flags
        self.toolbox_url = toolbox_url
        self._toolsets: dict[str, list] = {}
    
    async def get_neo4j_toolset(self) -> list | None:
        """Get Neo4j tools from genai-toolbox MCP server."""
        if not self.flags.mcp_toolbox_neo4j:
            return None
        
        if "neo4j" not in self._toolsets:
            async with ToolboxClient(self.toolbox_url) as client:
                tools = await client.load_toolset("neo4j_toolset")
                self._toolsets["neo4j"] = tools
        return self._toolsets["neo4j"]
    
    async def get_redis_toolset(self) -> list | None:
        """Get Redis tools from genai-toolbox MCP server."""
        if not self.flags.mcp_toolbox_redis:
            return None
        
        if "redis" not in self._toolsets:
            async with ToolboxClient(self.toolbox_url) as client:
                tools = await client.load_toolset("redis_toolset")
                self._toolsets["redis"] = tools
        return self._toolsets["redis"]
    
    async def close_all(self) -> None:
        """Clean up cached toolsets."""
        self._toolsets.clear()
```

---

## Platform Service Mapping

### A2A Agent Cards per Service

Each Kitchen Brigade service can expose an Agent Card:

| Service | Port | A2A Endpoint | Skills |
|---------|------|--------------|--------|
| **ai-agents** | 8082 | `/.well-known/agent-card.json` | 8 agent functions + pipelines |
| **semantic-search** | 8081 | `/.well-known/agent-card.json` | `search`, `index`, `similarity` |
| **Code-Orchestrator** | 8083 | `/.well-known/agent-card.json` | `embed`, `classify`, `extract_keywords` |
| **inference-service** | 8085 | `/.well-known/agent-card.json` | `generate`, `critique`, `debate`, `ensemble` |
| **audit-service** | 8084 | `/.well-known/agent-card.json` | `format_citations`, `track_provenance` |
| **llm-gateway** | 8080 | `/.well-known/agent-card.json` | Aggregates all downstream skills |

### Service Communication Flow (with A2A)

```
External A2A Client                    External MCP Client
       │                                      │
       ▼                                      ▼
┌─────────────────────────────────────────────────────────┐
│                    llm-gateway :8080                    │
│  ┌─────────────────┐    ┌──────────────────┐           │
│  │ REST Endpoints  │    │ A2A/MCP Gateway  │           │
│  │ (Phase 1)       │    │ (Phase 2 - flag) │           │
│  └────────┬────────┘    └────────┬─────────┘           │
│           │                      │                      │
│           └──────────┬───────────┘                      │
│                      ▼                                  │
└──────────────────────┼──────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  ai-agents  │ │semantic-srch│ │inference-svc│
│   :8082     │ │   :8081     │ │   :8085     │
│             │ │             │ │             │
│ Agent Card  │ │ Agent Card  │ │ Agent Card  │
│ A2A Tasks   │ │ MCP Qdrant  │ │ A2A Skills  │
│ MCP Server  │ │             │ │             │
└─────────────┘ └─────────────┘ └─────────────┘
```

---

## Implementation Roadmap

### Phase 2 Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| **Week 1** | Feature Flags + Agent Card | `ProtocolFeatureFlags`, basic Agent Card endpoint |
| **Week 2** | MCP Server | Expose agent functions as MCP tools |
| **Week 3** | MCP Client (Qdrant) | MCP Toolbox integration for vector search |
| **Week 4** | A2A Task Lifecycle | Task state mapping, basic endpoints |
| **Week 5** | A2A Streaming | SSE support for task updates |
| **Week 6** | Integration Testing | End-to-end protocol tests |

### Detailed Task Breakdown

#### Week 1: Feature Flags + Agent Card

```
□ 1.1 Create feature flag configuration
    - ProtocolFeatureFlags pydantic model
    - Environment variable mapping
    - Feature availability helpers
    
□ 1.2 Implement Agent Card generation
    - AgentCard Pydantic model
    - Skill extraction from agent function registry
    - Version and capability declaration
    
□ 1.3 Add well-known endpoint
    - GET /.well-known/agent-card.json
    - Feature flag guard
    - Health check integration
```

#### Week 2: MCP Server

```
□ 2.1 Install MCP dependencies
    - pip install mcp a2a-sdk
    - Add to requirements.txt
    
□ 2.2 Implement MCP server
    - Tool listing from agent function registry
    - Tool execution → agent function mapping
    - Response formatting
    
□ 2.3 MCP server lifecycle
    - Startup/shutdown hooks
    - Connection management
    - Error handling
```

#### Week 3: MCP Client (Qdrant)

```
□ 3.1 McpToolboxManager implementation
    - Toolset initialization
    - Connection pooling
    - Graceful shutdown
    
□ 3.2 Qdrant MCP Toolset integration
    - Replace direct Qdrant client calls (optional)
    - Tool filtering for security
    - Response mapping
    
□ 3.3 cross_reference enhancement
    - Option to use MCP Toolbox
    - Fallback to direct client
```

#### Week 4: A2A Task Lifecycle

```
□ 4.1 A2A Pydantic models
    - Task, Message, Artifact, Part
    - TaskStatus, TaskState enum
    - SendMessageRequest/Response
    
□ 4.2 A2A endpoints
    - POST /a2a/v1/message:send
    - GET /a2a/v1/tasks/{id}
    - POST /a2a/v1/tasks/{id}:cancel
    
□ 4.3 Task state storage
    - In-memory for development
    - Redis backing for persistence
```

#### Week 5: A2A Streaming

```
□ 5.1 SSE implementation
    - POST /a2a/v1/message:stream
    - TaskStatusUpdateEvent streaming
    - TaskArtifactUpdateEvent streaming
    
□ 5.2 Pipeline → A2A event mapping
    - Stage start → working
    - Stage complete → artifact update
    - Pipeline complete → completed
```

#### Week 6: Integration Testing

```
□ 6.1 A2A protocol tests
    - Agent Card validation
    - Task lifecycle tests
    - Streaming tests
    
□ 6.2 MCP integration tests
    - MCP server tool listing
    - MCP tool execution
    - MCP Toolbox tests
    
□ 6.3 Feature flag tests
    - Enabled/disabled behavior
    - Partial enablement
    - Rollback verification
```

---

## Testing Strategy

### Feature Flag Testing

```python
# tests/test_feature_flags.py
import pytest
from src.config.feature_flags import ProtocolFeatureFlags

class TestProtocolFeatureFlags:
    
    def test_all_disabled_by_default(self):
        """All protocol features disabled by default."""
        flags = ProtocolFeatureFlags()
        assert not flags.a2a_enabled
        assert not flags.mcp_enabled
        assert not flags.a2a_available()
        assert not flags.mcp_available()
    
    def test_a2a_requires_master_flag(self):
        """A2A sub-features require master flag."""
        flags = ProtocolFeatureFlags(
            a2a_enabled=False,
            a2a_agent_card_enabled=True
        )
        assert not flags.a2a_available()
    
    def test_partial_enablement(self):
        """Can enable specific features."""
        flags = ProtocolFeatureFlags(
            mcp_enabled=True,
            mcp_semantic_search=True,
            mcp_toolbox_neo4j=False
        )
        assert flags.mcp_available()
        assert flags.mcp_semantic_search
        assert not flags.mcp_toolbox_neo4j
```

### A2A Protocol Tests

```python
# tests/test_a2a_protocol.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_agent_card_disabled_by_default(client: AsyncClient):
    """Agent Card returns 404 when disabled."""
    response = await client.get("/.well-known/agent-card.json")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_agent_card_enabled(client_with_a2a: AsyncClient):
    """Agent Card returns valid JSON when enabled."""
    response = await client_with_a2a.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    card = response.json()
    assert card["protocolVersion"] == "0.3.0"
    assert "skills" in card
    assert len(card["skills"]) == 8  # 8 agent functions

@pytest.mark.asyncio
async def test_a2a_task_lifecycle(client_with_a2a: AsyncClient):
    """Test A2A task lifecycle: send → working → completed."""
    # Send message
    response = await client_with_a2a.post("/a2a/v1/message:send", json={
        "message": {
            "role": "user",
            "parts": [{"text": "Extract keywords from this text"}],
            "messageId": "test-msg-1"
        }
    })
    assert response.status_code == 200
    result = response.json()
    assert "task" in result
    task_id = result["task"]["id"]
    
    # Check task status
    response = await client_with_a2a.get(f"/a2a/v1/tasks/{task_id}")
    assert response.status_code == 200
    task = response.json()
    assert task["status"]["state"] in ["submitted", "working", "completed"]
```

### MCP Integration Tests

```python
# tests/test_mcp_integration.py
import pytest

@pytest.mark.asyncio
async def test_mcp_server_lists_tools(mcp_server):
    """MCP server lists all agent functions as tools."""
    tools = await mcp_server.list_tools()
    tool_names = [t.name for t in tools]
    
    assert "extract_structure" in tool_names
    assert "generate_code" in tool_names
    assert "cross_reference" in tool_names
    assert len(tool_names) == 8

@pytest.mark.asyncio
async def test_mcp_tool_execution(mcp_server):
    """MCP tool call executes agent function."""
    result = await mcp_server.call_tool(
        "extract_structure",
        {"content": "Hello world", "extraction_type": "keywords"}
    )
    assert len(result) > 0
    assert result[0].type == "text"
```

---

## Rollback Plan

### Immediate Rollback

All Phase 2 features can be disabled instantly via environment variables:

```bash
# Disable all A2A features
export AGENTS_A2A_ENABLED=false

# Disable all MCP features
export AGENTS_MCP_ENABLED=false

# Restart service
docker-compose restart ai-agents
```

### Feature-Level Rollback

Individual features can be disabled:

```bash
# Keep MCP but disable A2A
export AGENTS_A2A_ENABLED=false
export AGENTS_MCP_ENABLED=true
export AGENTS_MCP_SEMANTIC_SEARCH=true

# Disable specific MCP toolbox
export AGENTS_MCP_TOOLBOX_NEO4J=false
```

### Rollback Verification

```python
# scripts/verify_rollback.py
async def verify_rollback():
    """Verify all Phase 2 features are disabled."""
    
    # Check Agent Card is 404
    response = await client.get("/.well-known/agent-card.json")
    assert response.status_code == 404, "Agent Card should be disabled"
    
    # Check A2A endpoints are 501
    response = await client.post("/a2a/v1/message:send", json={})
    assert response.status_code == 501, "A2A should be disabled"
    
    # Check Phase 1 endpoints still work
    response = await client.post("/v1/functions/extract-structure/run", json={
        "content": "test",
        "extraction_type": "keywords"
    })
    assert response.status_code == 200, "Phase 1 endpoints must remain functional"
    
    print("✅ Rollback verification passed")
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.2.0 | 2026-01-03 | Renamed `mcp_toolbox_qdrant` → `mcp_semantic_search`; Corrected MCP Toolbox table (genai-toolbox does NOT support Qdrant); Added SemanticSearchMcpWrapper for hybrid RAG; Clarified semantic-search-service as hallucination-reduction layer |
| 1.1.0 | 2026-01-03 | MCP Client code corrected to use `toolbox-core` SDK; Status updated to reflect 6/8 WBS blocks complete |
| 1.0.0 | 2025-12-29 | Initial Protocol Integration Architecture (Phase 2) |

---

## References

### Internal Documentation
- [AGENT_FUNCTIONS_ARCHITECTURE.md](AGENT_FUNCTIONS_ARCHITECTURE.md) - Phase 1 architecture
- [ADK_MIGRATION_GUIDE.md](ADK_MIGRATION_GUIDE.md) - Phase 3 full ADK adoption
- [inference-service/MODEL_LIBRARY.md](../../inference-service/docs/MODEL_LIBRARY.md) - Model configurations

### External Documentation
- [A2A Protocol Specification](https://a2a-protocol.org/latest/specification/) - Agent-to-Agent protocol
- [A2A GitHub](https://github.com/a2aproject/A2A) - A2A reference implementations
- [MCP Documentation](https://modelcontextprotocol.io/) - Model Context Protocol
- [ADK MCP Integration](https://google.github.io/adk-docs/tools-custom/mcp-tools/) - ADK + MCP patterns
- [MCP Toolbox for Databases](https://google.github.io/adk-docs/tools/google-cloud/mcp-toolbox-for-databases/) - Database toolsets
