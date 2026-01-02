# ADK Migration Guide (Option A)

> **Version:** 1.0.0  
> **Created:** 2025-12-29  
> **Status:** Future Roadmap  
> **Reference:** [AGENT_FUNCTIONS_ARCHITECTURE.md](AGENT_FUNCTIONS_ARCHITECTURE.md), [Google ADK Documentation](https://google.github.io/adk-docs/)

## Overview

This document provides a comprehensive migration guide for **Option A: Full ADK Adoption**. This is a future roadmap—current implementation uses Option C (Cherry-Pick Patterns) as documented in [AGENT_FUNCTIONS_ARCHITECTURE.md](AGENT_FUNCTIONS_ARCHITECTURE.md).

### Migration Options Summary

| Option | Description | Status | Document |
|--------|-------------|--------|----------|
| **Option A** | Full ADK framework adoption | Future | This document |
| **Option B** | A2A Protocol for multi-platform | Deferred | Not planned |
| **Option C** | Cherry-pick patterns only | **Current** | [AGENT_FUNCTIONS_ARCHITECTURE.md](AGENT_FUNCTIONS_ARCHITECTURE.md) |

### Why Option A Later?

Full ADK adoption provides:
- **Declarative workflow composition** - SequentialAgent, ParallelAgent, LoopAgent
- **Built-in state management** - `temp:`, `user:`, `app:` prefixes
- **A2A protocol support** - Cross-platform agent interoperability
- **Callback system** - Lifecycle hooks for observability
- **Multi-language SDKs** - Python, TypeScript, Go, Java

Current blockers for immediate adoption:
- Requires refactoring existing pipeline orchestration
- Dependency on `google-adk` package (new dependency)
- Testing infrastructure needs adaptation
- Team learning curve

---

## Table of Contents

1. [ADK Core Concepts](#adk-core-concepts)
2. [Mapping to Platform Architecture](#mapping-to-platform-architecture)
3. [Migration Phases](#migration-phases)
4. [Workflow Agent Implementation](#workflow-agent-implementation)
5. [State Management Migration](#state-management-migration)
6. [Artifact Service Integration](#artifact-service-integration)
7. [A2A Protocol Considerations](#a2a-protocol-considerations)
8. [Testing Strategy](#testing-strategy)
9. [Rollback Plan](#rollback-plan)

---

## ADK Core Concepts

### Agent Types

| ADK Agent Type | Purpose | Our Equivalent |
|----------------|---------|----------------|
| `LlmAgent` | Single LLM with tools | Agent function with preset |
| `SequentialAgent` | Execute agents in order | Pipeline stages |
| `ParallelAgent` | Execute agents concurrently | Parallel cross_reference |
| `LoopAgent` | Repeat until condition | Validation retry loop |
| `CustomAgent` | Custom logic | Pipeline orchestrator |

### State Prefixes

```python
# ADK state prefix system
state["temp:key"]    # Current invocation only - cleared after
state["user:key"]    # Persists across sessions for user
state["app:key"]     # Application-wide shared state
```

### Artifacts

```python
# ADK artifacts API
artifacts.save_artifact(
    filename="output.json",
    artifact=json.dumps(data),
    mime_type="application/json"
)

# Versioning
artifacts.save_artifact(
    filename="document_v2.md",  # Explicit version in filename
    artifact=content
)
```

### Workflow Composition

```python
# ADK SequentialAgent
from google.adk.agents import SequentialAgent

pipeline = SequentialAgent(
    name="chapter_summarization",
    sub_agents=[
        extract_structure_agent,
        cross_reference_agent,
        summarize_content_agent,
        validate_agent,
    ]
)
```

---

## Mapping to Platform Architecture

### Kitchen Brigade → ADK Agents

| Kitchen Brigade Role | Service | ADK Mapping |
|---------------------|---------|-------------|
| **Router** | llm-gateway:8080 | API Gateway (unchanged) |
| **Expeditor** | ai-agents:8082 | `SequentialAgent` orchestrator |
| **Cookbook** | semantic-search:8081 | Tool for `cross_reference` agent |
| **Sous Chef** | Code-Orchestrator:8083 | Tool for `analyze_artifact` agent |
| **Line Cook** | inference-service:8085 | LLM backend for all agents |
| **Auditor** | audit-service:8084 | Callback hooks |
| **Pantry** | ai-platform-data | Artifact storage |

### Agent Functions → ADK Agents

```python
# Current: Agent function
class ExtractStructureFunction:
    def run(self, input: ExtractStructureInput) -> ExtractStructureOutput:
        # Implementation
        pass

# ADK Migration: LlmAgent with tools
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

extract_structure_agent = LlmAgent(
    name="extract_structure",
    model="phi-4",
    description="Extract keywords, concepts, entities, or outline from raw content",
    instruction=EXTRACT_STRUCTURE_PROMPT,
    output_key="temp:extracted_structure",  # State prefix
)
```

### Pipeline → SequentialAgent

```python
# Current: Pipeline definition (YAML)
pipeline:
  stages:
    - extract_structure
    - cross_reference
    - summarize_content
    - validate_against_spec

# ADK Migration: SequentialAgent
chapter_summarization_pipeline = SequentialAgent(
    name="chapter_summarization",
    description="Summarize chapter with citations and validation",
    sub_agents=[
        extract_structure_agent,
        cross_reference_agent,
        summarize_content_agent,
        validate_against_spec_agent,
    ],
)
```

---

## Migration Phases

### Phase 1: Foundation (2-3 weeks)

**Objective:** Add ADK as optional dependency, create adapter layer.

| Task | Effort | Dependencies |
|------|--------|--------------|
| Add `google-adk` to `requirements.txt` | 1 day | None |
| Create `adk_adapter.py` bridge module | 3 days | ADK installed |
| Implement state prefix helpers | 2 days | Adapter complete |
| Create ADK agent wrappers for existing functions | 5 days | State helpers |
| Unit tests for adapter layer | 3 days | Wrappers complete |

**Deliverable:** ADK can be optionally enabled; existing functions work unchanged.

### Phase 2: Workflow Agents (2-3 weeks)

**Objective:** Replace custom pipeline orchestration with ADK workflow agents.

| Task | Effort | Dependencies |
|------|--------|--------------|
| Implement `SequentialAgent` for chapter summarization | 3 days | Phase 1 |
| Implement `ParallelAgent` for cross_reference | 2 days | Phase 1 |
| Implement `LoopAgent` for validation retry | 2 days | Phase 1 |
| Integration tests for workflow agents | 3 days | Workflow agents |
| Performance benchmarking | 2 days | Integration tests |
| Documentation updates | 2 days | All above |

**Deliverable:** Pipelines run via ADK workflow agents.

### Phase 3: State Migration (1-2 weeks)

**Objective:** Migrate caches to ADK state management.

| Task | Effort | Dependencies |
|------|--------|--------------|
| Map `handoff_cache` → `temp:` prefix | 2 days | Phase 2 |
| Map `compression_cache` → `user:` prefix | 2 days | Phase 2 |
| Map `artifact_store` → `app:` prefix | 2 days | Phase 2 |
| Update all state access patterns | 3 days | Mapping complete |
| Cache migration tests | 2 days | Access patterns |

**Deliverable:** All state uses ADK prefix conventions.

### Phase 4: Full Integration (2 weeks)

**Objective:** Complete ADK adoption, deprecate legacy code.

| Task | Effort | Dependencies |
|------|--------|--------------|
| Remove legacy pipeline orchestrator | 2 days | Phase 3 |
| Implement ADK callbacks for audit | 3 days | Phase 3 |
| A2A protocol evaluation | 3 days | Phase 3 |
| End-to-end testing | 3 days | All above |
| Production deployment | 2 days | E2E tests |
| Legacy code cleanup | 2 days | Deployment |

**Deliverable:** Full ADK adoption complete.

### Total Estimated Effort

| Phase | Duration | Risk Level |
|-------|----------|------------|
| Phase 1: Foundation | 2-3 weeks | Low |
| Phase 2: Workflow Agents | 2-3 weeks | Medium |
| Phase 3: State Migration | 1-2 weeks | Low |
| Phase 4: Full Integration | 2 weeks | Medium |
| **Total** | **7-10 weeks** | **Medium** |

---

## Workflow Agent Implementation

### SequentialAgent Example

```python
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.tools import FunctionTool

# Define sub-agents
extract_agent = LlmAgent(
    name="extract_structure",
    model="phi-4",
    instruction="Extract keywords, concepts, and outline from the chapter.",
    output_key="temp:structure",
)

summarize_agent = LlmAgent(
    name="summarize_content",
    model="qwen2.5-14b",
    instruction="Summarize the chapter using the extracted structure.",
    input_keys=["temp:structure"],  # Read from previous stage
    output_key="temp:summary",
)

validate_agent = LlmAgent(
    name="validate_against_spec",
    model="deepseek-r1-7b",
    instruction="Validate summary covers all concepts from structure.",
    input_keys=["temp:structure", "temp:summary"],
    output_key="temp:validation",
)

# Compose into pipeline
chapter_pipeline = SequentialAgent(
    name="chapter_summarization",
    description="Extract, summarize, and validate chapter content",
    sub_agents=[extract_agent, summarize_agent, validate_agent],
)
```

### ParallelAgent Example

```python
from google.adk.agents import ParallelAgent

# Parallel cross-reference across multiple repositories
parallel_search = ParallelAgent(
    name="parallel_cross_reference",
    description="Search multiple knowledge bases concurrently",
    sub_agents=[
        cross_reference_textbooks_agent,
        cross_reference_code_agent,
        cross_reference_schemas_agent,
    ],
)
```

### LoopAgent Example

```python
from google.adk.agents import LoopAgent

# Retry validation with feedback
validation_loop = LoopAgent(
    name="validation_with_retry",
    description="Validate and regenerate until passing",
    sub_agent=summarize_and_validate_agent,
    max_iterations=3,
    continue_condition=lambda state: not state.get("temp:validation_passed"),
)
```

---

## State Management Migration

### Current → ADK Mapping

| Current Implementation | ADK Equivalent | Migration Notes |
|------------------------|----------------|-----------------|
| `handoff_cache[key]` | `state["temp:{key}"]` | Auto-cleared after pipeline |
| `redis.get(f"compress:{key}")` | `state["user:{key}"]` | Persists across sessions |
| `qdrant.retrieve(key)` | `state["app:{key}"]` | Application-wide |

### Migration Pattern

```python
# Before (current implementation)
class PipelineOrchestrator:
    def __init__(self):
        self.handoff_cache = {}
    
    def execute_stage(self, stage, input_data):
        result = stage.run(input_data)
        self.handoff_cache[stage.name] = result
        return result

# After (ADK migration)
from google.adk.agents import SequentialAgent

class ADKPipelineOrchestrator:
    def __init__(self):
        self.pipeline = SequentialAgent(...)
    
    async def execute(self, input_data):
        # ADK handles state automatically via temp: prefix
        result = await self.pipeline.run(input_data)
        return result
```

---

## Artifact Service Integration

### Current Artifact Storage

```python
# Current: Direct cache writes
async def save_artifact(self, name: str, data: bytes):
    cache_key = f"artifact:{self.pipeline_id}:{name}"
    await self.redis.set(cache_key, data)
```

### ADK Artifact Service

```python
from google.adk.artifacts import Artifacts

async def save_artifact_adk(self, name: str, data: bytes, mime_type: str):
    artifacts = Artifacts(session=self.session)
    await artifacts.save_artifact(
        filename=f"{name}_v{self.version}",
        artifact=data,
        mime_type=mime_type,
    )
```

### Versioning Strategy

```python
# ADK versioning via filename
artifacts.save_artifact(filename="summary_v1.json", artifact=v1_data)
artifacts.save_artifact(filename="summary_v2.json", artifact=v2_data)

# Load latest
latest = artifacts.load_artifact(filename="summary_v2.json")

# Load all versions
versions = artifacts.list_artifacts(pattern="summary_v*.json")
```

---

## A2A Protocol Considerations

### When to Consider A2A

A2A (Agent-to-Agent) protocol is relevant when:
- Integrating with external agent platforms
- Multi-organization agent collaboration
- Standardized agent discovery and communication

### A2A Agent Card

```json
{
  "name": "ai-agents",
  "description": "Agent function orchestrator for AI Platform",
  "url": "http://ai-agents:8082",
  "version": "1.1.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "stateTransitionHistory": true
  },
  "skills": [
    {
      "id": "extract_structure",
      "name": "Extract Structure",
      "description": "Extract keywords, concepts, entities from content"
    },
    {
      "id": "summarize_content",
      "name": "Summarize Content",
      "description": "Compress content while preserving invariants"
    }
  ]
}
```

### Recommendation

**Defer A2A adoption** until:
1. External integration requirements emerge
2. ADK A2A protocol matures (currently early stage)
3. Multi-platform agent ecosystem develops

---

## Testing Strategy

### Unit Tests

```python
import pytest
from google.adk.testing import MockLlmAgent, MockState

@pytest.fixture
def mock_extract_agent():
    return MockLlmAgent(
        name="extract_structure",
        mock_output={"keywords": ["test"], "concepts": ["example"]},
    )

async def test_sequential_pipeline(mock_extract_agent, mock_summarize_agent):
    pipeline = SequentialAgent(
        name="test_pipeline",
        sub_agents=[mock_extract_agent, mock_summarize_agent],
    )
    
    result = await pipeline.run({"content": "Test content"})
    
    assert result["temp:summary"] is not None
```

### Integration Tests

```python
@pytest.mark.integration
async def test_chapter_summarization_e2e():
    """End-to-end test with real ADK agents."""
    pipeline = create_chapter_summarization_pipeline()
    
    result = await pipeline.run({
        "chapter_text": SAMPLE_CHAPTER,
        "book_id": "test-book-001",
    })
    
    assert result["temp:validation_passed"] is True
    assert "summary" in result["temp:summary"]
```

---

## Rollback Plan

### Feature Flags

```python
# config/settings.py
ADK_ENABLED = os.getenv("ADK_ENABLED", "false").lower() == "true"

# Usage
if settings.ADK_ENABLED:
    orchestrator = ADKPipelineOrchestrator()
else:
    orchestrator = LegacyPipelineOrchestrator()
```

### Rollback Steps

1. Set `ADK_ENABLED=false` in environment
2. Restart ai-agents service
3. Verify legacy orchestrator active
4. Monitor for issues

### Data Compatibility

ADK state uses same underlying storage (Redis). Rollback preserves:
- `user:` prefixed keys
- `app:` prefixed artifacts
- Only `temp:` keys lost (expected, as they're ephemeral)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-29 | Initial ADK migration guide for Option A |

---

## References

### Internal Documentation
- [AGENT_FUNCTIONS_ARCHITECTURE.md](AGENT_FUNCTIONS_ARCHITECTURE.md) - Current architecture with Option C patterns
- [AI_CODING_PLATFORM_ARCHITECTURE.md](../../textbooks/pending/platform/AI_CODING_PLATFORM_ARCHITECTURE.md) - Kitchen Brigade model
- [CODING_PATTERNS_ANALYSIS.md](../../textbooks/Guidelines/CODING_PATTERNS_ANALYSIS.md) - Anti-pattern catalog

### External Documentation
- [Google ADK Documentation](https://google.github.io/adk-docs/) - Official ADK docs
- [ADK Multi-Agent Systems](https://google.github.io/adk-docs/agents/multi-agents/) - Workflow agents
- [ADK Context Management](https://google.github.io/adk-docs/context/) - State and context
- [ADK Artifacts](https://google.github.io/adk-docs/artifacts/) - Artifact service
- [A2A Protocol](https://google.github.io/adk-docs/a2a/) - Agent-to-Agent protocol
