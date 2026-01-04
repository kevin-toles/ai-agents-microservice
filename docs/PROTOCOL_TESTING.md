# Protocol Integration Testing

> **Version:** 1.0.0  
> **Created:** 2026-01-03  
> **WBS Reference:** WBS-PI7  
> **Status:** Complete

---

## Overview

This document describes the testing strategy and procedures for the Protocol Integration Architecture (A2A + MCP).

## Test Categories

### 1. Unit Tests

| Location | Coverage | Description |
|----------|----------|-------------|
| `tests/unit/config/test_feature_flags.py` | Feature Flags | All 11 feature flags and their interactions |
| `tests/unit/a2a/test_models.py` | A2A Models | AgentCard, Skill, Task, Message models |
| `tests/unit/a2a/test_agent_card.py` | Agent Card | Card generation from registry |
| `tests/unit/a2a/test_events.py` | A2A Events | SSE event models |
| `tests/unit/a2a/test_task_store.py` | Task Store | In-memory task persistence |
| `tests/unit/mcp/test_server.py` | MCP Server | Tool listing and execution |
| `tests/unit/mcp/test_semantic_search_wrapper.py` | Semantic Search | Hybrid RAG wrapper |
| `tests/unit/mcp/test_toolbox_manager.py` | Toolbox Manager | Neo4j/Redis toolsets |

### 2. Integration Tests

| Location | Coverage | Description |
|----------|----------|-------------|
| `tests/integration/test_mcp_lifecycle.py` | MCP Lifecycle | Server startup/shutdown |
| `tests/integration/test_mcp_client.py` | MCP Client | Client initialization |
| `tests/integration/test_phase1_unchanged.py` | Phase 1 Stability | Phase 1 endpoints always work |

### 3. E2E Tests

| Location | AC | Description |
|----------|-----|-------------|
| `tests/e2e/test_a2a_schema.py` | AC-PI7.1 | Agent Card schema validation |
| `tests/e2e/test_feature_flags.py` | AC-PI7.7-9 | Feature flag matrix testing |

---

## Running Tests

### Full Protocol Integration Suite

```bash
# All unit tests
pytest tests/unit/config/test_feature_flags.py tests/unit/a2a/ tests/unit/mcp/ -v

# All integration tests
pytest tests/integration/test_mcp_*.py tests/integration/test_phase1_unchanged.py -v

# All E2E tests
pytest tests/e2e/ -v

# Complete suite
pytest tests/unit/config/test_feature_flags.py \
       tests/unit/a2a/ \
       tests/unit/mcp/ \
       tests/integration/test_mcp_*.py \
       tests/integration/test_phase1_unchanged.py \
       tests/e2e/test_feature_flags.py \
       tests/e2e/test_a2a_schema.py -v
```

### Test by Feature

```bash
# A2A Protocol only
pytest tests/unit/a2a/ tests/e2e/test_a2a_schema.py -v

# MCP Protocol only
pytest tests/unit/mcp/ tests/integration/test_mcp_*.py -v

# Feature Flags only
pytest tests/unit/config/test_feature_flags.py tests/e2e/test_feature_flags.py -v
```

---

## Feature Flag Test Matrix

The E2E tests verify all combinations of feature flag states:

| State | A2A | MCP | Agent Card | A2A Endpoints | Phase 1 |
|-------|-----|-----|------------|---------------|---------|
| All Disabled | ❌ | ❌ | 404 | 501 | ✅ 200 |
| All Enabled | ✅ | ✅ | 200 | 200 | ✅ 200 |
| A2A Only | ✅ | ❌ | 200 | 200 | ✅ 200 |
| MCP Only | ❌ | ✅ | 404 | 501 | ✅ 200 |

**Key Invariant:** Phase 1 endpoints always return 200 regardless of Phase 2 state.

---

## Rollback Verification

### Using the Script

```bash
# Against local service
python scripts/verify_rollback.py --base-url http://localhost:8082

# Against Docker service
python scripts/verify_rollback.py --base-url http://ai-agents:8082
```

### Expected Output (Passed)

```
🔍 Verifying rollback at http://localhost:8082...

============================================================
ROLLBACK VERIFICATION RESULTS
============================================================
✅ Agent Card Disabled
   /.well-known/agent-card.json returns 404
✅ A2A message:send Disabled
   POST /a2a/v1/message:send returns 501
✅ A2A message:stream Disabled
   POST /a2a/v1/message:stream returns 501
✅ A2A Task Status Disabled
   GET /a2a/v1/tasks/{id} returns 501
✅ Phase 1 Functions List
   GET /v1/functions returns 200 with functions
✅ Phase 1 Extract Structure
   POST /v1/functions/extract-structure/run returns 200
✅ Health Endpoint
   GET /health returns 200
============================================================
TOTAL: 7 passed, 0 failed
============================================================

✅ Rollback verification passed
```

---

## CI/CD Integration

The GitHub Actions workflow (`.github/workflows/protocol_tests.yml`) runs on every PR:

1. **Unit Tests** - Feature flags, A2A models, MCP server
2. **Integration Tests** - MCP lifecycle, Phase 1 stability
3. **E2E Tests (Disabled)** - Verify disabled state behavior
4. **E2E Tests (Enabled)** - Verify enabled state behavior
5. **Rollback Verification** - Verify rollback script works
6. **Coverage Report** - Upload to Codecov

---

## Troubleshooting

### Tests Fail: "Module not found"

Ensure you're running from the ai-agents directory:

```bash
cd /Users/kevintoles/POC/ai-agents
python -m pytest tests/e2e/ -v
```

### Tests Fail: Feature Flags Not Applied

The E2E fixtures reload the app with fresh environment variables. If this fails, check:

1. `conftest.py` uses `_create_fresh_app_with_env()` helper
2. No cached imports in test functions

### Rollback Script Fails

1. Ensure the service is running
2. Check the base URL is correct
3. Verify all AGENTS_* env vars are set to false

---

## Acceptance Criteria Coverage

| AC ID | Requirement | Test File |
|-------|-------------|-----------|
| AC-PI7.1 | Agent Card validates against A2A schema | `test_a2a_schema.py` |
| AC-PI7.7 | All disabled → 501/404 | `test_feature_flags.py` |
| AC-PI7.8 | Partial enablement works | `test_feature_flags.py` |
| AC-PI7.9 | Phase 1 always returns 200 | `test_feature_flags.py` |
| AC-PI7.10 | Rollback script verifies | `scripts/verify_rollback.py` |

---

## References

- [PROTOCOL_INTEGRATION_ARCHITECTURE.md](architecture/Future/PROTOCOL_INTEGRATION_ARCHITECTURE.md)
- [WBS_PROTOCOL_INTEGRATION.md](WBS_PROTOCOL_INTEGRATION.md)
- [A2A Protocol Specification](https://a2a-protocol.org/latest/specification/)
- [MCP Documentation](https://modelcontextprotocol.io/)
