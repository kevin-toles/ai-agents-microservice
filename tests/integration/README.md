# Integration Tests

WBS Reference: **WBS-AGT20 Integration Testing**

This directory contains integration tests for the ai-agents service, testing the complete system with external service dependencies.

## Overview

The integration tests verify:

| AC | Description | Test File |
|----|-------------|-----------|
| AC-20.1 | E2E test: function request → response | `test_functions.py` |
| AC-20.2 | E2E test: pipeline request → response with citations | `test_pipelines.py` |
| AC-20.3 | Service integration: ai-agents → inference-service | `test_inference.py` |
| AC-20.4 | Service integration: ai-agents → semantic-search-service | `test_semantic_search.py` |
| AC-20.5 | Service integration: ai-agents → audit-service | `test_audit.py` |
| AC-20.6 | Load test: 5 concurrent pipeline requests | `test_load.py` |

## Prerequisites

### Required Services

The integration tests require the following services to be running:

| Service | Default URL | Environment Variable |
|---------|-------------|---------------------|
| ai-agents | http://localhost:8082 | `AI_AGENTS_URL` |
| inference-service | http://localhost:8085 | `INFERENCE_SERVICE_URL` |
| semantic-search-service | http://localhost:8081 | `SEMANTIC_SEARCH_URL` |
| audit-service | http://localhost:8084 | `AUDIT_SERVICE_URL` |

### Environment Variables

Control which live services are tested:

```bash
# Enable individual service tests
export LIVE_INFERENCE=true
export LIVE_SEMANTIC_SEARCH=true
export LIVE_AUDIT=true

# Or enable all at once
export LIVE_ALL_SERVICES=true
```

## Running Tests

### Run All Integration Tests (Non-Slow)

```bash
pytest tests/integration/ -m "not slow" -v
```

### Run With Docker Compose

```bash
# Start test environment
docker-compose -f docker/docker-compose.test.yml up -d

# Wait for services to be healthy
sleep 10

# Run tests
export LIVE_ALL_SERVICES=true
pytest tests/integration/ -m "not slow" -v

# Stop environment
docker-compose -f docker/docker-compose.test.yml down
```

### Run Specific Test Files

```bash
# Function E2E tests
pytest tests/integration/test_functions.py -v

# Pipeline E2E tests
pytest tests/integration/test_pipelines.py -v

# Service integration tests
pytest tests/integration/test_inference.py -v
pytest tests/integration/test_semantic_search.py -v
pytest tests/integration/test_audit.py -v

# Load tests (marked slow)
pytest tests/integration/test_load.py -v
```

### Run Load Tests

Load tests are marked as `slow` and are excluded by default:

```bash
# Run only load tests
pytest tests/integration/test_load.py -v -m slow

# Run all tests including slow ones
pytest tests/integration/ -v
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

### HTTP Clients

- `http_client` - Generic async HTTP client
- `inference_client` - Client for inference-service
- `semantic_search_client` - Client for semantic-search-service
- `audit_client` - Client for audit-service
- `ai_agents_client` - Client for ai-agents service

### Service Health Fixtures

These fixtures check if services are available and skip tests if not:

- `ensure_inference_service` - Skip if inference-service unavailable
- `ensure_semantic_search` - Skip if semantic-search unavailable
- `ensure_audit_service` - Skip if audit-service unavailable
- `ensure_ai_agents` - Skip if ai-agents unavailable

### Test Data

- `sample_function_input` - Sample input for function invocation
- `sample_pipeline_input` - Sample input for pipeline execution

## Test Categories

### E2E Tests

Full end-to-end tests that verify complete request/response flows:

```python
class TestFunctionE2E:
    async def test_function_invoke_e2e(self, ensure_ai_agents):
        """Invoke function and verify response structure."""
```

### Service Integration Tests

Tests that verify communication between services:

```python
class TestInferenceServiceIntegration:
    async def test_completion_request(self, ensure_inference_service):
        """Verify ai-agents can reach inference-service."""
```

### Load Tests

Performance and concurrency tests:

```python
class TestConcurrentPipelines:
    async def test_five_concurrent_pipeline_requests(self, ensure_ai_agents):
        """AC-20.6: 5 concurrent pipelines complete within 60s."""
```

## Exit Criteria

Per WBS-AGT20, the integration tests pass when:

1. ✅ `pytest tests/integration/ -m "not slow"` passes
2. ✅ Chapter summarization pipeline returns CitedContent with footnotes
3. ✅ inference-service:8085 receives completion requests
4. ✅ semantic-search-service:8081 receives query requests
5. ✅ audit-service:8084 receives citation audit records
6. ✅ 5 concurrent pipelines complete within 60s timeout

## Troubleshooting

### Tests Skipped

If tests are being skipped, check:

1. **Service availability**: Verify services are running
   ```bash
   curl http://localhost:8082/health
   curl http://localhost:8085/health
   curl http://localhost:8081/health
   curl http://localhost:8084/health
   ```

2. **Environment variables**: Ensure `LIVE_*` variables are set
   ```bash
   export LIVE_ALL_SERVICES=true
   ```

### Connection Errors

If seeing connection refused errors:

1. Check Docker containers are running
   ```bash
   docker-compose -f docker/docker-compose.test.yml ps
   ```

2. Check network connectivity
   ```bash
   docker network ls | grep ai-agents-test
   ```

### Timeout Errors

If tests timeout:

1. Increase timeout in fixtures (see `conftest.py`)
2. Check service health endpoints
3. Review service logs for errors

## Related Documentation

- [AGENT_FUNCTIONS_ARCHITECTURE.md](../../docs/AGENT_FUNCTIONS_ARCHITECTURE.md) - Agent function specifications
- [WBS.md](../../docs/WBS.md) - Work breakdown structure
- [README.md](../../README.md) - Project overview
