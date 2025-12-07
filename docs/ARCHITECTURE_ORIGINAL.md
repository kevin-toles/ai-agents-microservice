# AI Agents Microservice

## Overview

The AI Agents service is a **microservice** that exposes specialized AI agents (Code Review, Architecture Analysis, Doc Generation) via REST APIs. Applications call these endpoints to perform agent tasks. The agents internally use the LLM Gateway and Semantic Search microservices.

## Architecture Type

**Microservice** - Independently deployable, stateless, horizontally scalable. Agents are exposed as API endpoints, not as libraries.

---

## Folder Structure

```
ai-agents/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── code_review.py       # POST /v1/agents/code-review
│   │   │   ├── architecture.py      # POST /v1/agents/architecture
│   │   │   ├── doc_generate.py      # POST /v1/agents/doc-generate
│   │   │   └── health.py            # /health, /ready
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   └── logging.py
│   │   └── deps.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Pydantic settings
│   │   └── exceptions.py
│   │
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── gateway_client.py        # HTTP client for llm-gateway
│   │   └── search_client.py         # HTTP client for semantic-search
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract agent interface
│   │   │
│   │   ├── code_review/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py             # Code Review Agent implementation
│   │   │   ├── prompts.py           # System prompts and templates
│   │   │   └── tools/
│   │   │       ├── __init__.py
│   │   │       ├── codebert.py      # CodeBERT embedding tool
│   │   │       ├── graphcodebert.py # GraphCodeBERT analysis
│   │   │       └── codet5.py        # CodeT5 summarization
│   │   │
│   │   ├── architecture/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py             # Architecture Agent implementation
│   │   │   ├── prompts.py
│   │   │   └── tools/
│   │   │       ├── __init__.py
│   │   │       ├── sarif_parser.py  # SARIF report parsing
│   │   │       ├── caesar.py        # CAESAR security analysis
│   │   │       └── dependency.py    # Dependency graph analysis
│   │   │
│   │   └── doc_generate/
│   │       ├── __init__.py
│   │       ├── agent.py             # Doc Generate Agent implementation
│   │       ├── prompts.py
│   │       └── tools/
│   │           ├── __init__.py
│   │           ├── ast_parser.py    # AST extraction
│   │           └── docstring.py     # Docstring analysis
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py              # Tool registry for all agents
│   │   └── executor.py              # Tool execution logic
│   │
│   ├── formatters/
│   │   ├── __init__.py
│   │   ├── sarif.py                 # SARIF output formatter
│   │   ├── markdown.py              # Markdown output formatter
│   │   └── json.py                  # JSON output formatter
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py              # API request models
│   │   ├── responses.py             # API response models
│   │   └── domain.py                # Domain models
│   │
│   └── main.py                      # FastAPI app entry point
│
├── tests/
│   ├── unit/
│   │   ├── test_agents/
│   │   │   ├── test_code_review.py
│   │   │   ├── test_architecture.py
│   │   │   └── test_doc_generate.py
│   │   └── test_tools/
│   ├── integration/
│   │   ├── test_code_review_api.py
│   │   └── test_architecture_api.py
│   └── conftest.py
│
├── data/
│   └── models/                      # Cached CodeBERT/CodeT5 models (gitignored)
│
├── docs/
│   ├── ARCHITECTURE.md              # This file
│   ├── API.md                       # API documentation
│   ├── CODE_REVIEW_AGENT.md         # Code Review Agent details
│   ├── ARCHITECTURE_AGENT.md        # Architecture Agent details
│   └── DOC_GENERATE_AGENT.md        # Doc Generate Agent details
│
├── scripts/
│   ├── start.sh
│   └── download_models.py           # Pre-download CodeBERT/CodeT5
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## System Context

```
                          ┌─────────────────────────────────────────┐
                          │            CONSUMERS                     │
                          │                                          │
                          │  ┌────────────┐  ┌────────────────────┐ │
                          │  │ CI/CD      │  │ IDE Extensions     │ │
                          │  │ Pipelines  │  │ (VS Code, etc.)    │ │
                          │  └─────┬──────┘  └─────────┬──────────┘ │
                          │        │                   │            │
                          │        │   ┌───────────────┘            │
                          │        │   │  ┌────────────────────┐   │
                          │        │   │  │ llm-doc-enhancer   │   │
                          │        │   │  └─────────┬──────────┘   │
                          └────────┼───┼────────────┼──────────────┘
                                   │   │            │
                                   ▼   ▼            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         AI AGENTS MICROSERVICE                                │
│                             (Port 8082)                                       │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           API Layer (FastAPI)                            │ │
│  │  POST /v1/agents/code-review  │  POST /v1/agents/architecture  │ ...    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                             AGENTS                                    │    │
│  │                                                                       │    │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐          │    │
│  │  │  Code Review   │  │  Architecture  │  │  Doc Generate  │          │    │
│  │  │  Agent         │  │  Agent         │  │  Agent         │          │    │
│  │  │                │  │                │  │                │          │    │
│  │  │ Tools:         │  │ Tools:         │  │ Tools:         │          │    │
│  │  │ • CodeBERT     │  │ • SARIF Parser │  │ • AST Parser   │          │    │
│  │  │ • GraphCode    │  │ • CAESAR       │  │ • Docstring    │          │    │
│  │  │ • CodeT5       │  │ • Dependency   │  │   Analyzer     │          │    │
│  │  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘          │    │
│  │          │                   │                   │                    │    │
│  └──────────┼───────────────────┼───────────────────┼────────────────────┘    │
│             │                   │                   │                         │
│             └───────────────────┼───────────────────┘                         │
│                                 │                                             │
│  ┌──────────────────────────────┴──────────────────────────────────────────┐ │
│  │                          HTTP Clients                                    │ │
│  │  ┌─────────────────────────┐  ┌─────────────────────────────────────┐   │ │
│  │  │ GatewayClient           │  │ SearchClient                        │   │ │
│  │  │ → llm-gateway:8080      │  │ → semantic-search:8081              │   │ │
│  │  └─────────────────────────┘  └─────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
         ┌──────────────────┐          ┌──────────────────┐
         │ llm-gateway      │          │ semantic-search  │
         │ microservice     │          │ microservice     │
         │ (Port 8080)      │          │ (Port 8081)      │
         └──────────────────┘          └──────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/agents/code-review` | Run code review on diff/PR |
| POST | `/v1/agents/architecture` | Run architecture analysis |
| POST | `/v1/agents/doc-generate` | Generate documentation |
| GET | `/v1/agents` | List available agents |
| GET | `/health` | Health check |
| GET | `/ready` | Readiness check |

---

## Agents

### Code Review Agent
- **Input**: Code diff, PR metadata
- **Tools**: CodeBERT (embeddings), GraphCodeBERT (data flow), CodeT5 (summarization)
- **Output**: Review comments with line annotations, SARIF format optional

### Architecture Agent
- **Input**: Repository path, SARIF reports from static analyzers
- **Tools**: SARIF Parser, CAESAR (security), Dependency Graph Builder
- **Output**: Architecture assessment, security findings, refactoring suggestions

### Doc Generate Agent
- **Input**: Source files
- **Tools**: AST Parser, Docstring Analyzer
- **Output**: Generated documentation (docstrings, README sections, API docs)

---

## Code-Specific Tools

| Tool | Model | Purpose |
|------|-------|---------|
| CodeBERT | `microsoft/codebert-base` | Code embeddings, similarity |
| GraphCodeBERT | `microsoft/graphcodebert-base` | Data flow analysis |
| CodeT5 | `Salesforce/codet5-base` | Code summarization |
| SARIF Parser | N/A | Parse static analysis results |
| CAESAR | N/A | Security analysis |

---

## Dependencies

| Dependency | Type | Purpose |
|------------|------|---------|
| llm-gateway | Microservice | LLM inference with tool-use |
| semantic-search-service | Microservice | Code similarity search |
| HuggingFace Hub | External | CodeBERT/CodeT5 models |

---

## Deployment

```yaml
# docker-compose.yml
services:
  ai-agents:
    build: .
    ports:
      - "8082:8082"
    volumes:
      - ./data/models:/data/models
    environment:
      - LLM_GATEWAY_URL=http://llm-gateway:8080
      - SEMANTIC_SEARCH_URL=http://semantic-search:8081
      - MODEL_CACHE_PATH=/data/models
    depends_on:
      - llm-gateway
      - semantic-search
```

---

## Configuration

```python
# src/core/config.py
class Settings(BaseSettings):
    # Service
    service_name: str = "ai-agents"
    port: int = 8082
    
    # Microservice URLs
    llm_gateway_url: str = "http://localhost:8080"
    semantic_search_url: str = "http://localhost:8081"
    
    # Model paths
    model_cache_path: str = "/data/models"
    
    # Agent settings
    default_llm_model: str = "claude-3-sonnet-20240229"
    code_review_max_diff_size: int = 10000
    
    class Config:
        env_prefix = "AI_AGENTS_"
```
