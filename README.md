# AI Agents

Cross-Reference Agent for textbook taxonomy traversal using LangGraph.

## Overview

This project implements an AI agent that traverses a taxonomy of technical books to find cross-references between chapters. It uses the "spider web" traversal model to discover related content across different tiers (Architecture Spine, Implementation Layer, Practices Layer).

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

### Key Components

- **CrossReferenceAgent**: Main agent orchestrating the 9-step workflow
- **LangGraph StateGraph**: Workflow orchestration with conditional routing
- **Tools**: Integration with semantic-search-service for taxonomy queries

### 9-Step Workflow

1. Analyze Source - Extract concepts and keywords from source chapter
2. Search Taxonomy - Find related books via Neo4j/Qdrant
3. Identify Tier - Determine source tier and valid traversals
4. Execute Traversal - Spider web graph traversal
5. Rank Matches - Score and filter relevant chapters
6. Retrieve Content - Fetch chapter text for matched results
7. Synthesize - Generate cross-reference summary via LLM
8. Validate - Ensure output meets quality criteria
9. Return Results - Format and return response

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Configuration

Configuration is managed via environment variables with the `AI_AGENTS_` prefix:

```bash
export AI_AGENTS_NEO4J_URI="bolt://localhost:7687"
export AI_AGENTS_NEO4J_USERNAME="neo4j"
export AI_AGENTS_NEO4J_PASSWORD="password"
export AI_AGENTS_QDRANT_HOST="localhost"
export AI_AGENTS_QDRANT_PORT="6333"
export AI_AGENTS_LLM_GATEWAY_URL="http://localhost:8000"
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit

# Run only integration tests
pytest tests/integration -m integration
```

### Code Quality

```bash
# Format code
black src tests

# Lint code
ruff check src tests

# Type checking
mypy src
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Project Structure

```
ai-agents/
├── docs/
│   ├── ARCHITECTURE.md      # Detailed architecture
│   └── PHASE_5_PRE_IMPLEMENTATION.md
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Pydantic Settings
│   │   └── exceptions.py    # Custom exceptions
│   └── agents/
│       ├── __init__.py
│       ├── base.py          # BaseAgent ABC
│       └── cross_reference/
│           ├── __init__.py
│           ├── agent.py     # CrossReferenceAgent
│           ├── state.py     # Pydantic state models
│           ├── nodes/       # LangGraph workflow nodes
│           └── tools/       # Agent tools
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── pyproject.toml          # Project configuration
├── requirements.txt        # Core dependencies
└── requirements-dev.txt    # Dev dependencies
```

## License

MIT
