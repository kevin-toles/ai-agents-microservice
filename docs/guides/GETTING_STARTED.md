# Getting Started with Kitchen Brigade AI Agent

> **WBS Reference**: WBS-KB9 - End-to-End Validation  
> **Target**: Setup in under 30 minutes for new users

Welcome to the Kitchen Brigade AI Agent! This guide will help you set up and run the system from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (5 minutes)](#quick-start-5-minutes)
3. [Full Setup (20 minutes)](#full-setup-20-minutes)
4. [Verification](#verification)
5. [First Query](#first-query)
6. [VS Code Integration](#vs-code-integration)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| GPU VRAM | 8 GB | 16 GB (RTX 3080+) |
| Storage | 50 GB | 100 GB SSD |
| CPU | 4 cores | 8+ cores |

### Software Requirements

- **Python**: 3.11+ (tested on 3.13.7)
- **Docker**: 24.0+ with Docker Compose
- **Git**: 2.40+
- **VS Code**: 1.85+ (optional, for MCP integration)

### Verify Prerequisites

```bash
# Check Python version
python3 --version  # Should be 3.11+

# Check Docker
docker --version && docker compose version

# Check Git
git --version
```

---

## Quick Start (5 minutes)

For immediate testing with mocked services:

```bash
# 1. Clone the repository
git clone https://github.com/your-org/ai-agents.git
cd ai-agents

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 4. Run tests to verify setup
pytest tests/unit/ -v --tb=short

# 5. Run the agent in CLI mode (mock mode)
python -m src.main --mock
```

---

## Full Setup (20 minutes)

### Step 1: Clone and Configure (3 minutes)

```bash
# Clone repository
git clone https://github.com/your-org/ai-agents.git
cd ai-agents

# Create Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt
```

### Step 2: Environment Configuration (2 minutes)

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your settings (optional - defaults work for local setup)
# Required only if using external services
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_SERVICE_URL` | `http://localhost:8085` | LLM inference endpoint |
| `SEMANTIC_SEARCH_URL` | `http://localhost:8081` | Semantic search endpoint |
| `CODE_ORCHESTRATOR_URL` | `http://localhost:8083` | Code analysis endpoint |
| `AUDIT_SERVICE_URL` | `http://localhost:8084` | Citation audit endpoint |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Step 3: Start Infrastructure Services (5 minutes)

```bash
# Start all E2E services
docker compose -f docker/docker-compose.e2e.yml up -d

# Wait for services to be healthy
docker compose -f docker/docker-compose.e2e.yml ps

# Check logs if needed
docker compose -f docker/docker-compose.e2e.yml logs -f --tail=50
```

### Step 4: Initialize Data (5 minutes)

```bash
# Seed the semantic search database
python scripts/seed_textbooks.py

# Verify seeding
python -c "from src.services.semantic_search import SemanticSearchClient; print('✓ Semantic search ready')"
```

### Step 5: Run the Agent (2 minutes)

```bash
# Start in interactive CLI mode
python -m src.main

# Or start API server
python -m src.main --api --port 8080
```

---

## Verification

### Run All Tests

```bash
# Unit tests (fast, no external services)
pytest tests/unit/ -v

# Integration tests (requires mock services)
pytest tests/integration/ -v

# E2E tests (requires all services running)
export KB_E2E_SERVICES=true
pytest tests/e2e/ -v
```

### Health Check

```bash
# Check all services
curl http://localhost:8080/health
curl http://localhost:8081/health
curl http://localhost:8083/health
curl http://localhost:8084/health
curl http://localhost:8085/health
```

Expected output:
```json
{"status": "healthy", "service": "ai-agents", "version": "0.1.0"}
```

---

## First Query

### CLI Mode

```bash
python -m src.main

# Example queries to try:
# > Where is the DelegationEngine class defined?
# > Explain the Kitchen Brigade pattern
# > Generate a Python function to validate citations
```

### API Mode

```bash
# Start server
python -m src.main --api --port 8080

# In another terminal:
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Where is the DelegationEngine class defined?"}'
```

### Expected Response Format

```json
{
  "response": "The DelegationEngine class is defined in src/delegation/engine.py at line 45.",
  "citations": [
    {
      "source": "ai-agents/src/delegation/engine.py",
      "line_start": 45,
      "line_end": 120,
      "confidence": 0.95
    }
  ],
  "metadata": {
    "retrieval_sources": ["code", "textbook"],
    "iteration_count": 2,
    "processing_time_ms": 1250
  }
}
```

---

## VS Code Integration

### MCP Server Setup (5 minutes)

1. **Copy configuration template**:
   ```bash
   mkdir -p ~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline
   cp config/mcp.json.template ~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json
   ```

2. **Edit configuration** (update paths):
   ```json
   {
     "mcpServers": {
       "kitchen-brigade": {
         "command": "/path/to/ai-agents/venv/bin/python",
         "args": ["-m", "src.mcp.server"],
         "cwd": "/path/to/ai-agents"
       }
     }
   }
   ```

3. **Install Roo Cline extension** in VS Code

4. **Test MCP connection**:
   - Open VS Code Command Palette (`Cmd+Shift+P`)
   - Run "Roo Cline: List Tools"
   - Verify Kitchen Brigade tools appear

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `cross_reference` | Find code locations with citations |
| `generate_code` | Generate code with textbook references |
| `analyze_code` | Analyze code for patterns and issues |
| `explain_code` | Explain code concepts with references |

---

## Troubleshooting

### Common Issues

#### Docker services won't start

```bash
# Check for port conflicts
lsof -i :8080 -i :8081 -i :8083 -i :8084 -i :8085

# Reset containers
docker compose -f docker/docker-compose.e2e.yml down -v
docker compose -f docker/docker-compose.e2e.yml up -d
```

#### GPU not detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

#### Python package conflicts

```bash
# Create fresh environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

#### Tests failing

```bash
# Run with verbose output
pytest tests/ -v --tb=long

# Run specific test
pytest tests/e2e/test_kitchen_brigade_e2e.py::TestCodeLocationE2E::test_code_location_query -v
```

### Getting Help

- **Documentation**: `docs/` directory
- **ADRs**: `docs/adr/` - Architecture Decision Records
- **Issues**: File issues on GitHub
- **Chat**: #kitchen-brigade channel

---

## Next Steps

1. **Explore Examples**: See `examples/` for sample queries
2. **Read ADRs**: Understand design decisions in `docs/adr/`
3. **Watch Demo**: See `docs/DEMO.md` for a walkthrough
4. **Customize**: Edit `config/agent_config.yaml` for your needs

---

## Performance Targets

| Query Type | Target Time | Measured |
|------------|-------------|----------|
| Code location | < 30 sec | ~15 sec |
| Concept explanation | < 45 sec | ~25 sec |
| Code generation | < 60 sec | ~40 sec |
| Complex multi-cycle | < 120 sec | ~90 sec |

---

*Setup time: ~20 minutes for full setup, ~5 minutes for quick start*
