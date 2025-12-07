# AI Agents Microservice

## Overview

The AI Agents service is a **microservice** that exposes specialized AI agents via REST APIs. Applications call these endpoints to perform agent tasks. The agents internally use the LLM Gateway, Semantic Search, and Graph Database (Neo4j) microservices.

**This service is the core of the Unified Agent Platform** - supporting both batch processing (llm-document-enhancer) and interactive use cases (VS Code Copilot, IDE extensions).

## Architecture Type

**Microservice** - Independently deployable, stateless, horizontally scalable. Agents are exposed as API endpoints, not as libraries.

---

## Unified Platform Vision

This service supports multiple consumers with the same underlying agent infrastructure:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED AGENT PLATFORM                                   │
│                                                                                  │
│  Same agents, same tools, different consumers                                   │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                           CONSUMERS                                      │    │
│  │                                                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐   │    │
│  │  │ llm-doc-     │  │ VS Code      │  │ CI/CD        │  │ IDE        │   │    │
│  │  │ enhancer     │  │ Copilot      │  │ Pipelines    │  │ Extensions │   │    │
│  │  │              │  │              │  │              │  │            │   │    │
│  │  │ Batch        │  │ Interactive  │  │ Automated    │  │ Real-time  │   │    │
│  │  │ Processing   │  │ Agent        │  │ Review       │  │ Assist     │   │    │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘   │    │
│  │         │                 │                 │                │          │    │
│  │         └─────────────────┴─────────────────┴────────────────┘          │    │
│  │                                    │                                     │    │
│  └────────────────────────────────────┼─────────────────────────────────────┘    │
│                                       ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    AI AGENTS SERVICE (Port 8082)                         │    │
│  │                                                                          │    │
│  │  LangChain + LangGraph Orchestration                                    │    │
│  │                                                                          │    │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │    │
│  │  │ Cross-Reference│  │ Code Review    │  │ Architecture   │            │    │
│  │  │ Agent          │  │ Agent          │  │ Agent          │            │    │
│  │  └────────────────┘  └────────────────┘  └────────────────┘            │    │
│  │  ┌────────────────┐                                                     │    │
│  │  │ Doc Generate   │  ... (future agents)                               │    │
│  │  │ Agent          │                                                     │    │
│  │  └────────────────┘                                                     │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                          │
│              ┌────────────────────────┼────────────────────────┐                │
│              ▼                        ▼                        ▼                │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│  │   LLM Gateway    │    │ Semantic Search  │    │     Neo4j        │          │
│  │   (Port 8080)    │    │  + Qdrant        │    │   (Graph DB)     │          │
│  │                  │    │  (Port 8081)     │    │   (Port 7687)    │          │
│  │  • LLM inference │    │  • Embeddings    │    │  • Taxonomy      │          │
│  │  • Tool execution│    │  • Vector search │    │  • Relationships │          │
│  │  • Sessions      │    │  • Hybrid search │    │  • Graph queries │          │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
ai-agents/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── cross_reference.py   # POST /v1/agents/cross-reference (NEW)
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
│   │   ├── search_client.py         # HTTP client for semantic-search
│   │   └── graph_client.py          # HTTP client for Neo4j (NEW)
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract agent interface
│   │   ├── orchestrator.py          # LangChain/LangGraph orchestrator (NEW)
│   │   │
│   │   ├── cross_reference/         # NEW - Taxonomy Cross-Reference Agent
│   │   │   ├── __init__.py
│   │   │   ├── agent.py             # Cross-Reference Agent implementation
│   │   │   ├── prompts.py           # System prompts and templates
│   │   │   └── tools/
│   │   │       ├── __init__.py
│   │   │       ├── taxonomy.py      # search_taxonomy() tool
│   │   │       ├── similarity.py    # search_similar() tool
│   │   │       ├── metadata.py      # get_chapter_metadata() tool
│   │   │       ├── content.py       # get_chapter_text() tool
│   │   │       └── graph.py         # traverse_graph() tool
│   │   │
│   │   ├── code_review/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── prompts.py
│   │   │   └── tools/
│   │   │       ├── __init__.py
│   │   │       ├── codebert.py
│   │   │       ├── graphcodebert.py
│   │   │       └── codet5.py
│   │   │
│   │   ├── architecture/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── prompts.py
│   │   │   └── tools/
│   │   │       ├── __init__.py
│   │   │       ├── sarif_parser.py
│   │   │       ├── caesar.py
│   │   │       └── dependency.py
│   │   │
│   │   └── doc_generate/
│   │       ├── __init__.py
│   │       ├── agent.py
│   │       ├── prompts.py
│   │       └── tools/
│   │           ├── __init__.py
│   │           ├── ast_parser.py
│   │           └── docstring.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py              # Tool registry for all agents
│   │   ├── executor.py              # Tool execution logic
│   │   └── definitions.py           # Tool function schemas (NEW)
│   │
│   ├── graph/                       # NEW - Graph/Taxonomy Support
│   │   ├── __init__.py
│   │   ├── neo4j_client.py          # Neo4j connection and queries
│   │   ├── taxonomy.py              # Taxonomy graph operations
│   │   └── traversal.py             # Spider web traversal algorithms
│   │
│   ├── formatters/
│   │   ├── __init__.py
│   │   ├── sarif.py
│   │   ├── markdown.py
│   │   ├── json.py
│   │   └── chicago.py               # Chicago-style citation formatter (NEW)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   ├── responses.py
│   │   ├── domain.py
│   │   └── taxonomy.py              # Taxonomy/tier models (NEW)
│   │
│   └── main.py
│
├── tests/
│   ├── unit/
│   │   ├── test_agents/
│   │   │   ├── test_cross_reference.py  Instructions
│   │   │   ├── test_code_review.py
│   │   │   ├── test_architecture.py
│   │   │   └── test_doc_generate.py
│   │   ├── test_tools/
│   │   └── test_graph/                  # NEW
│   ├── integration/
│   │   ├── test_cross_reference_api.py  # NEW
│   │   ├── test_code_review_api.py
│   │   └── test_architecture_api.py
│   └── conftest.py
│
├── data/
│   └── models/
│
├── docs/
│   ├── ARCHITECTURE.md              # This file
│   ├── ARCHITECTURE_ORIGINAL.md     # Pre-graph version backup
│   ├── API.md
│   ├── CROSS_REFERENCE_AGENT.md     # NEW
│   ├── CODE_REVIEW_AGENT.md
│   ├── ARCHITECTURE_AGENT.md
│   ├── DOC_GENERATE_AGENT.md
│   └── TOOL_DEFINITIONS.md          # NEW
│
├── scripts/
│   ├── start.sh
│   └── download_models.py
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/agents/cross-reference` | **NEW** - Run taxonomy cross-referencing |
| POST | `/v1/agents/code-review` | Run code review on diff/PR |
| POST | `/v1/agents/architecture` | Run architecture analysis |
| POST | `/v1/agents/doc-generate` | Generate documentation |
| GET | `/v1/agents` | List available agents |
| GET | `/health` | Health check |
| GET | `/ready` | Readiness check |

---

## Agents

### Cross-Reference Agent (NEW)

The flagship agent for taxonomy-aware cross-referencing. Implements the spider web traversal model.

- **Input**: Source chapter, taxonomy configuration, traversal parameters
- **Tools**: 
  - `search_taxonomy()` → Query Neo4j for related books/tiers
  - `search_similar()` → Query Qdrant for similar chapters
  - `get_chapter_metadata()` → Retrieve keywords, concepts, summary
  - `get_chapter_text()` → Retrieve full chapter content
  - `traverse_graph()` → Execute spider web traversal paths
- **Output**: Scholarly annotation with Chicago-style citations, tier-aware structure

### Code Review Agent
- **Input**: Code diff, PR metadata
- **Tools**: CodeBERT, GraphCodeBERT, CodeT5
- **Output**: Review comments, SARIF format optional

### Architecture Agent
- **Input**: Repository path, SARIF reports
- **Tools**: SARIF Parser, CAESAR, Dependency Graph
- **Output**: Architecture assessment, security findings

### Doc Generate Agent
- **Input**: Source files
- **Tools**: AST Parser, Docstring Analyzer
- **Output**: Generated documentation

---

## Tool Definitions (Function Schemas)

All agents share a common tool registry. Below are the schemas for Cross-Reference Agent tools:

### search_taxonomy

```python
{
    "name": "search_taxonomy",
    "description": "Search the taxonomy graph for books related to a query. Returns books with their tier levels and relationship types.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (keyword, concept, or topic)"
            },
            "source_tier": {
                "type": "integer",
                "description": "The tier of the source book (for relationship calculation)"
            },
            "taxonomy_id": {
                "type": "string",
                "description": "ID of the taxonomy to search within"
            },
            "max_results": {
                "type": "integer",
                "default": 10,
                "description": "Maximum number of results to return"
            }
        },
        "required": ["query", "taxonomy_id"]
    }
}
```

**Returns:**
```json
{
    "results": [
        {
            "book": "Clean Code",
            "tier": 2,
            "relationship_to_source": "PERPENDICULAR",
            "relevance_score": 0.89,
            "chapters_matched": [3, 7, 12]
        }
    ]
}
```

### search_similar

```python
{
    "name": "search_similar",
    "description": "Find chapters semantically similar to the source content using vector search.",
    "parameters": {
        "type": "object",
        "properties": {
            "query_text": {
                "type": "string",
                "description": "Text to find similar content for"
            },
            "top_k": {
                "type": "integer",
                "default": 10,
                "description": "Number of similar results to return"
            },
            "filter_tier": {
                "type": "integer",
                "description": "Optional: Only return results from this tier"
            },
            "min_similarity": {
                "type": "number",
                "default": 0.7,
                "description": "Minimum similarity threshold (0.0-1.0)"
            }
        },
        "required": ["query_text"]
    }
}
```

**Returns:**
```json
{
    "results": [
        {
            "book": "Philosophy of Software Design",
            "chapter": 4,
            "title": "Modules Should Be Deep",
            "similarity": 0.92,
            "tier": 1,
            "keywords": ["abstraction", "interfaces", "complexity"]
        }
    ]
}
```

### get_chapter_metadata

```python
{
    "name": "get_chapter_metadata",
    "description": "Retrieve metadata for a specific chapter including keywords, concepts, and summary.",
    "parameters": {
        "type": "object",
        "properties": {
            "book": {
                "type": "string",
                "description": "Book title"
            },
            "chapter": {
                "type": "integer",
                "description": "Chapter number"
            }
        },
        "required": ["book", "chapter"]
    }
}
```

**Returns:**
```json
{
    "book": "Clean Code",
    "chapter": 3,
    "title": "Functions",
    "keywords": ["small functions", "single responsibility", "arguments"],
    "concepts": ["function design", "readability", "refactoring"],
    "summary": "This chapter covers best practices for writing clean functions...",
    "page_range": "33-58",
    "tier": 2
}
```

### get_chapter_text

```python
{
    "name": "get_chapter_text",
    "description": "Retrieve the full text content of a chapter for detailed analysis.",
    "parameters": {
        "type": "object",
        "properties": {
            "book": {
                "type": "string",
                "description": "Book title"
            },
            "chapter": {
                "type": "integer",
                "description": "Chapter number"
            },
            "pages": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional: Specific pages to retrieve"
            }
        },
        "required": ["book", "chapter"]
    }
}
```

**Returns:**
```json
{
    "book": "Clean Code",
    "chapter": 3,
    "title": "Functions",
    "content": "The first rule of functions is that they should be small...",
    "page_numbers": [33, 34, 35, "..."],
    "word_count": 8500
}
```

### traverse_graph

```python
{
    "name": "traverse_graph",
    "description": "Execute a spider web traversal across the taxonomy graph. Supports bidirectional, skip-tier, and non-linear paths.",
    "parameters": {
        "type": "object",
        "properties": {
            "start_node": {
                "type": "object",
                "properties": {
                    "book": {"type": "string"},
                    "chapter": {"type": "integer"},
                    "tier": {"type": "integer"}
                },
                "required": ["book", "chapter", "tier"]
            },
            "max_hops": {
                "type": "integer",
                "default": 3,
                "description": "Maximum traversal depth"
            },
            "relationship_types": {
                "type": "array",
                "items": {"type": "string", "enum": ["PARALLEL", "PERPENDICULAR", "SKIP_TIER"]},
                "default": ["PARALLEL", "PERPENDICULAR", "SKIP_TIER"],
                "description": "Which relationship types to follow"
            },
            "allow_cycles": {
                "type": "boolean",
                "default": true,
                "description": "Whether to allow revisiting tiers"
            },
            "direction": {
                "type": "string",
                "enum": ["UP", "DOWN", "BOTH"],
                "default": "BOTH",
                "description": "Direction of traversal (UP=higher tiers, DOWN=lower tiers)"
            }
        },
        "required": ["start_node"]
    }
}
```

**Returns:**
```json
{
    "paths": [
        {
            "nodes": [
                {"book": "Python Distilled", "chapter": 7, "tier": 1},
                {"book": "Clean Code", "chapter": 3, "tier": 2, "relationship": "PERPENDICULAR"},
                {"book": "Microservices", "chapter": 8, "tier": 3, "relationship": "PERPENDICULAR"},
                {"book": "Philosophy of SW", "chapter": 9, "tier": 1, "relationship": "SKIP_TIER"}
            ],
            "total_similarity": 0.78,
            "path_type": "non_linear"
        }
    ],
    "traversal_stats": {
        "nodes_visited": 12,
        "unique_books": 5,
        "tiers_covered": [1, 2, 3]
    }
}
```

---

## Cross-Reference Agent Prototype (LangChain)

High-level design for the Cross-Reference Agent using LangChain/LangGraph:

```python
# src/agents/cross_reference/agent.py

from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph

class CrossReferenceAgent:
    """
    Agentic cross-referencing using spider web taxonomy traversal.
    
    This agent:
    1. Reviews source chapter content and metadata
    2. Searches taxonomy for related books/chapters
    3. Traverses the graph following PARALLEL/PERPENDICULAR/SKIP_TIER edges
    4. Retrieves relevant chapter content for validation
    5. Synthesizes scholarly annotations with citations
    """
    
    def __init__(
        self,
        llm_gateway_client: GatewayClient,
        search_client: SearchClient,
        graph_client: GraphClient,
    ):
        self.llm = llm_gateway_client
        self.search = search_client
        self.graph = graph_client
        
        # Register tools
        self.tools = [
            Tool(
                name="search_taxonomy",
                func=self._search_taxonomy,
                description="Search taxonomy for related books"
            ),
            Tool(
                name="search_similar",
                func=self._search_similar,
                description="Find semantically similar chapters"
            ),
            Tool(
                name="get_chapter_metadata",
                func=self._get_metadata,
                description="Get chapter keywords and summary"
            ),
            Tool(
                name="get_chapter_text",
                func=self._get_text,
                description="Get full chapter content"
            ),
            Tool(
                name="traverse_graph",
                func=self._traverse,
                description="Execute spider web traversal"
            ),
        ]
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph state machine for cross-referencing.
        
        States:
        1. analyze_source - Understand source chapter
        2. search_taxonomy - Find related books in taxonomy
        3. traverse_graph - Follow spider web paths
        4. retrieve_content - Get relevant chapter texts
        5. synthesize - Generate annotation with citations
        """
        workflow = StateGraph(CrossReferenceState)
        
        workflow.add_node("analyze_source", self._analyze_source)
        workflow.add_node("search_taxonomy", self._search_taxonomy_node)
        workflow.add_node("traverse_graph", self._traverse_node)
        workflow.add_node("retrieve_content", self._retrieve_node)
        workflow.add_node("synthesize", self._synthesize_node)
        
        workflow.add_edge("analyze_source", "search_taxonomy")
        workflow.add_edge("search_taxonomy", "traverse_graph")
        workflow.add_edge("traverse_graph", "retrieve_content")
        workflow.add_edge("retrieve_content", "synthesize")
        
        workflow.set_entry_point("analyze_source")
        workflow.set_finish_point("synthesize")
        
        return workflow.compile()
    
    async def run(
        self,
        source_chapter: SourceChapter,
        taxonomy: Taxonomy,
        config: TraversalConfig,
    ) -> CrossReferenceResult:
        """
        Execute cross-referencing for a source chapter.
        
        The agent will:
        1. Analyze the source chapter to understand key concepts
        2. Search the taxonomy for related books across tiers
        3. Traverse the graph following relationship edges
        4. Retrieve relevant chapter content for validation
        5. Synthesize a scholarly annotation with tier-aware citations
        """
        state = CrossReferenceState(
            source=source_chapter,
            taxonomy=taxonomy,
            config=config,
        )
        
        result = await self.workflow.ainvoke(state)
        
        return CrossReferenceResult(
            annotation=result.annotation,
            citations=result.citations,
            traversal_path=result.traversal_path,
            tier_coverage=result.tier_coverage,
        )
```

---

## Dependencies

| Dependency | Type | Purpose |
|------------|------|---------|
| llm-gateway | Microservice | LLM inference with tool-use |
| semantic-search-service | Microservice | Embeddings, vector search (Qdrant) |
| Neo4j | Database | Taxonomy graph storage, Cypher queries |
| LangChain | Library | Agent orchestration |
| LangGraph | Library | State machine workflows |
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
      - NEO4J_URL=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - MODEL_CACHE_PATH=/data/models
    depends_on:
      - llm-gateway
      - semantic-search
      - neo4j

  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"  # Browser
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}

volumes:
  neo4j_data:
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
    
    # Neo4j (NEW)
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    
    # Model paths
    model_cache_path: str = "/data/models"
    
    # Agent settings
    default_llm_model: str = "claude-3-sonnet-20240229"
    
    # Cross-Reference Agent settings (NEW)
    max_traversal_hops: int = 3
    default_similarity_threshold: float = 0.7
    allow_graph_cycles: bool = True
    
    class Config:
        env_prefix = "AI_AGENTS_"
```

---

## Relationship Types (Spider Web Model)

The Cross-Reference Agent understands three relationship types:

| Relationship | Definition | Direction | Use Case |
|--------------|------------|-----------|----------|
| **PARALLEL** | Same tier level | ◄────► Bidirectional | Complementary approaches |
| **PERPENDICULAR** | Adjacent tiers (±1) | ◄────► Bidirectional | Theory ↔ Implementation |
| **SKIP_TIER** | Non-adjacent tiers (±2+) | ◄────► Bidirectional | Concept ↔ Operations |

All relationships are **bidirectional**. The graph is a **spider web**, not a one-way hierarchy.

---

## See Also

- [CROSS_REFERENCE_AGENT.md](./CROSS_REFERENCE_AGENT.md) - Detailed Cross-Reference Agent documentation
- [TOOL_DEFINITIONS.md](./TOOL_DEFINITIONS.md) - Complete tool function schemas
- [/textbooks/TIER_RELATIONSHIP_DIAGRAM.md](/textbooks/TIER_RELATIONSHIP_DIAGRAM.md) - Spider web traversal model
