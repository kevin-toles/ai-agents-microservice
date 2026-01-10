# MCP Server Tool Locations - Complete Reference

**Purpose:** Quick reference for finding all MCP tools in the ai-agents codebase.

---

## Primary Entry Point

```
ai-agents/src/mcp/stdio_server.py
```

This is the **main MCP server** that exposes all 9 tools to VS Code/Claude Desktop via FastMCP.

---

## All Tool Source Locations

| Layer | Path | What's There |
|-------|------|--------------|
| **MCP Server** | `src/mcp/` | MCP protocol implementation |
| **Agent Functions** | `src/functions/` | Core 8 agent function implementations |
| **Cross-Reference Tools** | `src/agents/cross_reference/tools/` | similarity, graph, content, metadata, taxonomy |
| **General Tools** | `src/tools/` | ast_parser, textbook_search, template_engine, code_validation |

---

## Detailed Paths

### 1. MCP Layer (`src/mcp/`)

```
src/mcp/
├── stdio_server.py           # ⭐ Main MCP server (9 @mcp.tool() decorators)
├── server.py                 # Alternative server implementation
├── agent_functions_server.py # Dict-based server for testing
├── toolbox_manager.py        # External genai-toolbox integration
├── semantic_search_wrapper.py # Wraps semantic-search-service
└── README.md                 # Architecture docs
```

### 2. Agent Functions (`src/functions/`)

These are the 8 core Kitchen Brigade agent functions:

```
src/functions/
├── extract_structure.py      # ExtractStructureFunction
├── summarize_content.py      # SummarizeContentFunction
├── generate_code.py          # GenerateCodeFunction
├── analyze_artifact.py       # AnalyzeArtifactFunction
├── validate_against_spec.py  # ValidateAgainstSpecFunction
├── decompose_task.py         # DecomposeTaskFunction
├── synthesize_outputs.py     # SynthesizeOutputsFunction
└── cross_reference.py        # CrossReferenceFunction
```

### 3. Cross-Reference Tools (`src/agents/cross_reference/tools/`)

Tools specific to the `cross_reference` agent function:

```
src/agents/cross_reference/tools/
├── similarity.py    # Semantic similarity search (Qdrant vectors)
├── graph.py         # Neo4j graph traversal
├── content.py       # Content retrieval
├── metadata.py      # Metadata lookup
└── taxonomy.py      # Classification/taxonomy
```

### 4. General Tools (`src/tools/`)

Shared tools used by multiple agent functions:

```
src/tools/
├── ast_parser.py        # AST parsing for extract_structure
├── textbook_search.py   # JSON textbook search for cross_reference
├── template_engine.py   # Jinja2 templates for generate_code
└── code_validation.py   # Code validation utilities
```

---

## Quick Search Paths

If you need to find tools, search these directories:

```
ai-agents/src/mcp/stdio_server.py           # MCP server entry point
ai-agents/src/mcp/                          # All MCP infrastructure
ai-agents/src/functions/                    # 8 agent function implementations
ai-agents/src/tools/                        # General tools
ai-agents/src/agents/cross_reference/tools/ # Cross-reference specific tools
```

---

## MCP Tools Exposed (9 total)

| Tool | Description | Source |
|------|-------------|--------|
| `extract_structure` | Parse code/document structure | `src/functions/extract_structure.py` |
| `summarize_content` | Generate summaries using LLM | `src/functions/summarize_content.py` |
| `generate_code` | Generate code from specifications | `src/functions/generate_code.py` |
| `analyze_artifact` | Deep analysis of code artifacts | `src/functions/analyze_artifact.py` |
| `validate_against_spec` | Validate code against specifications | `src/functions/validate_against_spec.py` |
| `decompose_task` | Break tasks into subtasks (WBS) | `src/functions/decompose_task.py` |
| `synthesize_outputs` | Combine multiple agent outputs | `src/functions/synthesize_outputs.py` |
| `cross_reference` | Search across code and book knowledge | `src/functions/cross_reference.py` |
| `llm_complete` | Tiered LLM fallback (local→cloud→passthrough) | `src/mcp/stdio_server.py` |

---

## Related Documentation

- [MCP README](src/mcp/README.md) - Full MCP architecture docs
- [Kitchen Brigade Architecture](docs/architecture/KITCHEN_BRIGADE_ARCHITECTURE.md) - Agent function design
- [config/mcp.json.template](config/mcp.json.template) - MCP server configuration template
