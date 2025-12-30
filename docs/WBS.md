# ai-agents Work Breakdown Structure (WBS)

> **Version:** 1.0.0  
> **Created:** 2025-12-29  
> **Status:** Planning Phase  
> **Reference:** [AGENT_FUNCTIONS_ARCHITECTURE.md](../../textbooks/pending/platform/AGENT_FUNCTIONS_ARCHITECTURE.md)

## Overview

This WBS defines the implementation tasks for the **Agent Functions Architecture** in the ai-agents service. Each WBS block is **self-contained** - when implementation and exit criteria are satisfied, the acceptance criteria is automatically satisfied.

**TDD Approach:** All implementation follows RED → GREEN → REFACTOR cycle.

**Document Priority Hierarchy:**
1. GUIDELINES_AI_Engineering_Building_Applications_AIML_LLM_ENHANCED.md
2. AI_CODING_PLATFORM_ARCHITECTURE.md (Kitchen Brigade)
3. llm-gateway ARCHITECTURE.md
4. AI-ML_taxonomy_20251128.json
5. CODING_PATTERNS_ANALYSIS.md

---

## WBS Summary

| Block | Name | Dependencies | Est. Effort |
|-------|------|--------------|-------------|
| WBS-AGT1 | Repository Scaffolding | None | 2 hours |
| WBS-AGT2 | Core Infrastructure | WBS-AGT1 | 4 hours |
| WBS-AGT3 | ADK State Management | WBS-AGT2 | 4 hours |
| WBS-AGT4 | Pydantic Schemas - Core Types | WBS-AGT2 | 4 hours |
| WBS-AGT5 | Agent Function Base Class | WBS-AGT3, WBS-AGT4 | 6 hours |
| WBS-AGT6 | extract_structure Function | WBS-AGT5 | 6 hours |
| WBS-AGT7 | summarize_content Function | WBS-AGT5 | 6 hours |
| WBS-AGT8 | generate_code Function | WBS-AGT5 | 8 hours |
| WBS-AGT9 | analyze_artifact Function | WBS-AGT5 | 6 hours |
| WBS-AGT10 | validate_against_spec Function | WBS-AGT5 | 4 hours |
| WBS-AGT11 | synthesize_outputs Function | WBS-AGT5 | 4 hours |
| WBS-AGT12 | decompose_task Function | WBS-AGT5 | 4 hours |
| WBS-AGT13 | cross_reference Function | WBS-AGT5 | 6 hours |
| WBS-AGT14 | Pipeline Orchestrator | WBS-AGT6-13 | 8 hours |
| WBS-AGT15 | Chapter Summarization Pipeline | WBS-AGT14 | 6 hours |
| WBS-AGT16 | Code Generation Pipeline | WBS-AGT14 | 6 hours |
| WBS-AGT17 | Citation Flow & Audit | WBS-AGT14 | 6 hours |
| WBS-AGT18 | API Routes | WBS-AGT14 | 6 hours |
| WBS-AGT19 | Anti-Pattern Compliance | All prior WBS | 4 hours |
| WBS-AGT20 | Integration Testing | All prior WBS | 8 hours |
| **WBS-AGT21** | **Code Reference Engine Client** | WBS-AGT2 | **6 hours** |
| **WBS-AGT22** | **Neo4j Graph Integration** | WBS-AGT21 | **6 hours** |
| **WBS-AGT23** | **Book/JSON Passage Retrieval** | WBS-AGT21, WBS-AGT22 | **8 hours** |
| **WBS-AGT24** | **Unified Knowledge Retrieval** | WBS-AGT13, WBS-AGT21-23 | **8 hours** |

**Total Estimated Effort:** ~136 hours

---

## WBS-AGT1: Repository Scaffolding

**Dependencies:** None  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-1.1 | Folder structure matches ai-agents service layout |
| AC-1.2 | All `__init__.py` files created for Python packages |
| AC-1.3 | pyproject.toml configured with project metadata |
| AC-1.4 | .env.example contains all environment variables |
| AC-1.5 | conftest.py with FakeClient pattern for testing |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT1.1 | Create src/functions/ directory | AC-1.1 | `src/functions/` |
| AGT1.2 | Create src/pipelines/ directory | AC-1.1 | `src/pipelines/` |
| AGT1.3 | Create src/schemas/ directory | AC-1.1 | `src/schemas/` |
| AGT1.4 | Create src/cache/ directory | AC-1.1 | `src/cache/` |
| AGT1.5 | Create src/citations/ directory | AC-1.1 | `src/citations/` |
| AGT1.6 | Create tests/unit/functions/ directory | AC-1.1 | `tests/unit/functions/` |
| AGT1.7 | Create tests/unit/pipelines/ directory | AC-1.1 | `tests/unit/pipelines/` |
| AGT1.8 | Create all __init__.py files | AC-1.2 | All package directories |
| AGT1.9 | Update pyproject.toml | AC-1.3 | `pyproject.toml` |
| AGT1.10 | Create .env.example | AC-1.4 | `.env.example` |
| AGT1.11 | Create conftest.py with FakeClient | AC-1.5 | `tests/conftest.py` |

### Exit Criteria

- [ ] `tree src/` shows functions/, pipelines/, schemas/, cache/, citations/
- [ ] `python -c "from src.functions import base"` succeeds
- [ ] `pip install -e .` succeeds
- [ ] `.env.example` contains INFERENCE_SERVICE_URL, SEMANTIC_SEARCH_URL, etc.
- [ ] FakeClient in conftest.py implements Protocol duck typing

---

## WBS-AGT2: Core Infrastructure

**Dependencies:** WBS-AGT1  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Architecture Overview

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-2.1 | Pydantic Settings loads all service URLs |
| AC-2.2 | Structured logging with JSON format |
| AC-2.3 | FastAPI app initializes on :8082 |
| AC-2.4 | HTTP client factory for downstream services |
| AC-2.5 | Service discovery for Kitchen Brigade services |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT2.1 | RED: Write config tests | AC-2.1 | `tests/unit/core/test_config.py` |
| AGT2.2 | GREEN: Implement Settings class | AC-2.1, AC-2.5 | `src/core/config.py` |
| AGT2.3 | RED: Write logging tests | AC-2.2 | `tests/unit/core/test_logging.py` |
| AGT2.4 | GREEN: Implement structured logging | AC-2.2 | `src/core/logging.py` |
| AGT2.5 | RED: Write HTTP client tests | AC-2.4 | `tests/unit/core/test_http.py` |
| AGT2.6 | GREEN: Implement HTTPClientFactory | AC-2.4 | `src/core/http.py` |
| AGT2.7 | GREEN: Implement FastAPI app | AC-2.3 | `src/main.py` |
| AGT2.8 | REFACTOR: Add service URLs as constants | AC-2.5 | `src/core/constants.py` |

### Exit Criteria

- [ ] `pytest tests/unit/core/` passes with 100% coverage
- [ ] Settings loads: `INFERENCE_SERVICE_URL=http://localhost:8085`
- [ ] HTTPClientFactory creates clients for inference-service, semantic-search, audit-service
- [ ] Service port is 8082 (Expeditor role in Kitchen Brigade)

---

## WBS-AGT3: ADK State Management

**Dependencies:** WBS-AGT2  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → ADK Pattern Integration

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-3.1 | State prefix constants: temp:, user:, app: |
| AC-3.2 | build_cache_key() follows ADK conventions |
| AC-3.3 | Artifact dataclass with version tracking |
| AC-3.4 | HandoffCache for temp: prefix (pipeline-local) |
| AC-3.5 | CompressionCache for user: prefix (Redis 24h TTL) |
| AC-3.6 | No mutable default arguments (AP-1.5) |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT3.1 | RED: Write state prefix tests | AC-3.1, AC-3.2 | `tests/unit/cache/test_state.py` |
| AGT3.2 | GREEN: Implement STATE_PREFIX_* constants | AC-3.1 | `src/cache/state.py` |
| AGT3.3 | GREEN: Implement build_cache_key() | AC-3.2 | `src/cache/state.py` |
| AGT3.4 | RED: Write Artifact tests | AC-3.3 | `tests/unit/cache/test_artifact.py` |
| AGT3.5 | GREEN: Implement Artifact dataclass | AC-3.3, AC-3.6 | `src/cache/artifact.py` |
| AGT3.6 | RED: Write HandoffCache tests | AC-3.4 | `tests/unit/cache/test_handoff.py` |
| AGT3.7 | GREEN: Implement HandoffCache | AC-3.4, AC-3.6 | `src/cache/handoff.py` |
| AGT3.8 | RED: Write CompressionCache tests | AC-3.5 | `tests/unit/cache/test_compression.py` |
| AGT3.9 | GREEN: Implement CompressionCache | AC-3.5 | `src/cache/compression.py` |
| AGT3.10 | Verify no mutable defaults | AC-3.6 | All dataclasses |

### Exit Criteria

- [ ] `pytest tests/unit/cache/` passes with 100% coverage
- [ ] `build_cache_key("temp:", "extract_structure", "ch1")` → `"temp:extract_structure:ch1"`
- [ ] Artifact.qualified_name returns `"{namespace}/{name}_v{version}"`
- [ ] HandoffCache uses asyncio.Lock (AP-10.1)
- [ ] `grep -r "= \[\]" src/cache/` returns 0 results in dataclasses

---

## WBS-AGT4: Pydantic Schemas - Core Types

**Dependencies:** WBS-AGT2  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Pydantic Schemas, Citation Flow

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-4.1 | SourceMetadata for provenance tracking |
| AC-4.2 | Citation with Chicago-style formatting |
| AC-4.3 | CitedContent with embedded [^N] markers |
| AC-4.4 | Finding/Violation models for analysis |
| AC-4.5 | All schemas have JSON schema export |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT4.1 | RED: Write SourceMetadata tests | AC-4.1 | `tests/unit/schemas/test_citations.py` |
| AGT4.2 | GREEN: Implement SourceMetadata | AC-4.1 | `src/schemas/citations.py` |
| AGT4.3 | RED: Write Citation tests | AC-4.2 | `tests/unit/schemas/test_citations.py` |
| AGT4.4 | GREEN: Implement Citation model | AC-4.2 | `src/schemas/citations.py` |
| AGT4.5 | Implement chicago_format() method | AC-4.2 | `src/schemas/citations.py` |
| AGT4.6 | RED: Write CitedContent tests | AC-4.3 | `tests/unit/schemas/test_citations.py` |
| AGT4.7 | GREEN: Implement CitedContent | AC-4.3 | `src/schemas/citations.py` |
| AGT4.8 | RED: Write Finding/Violation tests | AC-4.4 | `tests/unit/schemas/test_analysis.py` |
| AGT4.9 | GREEN: Implement Finding, Violation | AC-4.4 | `src/schemas/analysis.py` |
| AGT4.10 | Add model_json_schema() exports | AC-4.5 | All schema files |

### Exit Criteria

- [ ] `pytest tests/unit/schemas/` passes with 100% coverage
- [ ] Citation.chicago_format() returns proper Chicago-style string
- [ ] CitedContent preserves [^N] markers and footnotes list
- [ ] `Finding.model_json_schema()` returns valid JSON Schema

---

## WBS-AGT5: Agent Function Base Class

**Dependencies:** WBS-AGT3, WBS-AGT4  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Functions, CODING_PATTERNS_ANALYSIS.md

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-5.1 | AgentFunction ABC with run() abstract method |
| AC-5.2 | ABC signature uses **kwargs for flexibility (AP-ABC) |
| AC-5.3 | Context budget enforcement per function |
| AC-5.4 | Preset selection mechanism |
| AC-5.5 | FakeAgentFunction for testing |
| AC-5.6 | Protocol duck typing for type hints |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT5.1 | RED: Write AgentFunction ABC tests | AC-5.1, AC-5.2 | `tests/unit/functions/test_base.py` |
| AGT5.2 | GREEN: Implement AgentFunction ABC | AC-5.1, AC-5.2 | `src/functions/base.py` |
| AGT5.3 | RED: Write context budget tests | AC-5.3 | `tests/unit/functions/test_base.py` |
| AGT5.4 | GREEN: Implement CONTEXT_BUDGET_DEFAULTS | AC-5.3 | `src/functions/base.py` |
| AGT5.5 | GREEN: Implement enforce_budget() method | AC-5.3 | `src/functions/base.py` |
| AGT5.6 | RED: Write preset selection tests | AC-5.4 | `tests/unit/functions/test_base.py` |
| AGT5.7 | GREEN: Implement select_preset() | AC-5.4 | `src/functions/base.py` |
| AGT5.8 | GREEN: Implement FakeAgentFunction | AC-5.5 | `tests/unit/functions/fake_agent.py` |
| AGT5.9 | Add Protocol type hints | AC-5.6 | `src/functions/base.py` |
| AGT5.10 | REFACTOR: Verify ABC pattern | AC-5.2 | `src/functions/base.py` |

### Exit Criteria

- [ ] `pytest tests/unit/functions/test_base.py` passes with 100% coverage
- [ ] ABC prevents instantiation without implementing run()
- [ ] CONTEXT_BUDGET_DEFAULTS matches ARCHITECTURE.md values
- [ ] FakeAgentFunction passes all interface tests
- [ ] `mypy src/functions/base.py` reports 0 errors

---

## Implementation Order (Phase 1)

```
WBS-AGT1 (Scaffolding)
    │
    ▼
WBS-AGT2 (Core Infrastructure)
    │
    ├──────────────┐
    ▼              ▼
WBS-AGT3       WBS-AGT4
(ADK State)    (Schemas)
    │              │
    └──────┬───────┘
           ▼
    WBS-AGT5 (Base Class)
           │
    ┌──────┼──────┬──────┬──────┬──────┬──────┬──────┐
    ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
AGT6    AGT7   AGT8   AGT9  AGT10  AGT11  AGT12  AGT13
(extract)(sum) (gen) (analyze)(val)(synth)(decomp)(xref)
```

---

## WBS-AGT6: extract_structure Function

**Dependencies:** WBS-AGT5  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 1

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-6.1 | Parses JSON/Markdown/Code into hierarchical structure |
| AC-6.2 | Returns StructuredOutput with headings, sections, code_blocks |
| AC-6.3 | Context budget: 16384 input / 2048 output |
| AC-6.4 | Default preset: S1 (Light) |
| AC-6.5 | Supports artifact_type parameter |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT6.1 | RED: Write input schema tests | AC-6.1 | `tests/unit/functions/test_extract_structure.py` |
| AGT6.2 | GREEN: Implement ExtractStructureInput | AC-6.1 | `src/schemas/functions/extract_structure.py` |
| AGT6.3 | RED: Write output schema tests | AC-6.2 | `tests/unit/functions/test_extract_structure.py` |
| AGT6.4 | GREEN: Implement StructuredOutput | AC-6.2 | `src/schemas/functions/extract_structure.py` |
| AGT6.5 | RED: Write function tests | AC-6.1-6.5 | `tests/unit/functions/test_extract_structure.py` |
| AGT6.6 | GREEN: Implement ExtractStructureFunction | AC-6.1-6.5 | `src/functions/extract_structure.py` |
| AGT6.7 | Implement run() method | AC-6.1 | `src/functions/extract_structure.py` |
| AGT6.8 | REFACTOR: Add artifact_type dispatch | AC-6.5 | `src/functions/extract_structure.py` |

### Exit Criteria

- [ ] `pytest tests/unit/functions/test_extract_structure.py` passes with 100% coverage
- [ ] JSON input returns nested structure with depth levels
- [ ] Markdown input identifies headings H1-H6
- [ ] Context budget enforced (16384/2048)

---

## WBS-AGT7: summarize_content Function

**Dependencies:** WBS-AGT5  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 2

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-7.1 | Generates summaries with citation markers [^N] |
| AC-7.2 | Returns CitedContent with footnotes list |
| AC-7.3 | Context budget: 8192 input / 4096 output |
| AC-7.4 | Default preset: D4 (Standard) |
| AC-7.5 | Supports detail_level parameter (brief/standard/comprehensive) |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT7.1 | RED: Write input schema tests | AC-7.1, AC-7.5 | `tests/unit/functions/test_summarize_content.py` |
| AGT7.2 | GREEN: Implement SummarizeContentInput | AC-7.1, AC-7.5 | `src/schemas/functions/summarize_content.py` |
| AGT7.3 | RED: Write citation generation tests | AC-7.1, AC-7.2 | `tests/unit/functions/test_summarize_content.py` |
| AGT7.4 | GREEN: Implement citation marker injection | AC-7.1, AC-7.2 | `src/functions/summarize_content.py` |
| AGT7.5 | RED: Write detail_level tests | AC-7.5 | `tests/unit/functions/test_summarize_content.py` |
| AGT7.6 | GREEN: Implement SummarizeContentFunction | AC-7.1-7.5 | `src/functions/summarize_content.py` |
| AGT7.7 | Implement run() with inference-service call | AC-7.3, AC-7.4 | `src/functions/summarize_content.py` |

### Exit Criteria

- [ ] `pytest tests/unit/functions/test_summarize_content.py` passes with 100% coverage
- [ ] Output contains [^1], [^2] markers linked to footnotes
- [ ] detail_level="brief" produces <500 token output
- [ ] Chicago-style citations in footnotes list

---

## WBS-AGT8: generate_code Function

**Dependencies:** WBS-AGT5  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 3

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-8.1 | Generates code from natural language spec |
| AC-8.2 | Returns CodeOutput with language, code, explanation |
| AC-8.3 | Context budget: 4096 input / 8192 output |
| AC-8.4 | Default preset: D4 (Standard) |
| AC-8.5 | Supports target_language parameter |
| AC-8.6 | Includes test stubs when include_tests=True |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT8.1 | RED: Write input schema tests | AC-8.1, AC-8.5 | `tests/unit/functions/test_generate_code.py` |
| AGT8.2 | GREEN: Implement GenerateCodeInput | AC-8.1, AC-8.5 | `src/schemas/functions/generate_code.py` |
| AGT8.3 | RED: Write output schema tests | AC-8.2 | `tests/unit/functions/test_generate_code.py` |
| AGT8.4 | GREEN: Implement CodeOutput | AC-8.2 | `src/schemas/functions/generate_code.py` |
| AGT8.5 | RED: Write code generation tests | AC-8.1, AC-8.3 | `tests/unit/functions/test_generate_code.py` |
| AGT8.6 | GREEN: Implement GenerateCodeFunction | AC-8.1-8.4 | `src/functions/generate_code.py` |
| AGT8.7 | RED: Write test stub generation tests | AC-8.6 | `tests/unit/functions/test_generate_code.py` |
| AGT8.8 | GREEN: Implement include_tests logic | AC-8.6 | `src/functions/generate_code.py` |
| AGT8.9 | Implement language-specific prompts | AC-8.5 | `src/functions/generate_code.py` |

### Exit Criteria

- [ ] `pytest tests/unit/functions/test_generate_code.py` passes with 100% coverage
- [ ] Python spec generates valid Python code
- [ ] include_tests=True adds pytest test stubs
- [ ] Output code passes basic syntax validation

---

## WBS-AGT9: analyze_artifact Function

**Dependencies:** WBS-AGT5  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 4

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-9.1 | Analyzes code/docs for quality, patterns, issues |
| AC-9.2 | Returns AnalysisResult with findings list |
| AC-9.3 | Context budget: 16384 input / 2048 output |
| AC-9.4 | Default preset: D4 (Standard) |
| AC-9.5 | Supports analysis_type parameter (quality/security/patterns) |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT9.1 | RED: Write input schema tests | AC-9.1, AC-9.5 | `tests/unit/functions/test_analyze_artifact.py` |
| AGT9.2 | GREEN: Implement AnalyzeArtifactInput | AC-9.1, AC-9.5 | `src/schemas/functions/analyze_artifact.py` |
| AGT9.3 | RED: Write findings schema tests | AC-9.2 | `tests/unit/functions/test_analyze_artifact.py` |
| AGT9.4 | GREEN: Implement AnalysisResult, Finding | AC-9.2 | `src/schemas/functions/analyze_artifact.py` |
| AGT9.5 | RED: Write analysis tests | AC-9.1-9.5 | `tests/unit/functions/test_analyze_artifact.py` |
| AGT9.6 | GREEN: Implement AnalyzeArtifactFunction | AC-9.1-9.5 | `src/functions/analyze_artifact.py` |
| AGT9.7 | Implement quality analysis prompts | AC-9.5 | `src/functions/analyze_artifact.py` |
| AGT9.8 | Implement security analysis prompts | AC-9.5 | `src/functions/analyze_artifact.py` |

### Exit Criteria

- [ ] `pytest tests/unit/functions/test_analyze_artifact.py` passes with 100% coverage
- [ ] Each Finding has severity, category, description, location
- [ ] analysis_type="security" flags common vulnerabilities
- [ ] analysis_type="patterns" identifies design patterns

---

## WBS-AGT10: validate_against_spec Function

**Dependencies:** WBS-AGT5  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 5

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-10.1 | Compares artifact against specification |
| AC-10.2 | Returns ValidationResult with compliance %, violations |
| AC-10.3 | Context budget: 4096 input / 1024 output |
| AC-10.4 | Default preset: D4 (Standard) |
| AC-10.5 | Violations include line_number, expected, actual |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT10.1 | RED: Write input schema tests | AC-10.1 | `tests/unit/functions/test_validate_against_spec.py` |
| AGT10.2 | GREEN: Implement ValidateAgainstSpecInput | AC-10.1 | `src/schemas/functions/validate_against_spec.py` |
| AGT10.3 | RED: Write validation result tests | AC-10.2, AC-10.5 | `tests/unit/functions/test_validate_against_spec.py` |
| AGT10.4 | GREEN: Implement ValidationResult, Violation | AC-10.2, AC-10.5 | `src/schemas/functions/validate_against_spec.py` |
| AGT10.5 | RED: Write validation function tests | AC-10.1-10.4 | `tests/unit/functions/test_validate_against_spec.py` |
| AGT10.6 | GREEN: Implement ValidateAgainstSpecFunction | AC-10.1-10.4 | `src/functions/validate_against_spec.py` |
| AGT10.7 | Implement compliance percentage calc | AC-10.2 | `src/functions/validate_against_spec.py` |

### Exit Criteria

- [ ] `pytest tests/unit/functions/test_validate_against_spec.py` passes with 100% coverage
- [ ] compliance_percentage is 0-100 float
- [ ] Each Violation has expected vs actual comparison
- [ ] Empty violations list → compliance_percentage = 100.0

---

## WBS-AGT11: synthesize_outputs Function

**Dependencies:** WBS-AGT5  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 6

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-11.1 | Combines multiple outputs into coherent whole |
| AC-11.2 | Returns SynthesizedOutput with merged_content, source_map |
| AC-11.3 | Context budget: 8192 input / 4096 output |
| AC-11.4 | Default preset: S1 (Light) |
| AC-11.5 | Preserves citations from input sources |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT11.1 | RED: Write input schema tests | AC-11.1 | `tests/unit/functions/test_synthesize_outputs.py` |
| AGT11.2 | GREEN: Implement SynthesizeOutputsInput | AC-11.1 | `src/schemas/functions/synthesize_outputs.py` |
| AGT11.3 | RED: Write output schema tests | AC-11.2, AC-11.5 | `tests/unit/functions/test_synthesize_outputs.py` |
| AGT11.4 | GREEN: Implement SynthesizedOutput | AC-11.2 | `src/schemas/functions/synthesize_outputs.py` |
| AGT11.5 | RED: Write synthesis tests | AC-11.1-11.5 | `tests/unit/functions/test_synthesize_outputs.py` |
| AGT11.6 | GREEN: Implement SynthesizeOutputsFunction | AC-11.1-11.4 | `src/functions/synthesize_outputs.py` |
| AGT11.7 | Implement citation merging | AC-11.5 | `src/functions/synthesize_outputs.py` |
| AGT11.8 | Implement source_map tracking | AC-11.2 | `src/functions/synthesize_outputs.py` |

### Exit Criteria

- [ ] `pytest tests/unit/functions/test_synthesize_outputs.py` passes with 100% coverage
- [ ] source_map traces each section to original input
- [ ] Citations renumbered correctly after merge
- [ ] No duplicate content in merged output

---

## WBS-AGT12: decompose_task Function

**Dependencies:** WBS-AGT5  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 7

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-12.1 | Breaks complex task into subtasks |
| AC-12.2 | Returns TaskDecomposition with subtasks, dependencies |
| AC-12.3 | Context budget: 4096 input / 2048 output |
| AC-12.4 | Default preset: S2 |
| AC-12.5 | Subtasks form valid DAG (no cycles) |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT12.1 | RED: Write input schema tests | AC-12.1 | `tests/unit/functions/test_decompose_task.py` |
| AGT12.2 | GREEN: Implement DecomposeTaskInput | AC-12.1 | `src/schemas/functions/decompose_task.py` |
| AGT12.3 | RED: Write output schema tests | AC-12.2, AC-12.5 | `tests/unit/functions/test_decompose_task.py` |
| AGT12.4 | GREEN: Implement TaskDecomposition, Subtask | AC-12.2 | `src/schemas/functions/decompose_task.py` |
| AGT12.5 | RED: Write DAG validation tests | AC-12.5 | `tests/unit/functions/test_decompose_task.py` |
| AGT12.6 | GREEN: Implement cycle detection | AC-12.5 | `src/functions/decompose_task.py` |
| AGT12.7 | GREEN: Implement DecomposeTaskFunction | AC-12.1-12.4 | `src/functions/decompose_task.py` |

### Exit Criteria

- [ ] `pytest tests/unit/functions/test_decompose_task.py` passes with 100% coverage
- [ ] Each Subtask has id, description, depends_on list
- [ ] Cyclic dependencies raise ValidationError
- [ ] Topological sort produces valid execution order

---

## WBS-AGT13: cross_reference Function

**Dependencies:** WBS-AGT5  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 8

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-13.1 | Queries semantic-search-service for related content |
| AC-13.2 | Returns CrossReferenceResult with matches, relevance scores |
| AC-13.3 | Context budget: 2048 input / 4096 output |
| AC-13.4 | Default preset: S4 |
| AC-13.5 | Integrates with Qdrant via semantic-search-service |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT13.1 | RED: Write input schema tests | AC-13.1 | `tests/unit/functions/test_cross_reference.py` |
| AGT13.2 | GREEN: Implement CrossReferenceInput | AC-13.1 | `src/schemas/functions/cross_reference.py` |
| AGT13.3 | RED: Write output schema tests | AC-13.2 | `tests/unit/functions/test_cross_reference.py` |
| AGT13.4 | GREEN: Implement CrossReferenceResult, Match | AC-13.2 | `src/schemas/functions/cross_reference.py` |
| AGT13.5 | RED: Write semantic search integration tests | AC-13.5 | `tests/unit/functions/test_cross_reference.py` |
| AGT13.6 | GREEN: Implement SemanticSearchClient | AC-13.5 | `src/clients/semantic_search.py` |
| AGT13.7 | GREEN: Implement CrossReferenceFunction | AC-13.1-13.4 | `src/functions/cross_reference.py` |
| AGT13.8 | Implement relevance score normalization | AC-13.2 | `src/functions/cross_reference.py` |

### Exit Criteria

- [ ] `pytest tests/unit/functions/test_cross_reference.py` passes with 100% coverage
- [ ] Each Match has source, content, relevance_score (0.0-1.0)
- [ ] FakeSemanticSearchClient used in unit tests
- [ ] Integration test hits real semantic-search-service:8081

---

## WBS-AGT14: Pipeline Orchestrator

**Dependencies:** WBS-AGT6-13  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Pipeline DAGs

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-14.1 | PipelineOrchestrator executes function DAGs |
| AC-14.2 | Supports SequentialAgent, ParallelAgent, LoopAgent patterns |
| AC-14.3 | HandoffState flows between pipeline stages |
| AC-14.4 | Pipeline stages can be conditional |
| AC-14.5 | Saga compensation on stage failure |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT14.1 | RED: Write orchestrator tests | AC-14.1 | `tests/unit/pipelines/test_orchestrator.py` |
| AGT14.2 | GREEN: Implement PipelineOrchestrator | AC-14.1 | `src/pipelines/orchestrator.py` |
| AGT14.3 | RED: Write agent pattern tests | AC-14.2 | `tests/unit/pipelines/test_agents.py` |
| AGT14.4 | GREEN: Implement SequentialAgent | AC-14.2 | `src/pipelines/agents.py` |
| AGT14.5 | GREEN: Implement ParallelAgent | AC-14.2 | `src/pipelines/agents.py` |
| AGT14.6 | GREEN: Implement LoopAgent | AC-14.2 | `src/pipelines/agents.py` |
| AGT14.7 | RED: Write handoff tests | AC-14.3 | `tests/unit/pipelines/test_orchestrator.py` |
| AGT14.8 | GREEN: Implement stage handoff | AC-14.3 | `src/pipelines/orchestrator.py` |
| AGT14.9 | RED: Write saga tests | AC-14.5 | `tests/unit/pipelines/test_saga.py` |
| AGT14.10 | GREEN: Implement PipelineSaga | AC-14.5 | `src/pipelines/saga.py` |

### Exit Criteria

- [ ] `pytest tests/unit/pipelines/` passes with 100% coverage
- [ ] SequentialAgent executes stages in order
- [ ] ParallelAgent uses asyncio.gather
- [ ] Failed stage triggers compensation for completed stages

---

## WBS-AGT15: Chapter Summarization Pipeline

**Dependencies:** WBS-AGT14  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Pipeline DAG: chapter-summarization

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-15.1 | 4-stage pipeline: extract → cross_ref → summarize → validate |
| AC-15.2 | Produces CitedContent output with footnotes |
| AC-15.3 | Configurable via preset (Light/Standard/High Quality) |
| AC-15.4 | Registers as `/v1/pipelines/chapter-summarization/run` |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT15.1 | RED: Write pipeline definition tests | AC-15.1 | `tests/unit/pipelines/test_chapter_summarization.py` |
| AGT15.2 | GREEN: Define ChapterSummarizationPipeline | AC-15.1 | `src/pipelines/chapter_summarization.py` |
| AGT15.3 | Implement stage 1: extract_structure | AC-15.1 | `src/pipelines/chapter_summarization.py` |
| AGT15.4 | Implement stage 2: cross_reference | AC-15.1 | `src/pipelines/chapter_summarization.py` |
| AGT15.5 | Implement stage 3: summarize_content | AC-15.1 | `src/pipelines/chapter_summarization.py` |
| AGT15.6 | Implement stage 4: validate_against_spec | AC-15.1 | `src/pipelines/chapter_summarization.py` |
| AGT15.7 | RED: Write citation output tests | AC-15.2 | `tests/unit/pipelines/test_chapter_summarization.py` |
| AGT15.8 | GREEN: Implement citation aggregation | AC-15.2 | `src/pipelines/chapter_summarization.py` |
| AGT15.9 | Implement preset configuration | AC-15.3 | `src/pipelines/chapter_summarization.py` |

### Exit Criteria

- [ ] `pytest tests/unit/pipelines/test_chapter_summarization.py` passes
- [ ] Pipeline produces summary with [^N] citation markers
- [ ] Footnotes contain Chicago-style source references
- [ ] preset="high_quality" uses D10, preset="light" uses S1

---

## WBS-AGT16: Code Generation Pipeline

**Dependencies:** WBS-AGT14  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Pipeline DAG: code-generation

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-16.1 | 6-stage pipeline: decompose → cross_ref → generate → synthesize → analyze → validate |
| AC-16.2 | Produces CodeOutput with tests if requested |
| AC-16.3 | Parallel generation for independent subtasks |
| AC-16.4 | Registers as `/v1/pipelines/code-generation/run` |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT16.1 | RED: Write pipeline definition tests | AC-16.1 | `tests/unit/pipelines/test_code_generation.py` |
| AGT16.2 | GREEN: Define CodeGenerationPipeline | AC-16.1 | `src/pipelines/code_generation.py` |
| AGT16.3 | Implement stage 1: decompose_task | AC-16.1 | `src/pipelines/code_generation.py` |
| AGT16.4 | Implement stage 2: cross_reference | AC-16.1 | `src/pipelines/code_generation.py` |
| AGT16.5 | Implement stage 3: generate_code (parallel) | AC-16.1, AC-16.3 | `src/pipelines/code_generation.py` |
| AGT16.6 | Implement stage 4: synthesize_outputs | AC-16.1 | `src/pipelines/code_generation.py` |
| AGT16.7 | Implement stage 5: analyze_artifact | AC-16.1 | `src/pipelines/code_generation.py` |
| AGT16.8 | Implement stage 6: validate_against_spec | AC-16.1 | `src/pipelines/code_generation.py` |
| AGT16.9 | RED: Write parallel generation tests | AC-16.3 | `tests/unit/pipelines/test_code_generation.py` |
| AGT16.10 | GREEN: Implement ParallelAgent for stage 3 | AC-16.3 | `src/pipelines/code_generation.py` |

### Exit Criteria

- [ ] `pytest tests/unit/pipelines/test_code_generation.py` passes
- [ ] Subtasks from decompose_task generate code in parallel
- [ ] synthesize_outputs merges code fragments correctly
- [ ] Final output passes analyze_artifact quality check

---

## WBS-AGT17: Citation Flow & Audit

**Dependencies:** WBS-AGT14  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Citation Flow

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-17.1 | CitationManager tracks sources through pipeline |
| AC-17.2 | Chicago-style footnote formatting |
| AC-17.3 | Audit record sent to audit-service:8084 |
| AC-17.4 | Citation audit includes source_id, retrieval_score, usage_context |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT17.1 | RED: Write CitationManager tests | AC-17.1 | `tests/unit/citations/test_manager.py` |
| AGT17.2 | GREEN: Implement CitationManager | AC-17.1 | `src/citations/manager.py` |
| AGT17.3 | RED: Write formatting tests | AC-17.2 | `tests/unit/citations/test_formatter.py` |
| AGT17.4 | GREEN: Implement ChicagoFormatter | AC-17.2 | `src/citations/formatter.py` |
| AGT17.5 | RED: Write audit client tests | AC-17.3, AC-17.4 | `tests/unit/citations/test_audit.py` |
| AGT17.6 | GREEN: Implement AuditServiceClient | AC-17.3 | `src/clients/audit_service.py` |
| AGT17.7 | GREEN: Implement CitationAuditRecord | AC-17.4 | `src/schemas/audit.py` |
| AGT17.8 | Integrate audit into pipeline completion | AC-17.3 | `src/pipelines/orchestrator.py` |

### Exit Criteria

- [ ] `pytest tests/unit/citations/` passes with 100% coverage
- [ ] CitationManager assigns unique [^N] markers
- [ ] Chicago format: "Author, Title (Publisher, Year), Page."
- [ ] Audit POST to audit-service includes all citation metadata

---

## WBS-AGT18: API Routes

**Dependencies:** WBS-AGT14, WBS-AGT15, WBS-AGT16, WBS-AGT17  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-18.1 | POST /v1/functions/{name}/run executes single function |
| AC-18.2 | POST /v1/pipelines/{name}/run executes pipeline |
| AC-18.3 | GET /health returns service status |
| AC-18.4 | Request validation with Pydantic |
| AC-18.5 | Error responses match llm-gateway schema |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT18.1 | RED: Write function route tests | AC-18.1 | `tests/unit/api/test_functions.py` |
| AGT18.2 | GREEN: Implement /v1/functions/{name}/run | AC-18.1, AC-18.4 | `src/api/routes/functions.py` |
| AGT18.3 | RED: Write pipeline route tests | AC-18.2 | `tests/unit/api/test_pipelines.py` |
| AGT18.4 | GREEN: Implement /v1/pipelines/{name}/run | AC-18.2, AC-18.4 | `src/api/routes/pipelines.py` |
| AGT18.5 | RED: Write health route tests | AC-18.3 | `tests/unit/api/test_health.py` |
| AGT18.6 | GREEN: Implement /health endpoint | AC-18.3 | `src/api/routes/health.py` |
| AGT18.7 | RED: Write error handler tests | AC-18.5 | `tests/unit/api/test_errors.py` |
| AGT18.8 | GREEN: Implement error handlers | AC-18.5 | `src/api/error_handlers.py` |
| AGT18.9 | Register all routes in main.py | AC-18.1-18.5 | `src/main.py` |

### Exit Criteria

- [ ] `pytest tests/unit/api/` passes with 100% coverage
- [ ] POST /v1/functions/extract_structure/run returns StructuredOutput
- [ ] POST /v1/pipelines/chapter-summarization/run returns CitedContent
- [ ] Invalid function name returns 404 with error schema

---

## WBS-AGT19: Anti-Pattern Compliance

**Dependencies:** All prior WBS blocks  
**Reference:** CODING_PATTERNS_ANALYSIS.md

### Acceptance Criteria

| ID | Rule | Requirement |
|----|------|-------------|
| AC-19.1 | S1192 | No duplicated string literals (extract to constants) |
| AC-19.2 | S3776 | All functions cognitive complexity < 15 |
| AC-19.3 | S3516 | Functions return consistent types |
| AC-19.4 | S1172 | No unused parameters |
| AC-19.5 | AP-1.5 | No mutable default arguments in dataclasses |
| AC-19.6 | AP-ABC | ABC signatures use **kwargs for flexibility |
| AC-19.7 | Type Annotations | mypy --strict passes with 0 errors |

### WBS Tasks

| ID | Task | AC | File(s) | Tool |
|----|------|-----|---------|------|
| AGT19.1 | Audit for duplicated string literals | AC-19.1 | All src/*.py | SonarLint |
| AGT19.2 | Extract duplicates to constants | AC-19.1 | `src/core/constants.py` | Manual |
| AGT19.3 | Audit function complexity | AC-19.2 | All src/*.py | SonarLint |
| AGT19.4 | Refactor functions with CC >= 15 | AC-19.2 | As needed | Manual |
| AGT19.5 | Audit return type consistency | AC-19.3 | All src/*.py | SonarLint |
| AGT19.6 | Remove unused parameters | AC-19.4 | As needed | SonarLint |
| AGT19.7 | Verify no mutable defaults | AC-19.5 | All dataclasses | grep/AST |
| AGT19.8 | Verify ABC signatures | AC-19.6 | `src/functions/base.py` | Code review |
| AGT19.9 | Run mypy --strict | AC-19.7 | `src/` | mypy |
| AGT19.10 | Fix all mypy errors | AC-19.7 | As needed | Manual |
| AGT19.11 | Run ruff check | AC-19.7 | `src/` | ruff |

### Exit Criteria

- [ ] SonarLint: 0 S1192 issues
- [ ] SonarLint: 0 S3776 issues (max CC < 15)
- [ ] SonarLint: 0 S3516 issues
- [ ] SonarLint: 0 S1172 issues
- [ ] `grep -r "= \[\]" src/` returns 0 results in dataclasses
- [ ] AgentFunction.run() uses **kwargs pattern
- [ ] `mypy --strict src/` reports 0 errors
- [ ] `ruff check src/` reports 0 errors

---

## WBS-AGT20: Integration Testing

**Dependencies:** All prior WBS blocks  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md (all sections)

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-20.1 | E2E test: function request → response |
| AC-20.2 | E2E test: pipeline request → response with citations |
| AC-20.3 | Service integration: ai-agents → inference-service |
| AC-20.4 | Service integration: ai-agents → semantic-search-service |
| AC-20.5 | Service integration: ai-agents → audit-service |
| AC-20.6 | Load test: 5 concurrent pipeline requests |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT20.1 | Write function e2e test | AC-20.1 | `tests/integration/test_functions.py` |
| AGT20.2 | Write pipeline e2e test | AC-20.2 | `tests/integration/test_pipelines.py` |
| AGT20.3 | Write inference-service integration test | AC-20.3 | `tests/integration/test_inference.py` |
| AGT20.4 | Write semantic-search integration test | AC-20.4 | `tests/integration/test_semantic_search.py` |
| AGT20.5 | Write audit-service integration test | AC-20.5 | `tests/integration/test_audit.py` |
| AGT20.6 | Write load test | AC-20.6 | `tests/integration/test_load.py` |
| AGT20.7 | Create integration test fixtures | AC-20.1-6 | `tests/integration/conftest.py` |
| AGT20.8 | Create docker-compose.test.yml | AC-20.3-5 | `docker/docker-compose.test.yml` |
| AGT20.9 | Document integration test setup | AC-20.1-6 | `tests/integration/README.md` |

### Exit Criteria

- [ ] `pytest tests/integration/ -m "not slow"` passes
- [ ] Chapter summarization pipeline returns CitedContent with footnotes
- [ ] inference-service:8085 receives completion requests
- [ ] semantic-search-service:8081 receives query requests
- [ ] audit-service:8084 receives citation audit records
- [ ] 5 concurrent pipelines complete within 60s timeout

---

## Implementation Order (Complete)

```
WBS-AGT1 (Scaffolding)
    │
    ▼
WBS-AGT2 (Core Infrastructure)
    │
    ├──────────────┬──────────────────────────────────┐
    ▼              ▼                                  ▼
WBS-AGT3       WBS-AGT4                         WBS-AGT21
(ADK State)    (Schemas)                        (Code Ref Engine)
    │              │                                  │
    └──────┬───────┘                                  ▼
           ▼                                    WBS-AGT22
    WBS-AGT5 (Base Class)                       (Neo4j Graph)
           │                                          │
    ┌──────┼──────┬──────┬──────┬──────┬──────┬──────┤
    ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
AGT6    AGT7   AGT8   AGT9  AGT10  AGT11  AGT12  AGT13
(extract)(sum) (gen) (analyze)(val)(synth)(decomp)(xref)
    │      │      │      │      │      │      │      │
    └──────┴──────┴──────┴──────┴──────┴──────┴──────┘
                          │                          │
                          │                    WBS-AGT23
                          │                    (Book/JSON)
                          │                          │
                          ├──────────────────────────┘
                          ▼
                   WBS-AGT24 (Unified Knowledge Retrieval)
                          │
                          ▼
                   WBS-AGT14 (Pipeline Orchestrator)
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         WBS-AGT15   WBS-AGT16   WBS-AGT17
         (Chapter)   (Code Gen)  (Citations)
              │           │           │
              └───────────┴───────────┘
                          │
                          ▼
                   WBS-AGT18 (API Routes)
                          │
                          ▼
                   WBS-AGT19 (Anti-Pattern)
                          │
                          ▼
                   WBS-AGT20 (Integration)
```

---

## WBS-AGT21: Code Reference Engine Client

**Dependencies:** WBS-AGT2  
**Reference:** ai-platform-data/src/code_reference/engine.py

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-21.1 | Client wraps CodeReferenceEngine from ai-platform-data |
| AC-21.2 | Async interface for search, get_metadata, fetch_file |
| AC-21.3 | Integration with Qdrant for semantic code search |
| AC-21.4 | Integration with GitHub API for on-demand file retrieval |
| AC-21.5 | Returns CodeContext with citations for downstream |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT21.1 | Create CodeReferenceClient protocol | AC-21.2 | `src/clients/protocols.py` |
| AGT21.2 | Implement CodeReferenceClient wrapper | AC-21.1 | `src/clients/code_reference.py` |
| AGT21.3 | Add search_by_concept method | AC-21.3 | `src/clients/code_reference.py` |
| AGT21.4 | Add search_by_pattern method | AC-21.3 | `src/clients/code_reference.py` |
| AGT21.5 | Add fetch_file_content method | AC-21.4 | `src/clients/code_reference.py` |
| AGT21.6 | Create CodeContext → Citation mapper | AC-21.5 | `src/citations/code_citation.py` |
| AGT21.7 | Write unit tests with FakeCodeRefClient | AC-21.1-5 | `tests/unit/clients/test_code_reference.py` |
| AGT21.8 | Add code-reference config to .env.example | AC-21.1 | `.env.example` |

### Exit Criteria

- [ ] `from src.clients import CodeReferenceClient` succeeds
- [ ] `client.search("repository pattern")` returns CodeContext
- [ ] `client.fetch_file("backend/ddd/repository.py")` returns content
- [ ] Citations include file path, line range, and GitHub URL
- [ ] `pytest tests/unit/clients/test_code_reference.py` passes

---

## WBS-AGT22: Neo4j Graph Integration

**Dependencies:** WBS-AGT21  
**Reference:** ai-platform-data Neo4j schema, AGENT_FUNCTIONS_ARCHITECTURE.md

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-22.1 | Client connects to Neo4j for graph traversal |
| AC-22.2 | Query book → chapter → concept relationships |
| AC-22.3 | Query concept → code-reference-engine file mappings |
| AC-22.4 | Query cross-repo pattern relationships |
| AC-22.5 | Results include metadata for citation generation |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT22.1 | Create Neo4jClient protocol | AC-22.1 | `src/clients/protocols.py` |
| AGT22.2 | Implement Neo4jClient wrapper | AC-22.1 | `src/clients/neo4j_client.py` |
| AGT22.3 | Add get_concepts_for_chapter Cypher query | AC-22.2 | `src/clients/neo4j_client.py` |
| AGT22.4 | Add get_code_for_concept Cypher query | AC-22.3 | `src/clients/neo4j_client.py` |
| AGT22.5 | Add get_related_patterns Cypher query | AC-22.4 | `src/clients/neo4j_client.py` |
| AGT22.6 | Create GraphReference model | AC-22.5 | `src/schemas/graph_models.py` |
| AGT22.7 | Create GraphReference → Citation mapper | AC-22.5 | `src/citations/graph_citation.py` |
| AGT22.8 | Write unit tests with FakeNeo4jClient | AC-22.1-5 | `tests/unit/clients/test_neo4j.py` |

### Exit Criteria

- [ ] `from src.clients import Neo4jClient` succeeds
- [ ] `client.get_concepts_for_chapter("ch_001")` returns list[Concept]
- [ ] `client.get_code_for_concept("repository-pattern")` returns list[CodeReference]
- [ ] `client.get_related_patterns("saga")` returns cross-repo results
- [ ] `pytest tests/unit/clients/test_neo4j.py` passes

---

## WBS-AGT23: Book/JSON Passage Retrieval

**Dependencies:** WBS-AGT21, WBS-AGT22  
**Reference:** ai-platform-data/books/, semantic-search-service

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-23.1 | Retrieve passages from enriched book JSON files |
| AC-23.2 | Query passages via Qdrant vector similarity |
| AC-23.3 | Cross-reference passages with Neo4j concept nodes |
| AC-23.4 | Return structured BookPassage with citation metadata |
| AC-23.5 | Support filtering by book, chapter, concept |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT23.1 | Create BookPassageClient protocol | AC-23.1 | `src/clients/protocols.py` |
| AGT23.2 | Implement BookPassageClient | AC-23.1 | `src/clients/book_passage.py` |
| AGT23.3 | Add search_passages method (Qdrant) | AC-23.2 | `src/clients/book_passage.py` |
| AGT23.4 | Add get_passage_by_id method (JSON lookup) | AC-23.1 | `src/clients/book_passage.py` |
| AGT23.5 | Add get_passages_for_concept (Neo4j join) | AC-23.3 | `src/clients/book_passage.py` |
| AGT23.6 | Create BookPassage schema | AC-23.4 | `src/schemas/passage_models.py` |
| AGT23.7 | Create BookPassage → Citation mapper | AC-23.4 | `src/citations/book_citation.py` |
| AGT23.8 | Add filter_by_book, filter_by_chapter | AC-23.5 | `src/clients/book_passage.py` |
| AGT23.9 | Write unit tests with FakeBookPassageClient | AC-23.1-5 | `tests/unit/clients/test_book_passage.py` |

### Exit Criteria

- [ ] `from src.clients import BookPassageClient` succeeds
- [ ] `client.search_passages("repository pattern")` returns list[BookPassage]
- [ ] `client.get_passages_for_concept("ddd")` returns passages linked in Neo4j
- [ ] Citations include author, title, page numbers (Chicago format ready)
- [ ] `pytest tests/unit/clients/test_book_passage.py` passes

---

## WBS-AGT24: Unified Knowledge Retrieval

**Dependencies:** WBS-AGT13, WBS-AGT21, WBS-AGT22, WBS-AGT23  
**Reference:** AGENT_FUNCTIONS_ARCHITECTURE.md → cross_reference function

### Acceptance Criteria

| ID | Requirement |
|----|-------------|
| AC-24.1 | Single interface queries all knowledge sources |
| AC-24.2 | Orchestrates: Qdrant → Neo4j → code-reference-engine → books |
| AC-24.3 | Merges and ranks results across sources |
| AC-24.4 | Returns unified RetrievalResult with mixed citations |
| AC-24.5 | cross_reference agent function uses this retriever |
| AC-24.6 | Supports scope filtering (code-only, books-only, all) |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| AGT24.1 | Create UnifiedRetriever class | AC-24.1 | `src/retrieval/unified_retriever.py` |
| AGT24.2 | Implement multi-source query orchestration | AC-24.2 | `src/retrieval/unified_retriever.py` |
| AGT24.3 | Implement result merging strategy | AC-24.3 | `src/retrieval/merger.py` |
| AGT24.4 | Implement cross-source ranking | AC-24.3 | `src/retrieval/ranker.py` |
| AGT24.5 | Create RetrievalResult schema | AC-24.4 | `src/schemas/retrieval_models.py` |
| AGT24.6 | Create MixedCitation model | AC-24.4 | `src/citations/mixed_citation.py` |
| AGT24.7 | Update cross_reference function to use UnifiedRetriever | AC-24.5 | `src/functions/cross_reference.py` |
| AGT24.8 | Add scope filter parameter | AC-24.6 | `src/retrieval/unified_retriever.py` |
| AGT24.9 | Write unit tests for UnifiedRetriever | AC-24.1-6 | `tests/unit/retrieval/test_unified.py` |
| AGT24.10 | Write integration test with all sources | AC-24.2 | `tests/integration/test_unified_retrieval.py` |

### Exit Criteria

- [ ] `from src.retrieval import UnifiedRetriever` succeeds
- [ ] Query returns results from code-reference-engine, Neo4j, and books
- [ ] Results are ranked by relevance across sources
- [ ] Citations correctly identify source type (code, book, graph)
- [ ] `cross_reference("repository pattern")` returns mixed results
- [ ] `pytest tests/unit/retrieval/` passes

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-29 | Initial WBS based on AGENT_FUNCTIONS_ARCHITECTURE.md v1.2.0 |
| 1.1.0 | 2025-12-29 | Added WBS-AGT21-24: Code Reference Engine, Neo4j, Book/JSON, and Unified Knowledge Retrieval integration |
