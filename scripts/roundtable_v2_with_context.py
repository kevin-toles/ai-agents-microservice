#!/usr/bin/env python3
"""
Round Table v2: Architecture Review with Platform Context Injection

This script runs a focused Round Table discussion addressing:
1. Task Protocol / Execution Contract gaps (Issue 1)
2. Missing Router/Planner layer (Issue 2)
3. Reconciliation of ADR, Round Table Findings, and ADK Migration Guide
4. Decision: Single initiative vs split infrastructure/agents

The key difference from v1: Platform Context is injected between rounds
to provide codebase-specific evidence that LLMs cannot access directly.
"""

import asyncio
import json
import httpx
from datetime import datetime
from pathlib import Path

# Configuration
LLM_GATEWAY = "http://localhost:8080"
SEMANTIC_SEARCH = "http://localhost:8081"
OUTPUT_DIR = Path("/Users/kevintoles/POC/Platform Documentation/architecture_roundtable_v2")

# Participants with their focus areas
PARTICIPANTS = [
    {
        "name": "Qwen",
        "role": "Principal Engineer - Workflow Orchestration",
        "expertise": "State machines, workflow engines, task execution, ADK patterns",
        "model": "qwen2.5-7b",
        "provider": "local"
    },
    {
        "name": "DeepSeek", 
        "role": "Principal Engineer - Platform Architecture",
        "expertise": "Service boundaries, API design, protocol specification",
        "model": "deepseek-r1-7b",
        "provider": "local"
    },
    {
        "name": "GPT",
        "role": "Principal Engineer - Systems Integration",
        "expertise": "Contract testing, schema validation, deterministic execution",
        "model": "gpt-5.2",
        "provider": "openai"
    },
    {
        "name": "Claude",
        "role": "Principal Engineer - Developer Experience",
        "expertise": "API ergonomics, error handling, progressive disclosure",
        "model": "claude-opus-4-20250514",
        "provider": "anthropic"
    }
]

# ============================================================================
# PLATFORM CONTEXT INJECTIONS (This is what I provide between rounds)
# ============================================================================

PLATFORM_CONTEXT = {
    "current_architecture": """
## Current Platform State (from codebase analysis)

### Services Implemented:
- llm-gateway:8080 - Routes requests, auth, rate limiting
- ai-agents:8082 - Pipeline orchestrator (Kitchen Brigade)
- semantic-search:8081 - Qdrant + Neo4j hybrid search
- Code-Orchestrator:8083 - SBERT, CodeT5+, GraphCodeBERT, CodeBERT
- inference-service:8085 - Local LLM hosting (qwen2.5-7b, deepseek-r1-7b, phi-4)
- audit-service:8084 - Citation validation, footnotes

### What EXISTS in ai-agents/src/:
- pipelines/agents.py - SequentialAgent, ParallelAgent, LoopAgent (ADK-inspired)
- participants/ - LLM participant adapters
- clients/code_analysis.py - Code-Orchestrator client (17 endpoints)
- clients/semantic_search.py - Semantic search client

### What is MISSING (gaps identified):
1. NO Task Router - llm-gateway routes by model, not by task type
2. NO Task Protocol definitions - no YAML/JSON task schemas
3. NO Step validation - agents can skip steps or invent new ones
4. NO Workflow executor - SequentialAgent exists but no enforcer
5. NO Contract tests - A2A protocol undefined
""",

    "adk_migration_status": """
## ADK Migration Status (from ADK_MIGRATION_GUIDE.md)

### Current: Option C (Cherry-Pick Patterns)
- Using SequentialAgent, ParallelAgent, LoopAgent patterns
- NOT using google-adk package
- NOT using ADK state prefixes (temp:, user:, app:)
- NOT using ADK Artifacts API

### ADK Migration Phases (if adopted):
- Phase 1: Foundation (2-3 weeks) - Add ADK as optional dependency
- Phase 2: Workflow Agents (2-3 weeks) - Replace custom orchestration
- Phase 3: State Migration (1-2 weeks) - Adopt state prefixes
- Phase 4: Full Integration (2 weeks) - A2A protocol

### Key Question:
Should we implement Task Router/Executor BEFORE ADK migration,
or should ADK migration BE the implementation of Task Router/Executor?
""",

    "issue_1_analysis": """
## Issue 1: Task Protocol / Execution Contract

### The Problem:
Currently, when ai-agents receives a request:
1. It classifies the intent (somewhat)
2. It picks a pipeline (chapter_summarization, architecture_design, etc.)
3. The pipeline runs stages... but there's no ENFORCEMENT

### What's missing:
```
Task Protocol should define:
- task_id (e.g., ARCHITECTURE.RECOMMEND)
- preconditions (auth, context size, required params)
- step_list with order (Step 1 → Step 2 → Step 3)
- gates (validation between steps)
- retries/fallback behavior
- stop conditions
- trace requirements
```

### Evidence from codebase:
- ai-agents/config/pipelines.yaml - EXISTS but only defines stage ORDER
- No schema validation between stages
- No step-level failure handling
- No "current_step" tracking with enforcement

### The Code-Orchestrator Pipeline (what SHOULD happen):
1. SBERT: NL → semantic embeddings
2. CodeT5+: Extract keywords from code  
3. GraphCodeBERT: Validate terms, filter false positives
4. CodeBERT: Rank by NL↔Code similarity

This pipeline exists in Code-Orchestrator but ai-agents doesn't ENFORCE it.
""",

    "issue_2_analysis": """
## Issue 2: Missing Router/Planner Layer

### Current Flow:
```
VS Code → llm-gateway → (routes by model) → ai-agents/inference-service
```

### Required Flow:
```
VS Code → llm-gateway → Task Router (NEW) → Workflow Executor → Services
```

### What Task Router should do:
1. Classify intent (what CATEGORY is this request?)
2. Map to known protocol (ARCHITECTURE.RECOMMEND, REPO.AUDIT, etc.)
3. Extract parameters (services list, constraints, repo path)
4. Emit plan.json with ordered steps

### Options for implementation:
A) Rules-first router (keyword/concept extraction + graph lookup)
B) Small LLM for intent + slot filling (still deterministic steps)
C) Planner LLM (proposes steps, executor validates)

### Where to put it:
- Option 1: New module in ai-agents (ai-agents/src/router/)
- Option 2: New microservice (task-router:8086)
- Option 3: Extend llm-gateway with routing logic

### Key insight:
llm-gateway currently does AUTH + ROUTING BY MODEL
Task Router does ROUTING BY TASK TYPE + PLAN GENERATION
These are different concerns!
""",

    "reconciliation_analysis": """
## Documents to Reconcile

### 1. ARCHITECTURE_DECISION_RECORD.md (Infrastructure)
- Explicit mode declaration (docker/hybrid/native)
- Generated topology (topology.yaml → endpoints.generated.json)
- Fail-fast philosophy
- Readiness contracts
- Implementation: Phase 0-3, 7-10 weeks

### 2. ARCHITECTURE_ROUNDTABLE_FINDINGS.md (Blocking Issues)
- B1: GPU allocation undefined
- B2: Auth/secrets undocumented
- B3: A2A protocol missing
- B4: Readiness contracts unspecified
- B5: Observability not implemented

### 3. ADK_MIGRATION_GUIDE.md (Future AI-Agents)
- ADK adoption phases
- State management (temp:/user:/app:)
- Workflow agents (SequentialAgent, etc.)
- A2A protocol for external integration

### The Question:
These three documents describe OVERLAPPING but DIFFERENT concerns:
- ADR = Platform infrastructure startup/config
- Round Table = Operational gaps (auth, GPU, observability)
- ADK Migration = Agent workflow patterns

Should these be:
A) ONE initiative (sequential phases)
B) TWO initiatives (infrastructure vs agents)
C) THREE initiatives (each document = one track)
"""
}

# ============================================================================
# DISCUSSION PROMPTS
# ============================================================================

ROUND_1_PROMPT = """
# Round Table Architecture Review - Round 1

## Your Role
You are {name}, {role}.
Your expertise: {expertise}

## Context Documents Provided
1. ARCHITECTURE_DECISION_RECORD.md - Platform infrastructure decisions
2. ARCHITECTURE_ROUNDTABLE_FINDINGS.md - Previous review blocking issues
3. ADK_MIGRATION_GUIDE.md - Future AI agent migration plan
4. KITCHEN_BRIGADE_ARCHITECTURE.md - Current agent pipeline design

## Issues to Address

### Issue 1: Task Protocol / Execution Contract
The platform lacks deterministic task execution. Agents can skip steps, invent steps,
or execute in wrong order. We need:
- Task Protocol definitions (task_id, preconditions, steps, gates, retries)
- Workflow Executor that ENFORCES step order (not "agent goodwill")
- Schema validation between steps
- Checkpoint artifacts for multi-agent work

### Issue 2: Missing Router/Planner Layer
Currently: VS Code → llm-gateway → ai-agents (ad-hoc routing)
Needed: VS Code → llm-gateway → Task Router → Workflow Executor → Services

The Task Router should:
- Classify intent to known task types
- Map to protocol with defined steps
- Extract parameters
- Emit execution plan

## Platform Context
{platform_context}

## Your Task
Provide your initial assessment:
1. Which issue is more critical to address first?
2. How does this relate to the ADK migration timeline?
3. Should infrastructure (ADR) be implemented before or after agent improvements?
4. Propose a concrete implementation approach

Format your response with clear sections:
## My Assessment
## Priority Recommendation
## Implementation Approach
## Questions for Other Engineers
"""

ROUND_2_PROMPT = """
# Round Table Architecture Review - Round 2

## Your Role
You are {name}, {role}.
Your expertise: {expertise}

## Previous Round Summary
{previous_round_summary}

## Platform Context Update
{platform_context}

## Your Task
1. Respond to other engineers' proposals
2. Build on agreements, challenge disagreements
3. Propose specific technical decisions
4. Consider ADK migration implications

## Key Questions to Resolve
1. Task Router: Where should it live? (ai-agents module vs new service)
2. Task Protocol: JSON Schema vs Pydantic vs protobuf?
3. Sequencing: Infrastructure first or agent improvements first?
4. ADK: Adopt now as part of this work, or defer?

Format your response:
## Response to Other Engineers
## Technical Decisions I Propose
## Implementation Sequence
## ADK Migration Stance
"""

ROUND_3_PROMPT = """
# Round Table Architecture Review - Round 3 (Consensus Building)

## Your Role
You are {name}, {role}.

## Discussion So Far
{discussion_summary}

## Platform Context
{platform_context}

## Consensus Items (from previous rounds)
{consensus_items}

## Remaining Disagreements
{disagreements}

## Your Task
1. State your final position on each unresolved item
2. Propose compromises where possible
3. Identify blocking vs deferrable issues
4. Vote on the key decisions

## Decisions Requiring Vote
1. [ ] Implement Task Router as ai-agents module (vs new service)
2. [ ] Use JSON Schema for Task Protocols (vs Pydantic)
3. [ ] Infrastructure (ADR) BEFORE agent improvements
4. [ ] Defer ADK migration until after Task Protocol implementation
5. [ ] Single initiative with phased rollout (vs split tracks)

Format your response:
## My Votes (YES/NO/ABSTAIN for each)
## Rationale
## Proposed Compromise (if disagreeing)
## Final Recommendation
"""

async def call_llm(participant: dict, prompt: str) -> str:
    """Call LLM via gateway."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{LLM_GATEWAY}/v1/chat/completions",
                json={
                    "model": participant["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 4000,
                },
                headers={"X-Provider": participant["provider"]}
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[ERROR from {participant['name']}]: {str(e)}"

async def search_context(query: str) -> str:
    """Search semantic search for additional context."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{SEMANTIC_SEARCH}/v1/search",
                json={"query": query, "top_k": 3, "collection": "chapters"}
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                return "\n".join([f"- {r.get('content', '')[:200]}..." for r in results])
            return "[No results]"
        except Exception:
            return "[Search unavailable]"

async def run_round(round_num: int, prompt_template: str, context: dict, previous: str = "") -> dict:
    """Run a discussion round with all participants."""
    print(f"\n{'='*60}")
    print(f"ROUND {round_num}")
    print(f"{'='*60}")
    
    responses = {}
    
    for participant in PARTICIPANTS:
        print(f"\n→ {participant['name']} ({participant['model']})...")
        
        prompt = prompt_template.format(
            name=participant["name"],
            role=participant["role"],
            expertise=participant["expertise"],
            platform_context=context.get("platform_context", ""),
            previous_round_summary=previous,
            discussion_summary=context.get("discussion_summary", ""),
            consensus_items=context.get("consensus_items", ""),
            disagreements=context.get("disagreements", "")
        )
        
        response = await call_llm(participant, prompt)
        responses[participant["name"]] = response
        print(f"  ✓ {participant['name']} responded ({len(response)} chars)")
    
    return responses

def extract_consensus_and_disagreements(responses: dict) -> tuple:
    """Simple extraction of agreements and disagreements."""
    # This is a simplified version - in production, use LLM for extraction
    consensus = []
    disagreements = []
    
    # Look for common themes
    all_text = " ".join(responses.values()).lower()
    
    if "task router" in all_text and "ai-agents" in all_text:
        if all_text.count("ai-agents module") > all_text.count("new service"):
            consensus.append("Task Router should be an ai-agents module (majority)")
        else:
            disagreements.append("Task Router location: ai-agents module vs new service")
    
    if "infrastructure first" in all_text:
        consensus.append("Infrastructure (ADR) should be implemented first")
    
    if "defer adk" in all_text.lower():
        consensus.append("ADK migration should be deferred until Task Protocol is done")
    
    return consensus, disagreements

async def main():
    """Run the full Round Table discussion."""
    print("="*60)
    print("ROUND TABLE v2: Architecture Review with Platform Context")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Gather additional context via semantic search
    print("\n📚 Gathering additional context...")
    search_results = await search_context("task protocol workflow executor deterministic execution")
    
    # Build full platform context
    full_context = "\n\n".join([
        PLATFORM_CONTEXT["current_architecture"],
        PLATFORM_CONTEXT["issue_1_analysis"],
        PLATFORM_CONTEXT["issue_2_analysis"],
        PLATFORM_CONTEXT["adk_migration_status"],
        PLATFORM_CONTEXT["reconciliation_analysis"],
        f"\n## Additional Search Results\n{search_results}"
    ])
    
    all_responses = {}
    
    # Round 1: Initial Assessment
    round1_responses = await run_round(
        1, 
        ROUND_1_PROMPT, 
        {"platform_context": full_context}
    )
    all_responses["round1"] = round1_responses
    
    # Extract themes from Round 1
    round1_summary = "\n\n".join([
        f"### {name}\n{resp[:1500]}..." 
        for name, resp in round1_responses.items()
    ])
    
    # Round 2: Technical Decisions
    round2_responses = await run_round(
        2,
        ROUND_2_PROMPT,
        {
            "platform_context": PLATFORM_CONTEXT["adk_migration_status"],
            "previous_round_summary": round1_summary
        },
        previous=round1_summary
    )
    all_responses["round2"] = round2_responses
    
    # Extract consensus and disagreements
    consensus, disagreements = extract_consensus_and_disagreements(round2_responses)
    
    round2_summary = "\n\n".join([
        f"### {name}\n{resp[:1500]}..."
        for name, resp in round2_responses.items()
    ])
    
    # Round 3: Consensus Building
    round3_responses = await run_round(
        3,
        ROUND_3_PROMPT,
        {
            "platform_context": PLATFORM_CONTEXT["reconciliation_analysis"],
            "discussion_summary": round1_summary + "\n\n" + round2_summary,
            "consensus_items": "\n".join([f"- {c}" for c in consensus]) or "None identified yet",
            "disagreements": "\n".join([f"- {d}" for d in disagreements]) or "None identified"
        }
    )
    all_responses["round3"] = round3_responses
    
    # Generate final report
    print("\n📝 Generating final report...")
    
    report = f"""# Round Table v2: Architecture Review Report

**Generated:** {datetime.now().isoformat()}
**Rounds:** 3
**Focus:** Task Protocol, Router/Planner, Document Reconciliation

---

## Executive Summary

This review addressed two critical issues identified in prior discussions:
1. **Issue 1:** Lack of Task Protocol / Execution Contract
2. **Issue 2:** Missing Router/Planner layer between VS Code and deterministic execution

The panel was asked to reconcile three documents:
- ARCHITECTURE_DECISION_RECORD.md (Infrastructure)
- ARCHITECTURE_ROUNDTABLE_FINDINGS.md (Blocking Issues)
- ADK_MIGRATION_GUIDE.md (Agent Migration)

---

## Participants

| Engineer | Role | Model |
|----------|------|-------|
| Qwen | Workflow Orchestration | qwen2.5-7b |
| DeepSeek | Platform Architecture | deepseek-r1-7b |
| GPT | Systems Integration | gpt-5.2 |
| Claude | Developer Experience | claude-opus-4-20250514 |

---

## Round 1: Initial Assessments

"""
    
    for name, response in round1_responses.items():
        report += f"### {name}\n\n{response}\n\n---\n\n"
    
    report += """
## Round 2: Technical Decisions

"""
    
    for name, response in round2_responses.items():
        report += f"### {name}\n\n{response}\n\n---\n\n"
    
    report += """
## Round 3: Consensus Building & Votes

"""
    
    for name, response in round3_responses.items():
        report += f"### {name}\n\n{response}\n\n---\n\n"
    
    report += f"""
## Platform Context Provided

The following context was injected to inform the discussion:

### Current Architecture State
{PLATFORM_CONTEXT['current_architecture']}

### Issue 1 Analysis
{PLATFORM_CONTEXT['issue_1_analysis']}

### Issue 2 Analysis
{PLATFORM_CONTEXT['issue_2_analysis']}

### ADK Migration Status
{PLATFORM_CONTEXT['adk_migration_status']}

### Reconciliation Analysis
{PLATFORM_CONTEXT['reconciliation_analysis']}

---

*This review was conducted with platform context injection to ensure LLMs had accurate codebase information.*
"""
    
    # Save report
    report_path = OUTPUT_DIR / "ROUNDTABLE_V2_REPORT.md"
    report_path.write_text(report)
    print(f"\n✅ Report saved: {report_path}")
    
    # Save raw responses as JSON
    json_path = OUTPUT_DIR / "roundtable_v2_responses.json"
    json_path.write_text(json.dumps(all_responses, indent=2))
    print(f"✅ Raw responses saved: {json_path}")
    
    print("\n" + "="*60)
    print("ROUND TABLE v2 COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
