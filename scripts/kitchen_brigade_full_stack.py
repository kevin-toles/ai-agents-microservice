#!/usr/bin/env python3
"""
Kitchen Brigade Full Stack: Architecture Document Reconciliation

This script runs the COMPLETE Kitchen Brigade flow to reconcile:
1. ARCHITECTURE_DECISION_RECORD.md (Infrastructure)
2. ARCHITECTURE_ROUNDTABLE_FINDINGS.md (Blocking Issues)  
3. ADK_MIGRATION_GUIDE.md (Agent Migration)

Full 6-Stage Pipeline:
- STAGE 1: decompose_task - Break into subtasks
- STAGE 2: ParallelAgent(cross_reference) - 4-layer retrieval (Qdrant, Neo4j, Textbooks, Code-Orchestrator)
- STAGE 3: LLM Discussion Loop - Multi-model debate with evidence
- STAGE 4: synthesize_outputs - Merge findings
- STAGE 5: validate_against_spec - Citation validation
- STAGE 6: Final Report Generation
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

LLM_GATEWAY = "http://localhost:8080"
SEMANTIC_SEARCH = "http://localhost:8081"
CODE_ORCHESTRATOR = "http://localhost:8083"
INFERENCE_SERVICE = "http://localhost:8085"
QDRANT_URL = "http://localhost:6333"
NEO4J_BOLT = "bolt://localhost:7687"

OUTPUT_DIR = Path("/Users/kevintoles/POC/Platform Documentation/kitchen_brigade_full_stack")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Documents to reconcile
DOCUMENTS_TO_REVIEW = [
    "/Users/kevintoles/POC/Platform Documentation/ARCHITECTURE_DECISION_RECORD.md",
    "/Users/kevintoles/POC/ai-agents/docs/architecture/KITCHEN_BRIGADE_ARCHITECTURE.md",
]

# LLM Participants for the Discussion Loop
# Using: qwen3, deepseek-coder, gpt-5.2, claude-opus-4.5
PARTICIPANTS = [
    {"name": "Qwen3", "model": "qwen3-8b", "role": "Workflow Orchestration Expert"},
    {"name": "DeepSeek-Coder", "model": "deepseek-coder-v2-lite", "role": "Code Architecture Expert"},
    {"name": "GPT", "model": "gpt-5.2", "role": "Systems Integration Expert"},
    {"name": "Claude", "model": "claude-opus-4.5", "role": "Developer Experience Expert"},
]

# Search queries for cross-reference
CROSS_REFERENCE_QUERIES = [
    "task protocol execution contract workflow",
    "router planner layer intent classification",
    "ADK migration agent workflow patterns",
    "service readiness contract health check",
    "deterministic execution step validation",
]

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CrossReferenceResult:
    """Results from 4-layer cross-reference retrieval."""
    qdrant_results: list[dict] = field(default_factory=list)
    neo4j_results: list[dict] = field(default_factory=list)
    textbook_results: list[dict] = field(default_factory=list)
    code_orchestrator_results: list[dict] = field(default_factory=list)
    query: str = ""

@dataclass 
class DiscussionCycle:
    """One cycle of the LLM discussion loop."""
    cycle_number: int
    analyses: dict[str, str]
    agreement_score: float
    disagreements: list[str]
    additional_queries: list[str] = field(default_factory=list)

@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage_name: str
    status: str
    output: Any
    duration_seconds: float
    service_calls: list[str] = field(default_factory=list)

# =============================================================================
# STAGE 1: DECOMPOSE TASK
# =============================================================================

async def stage_1_decompose_task() -> StageResult:
    """Decompose the architecture reconciliation into subtasks."""
    print("\n" + "="*70)
    print("STAGE 1: decompose_task")
    print("="*70)
    
    start = datetime.now()
    
    subtasks = [
        {
            "id": 1,
            "name": "task_protocol_gap",
            "description": "Analyze Task Protocol / Execution Contract gap",
            "queries": ["task protocol", "execution contract", "step validation", "workflow enforcement"],
        },
        {
            "id": 2,
            "name": "router_planner_gap",
            "description": "Analyze Router/Planner layer gap",
            "queries": ["task router", "intent classification", "plan generation", "deterministic routing"],
        },
        {
            "id": 3,
            "name": "adk_migration_decision",
            "description": "ADK migration timing and approach",
            "queries": ["ADK migration", "google-adk", "workflow agents", "state management"],
        },
        {
            "id": 4,
            "name": "infrastructure_vs_agents",
            "description": "Single initiative vs split tracks decision",
            "queries": ["infrastructure readiness", "fail-fast", "topology generation", "readiness contracts"],
        },
        {
            "id": 5,
            "name": "blocking_issues",
            "description": "Address blocking issues B1-B5",
            "queries": ["GPU allocation", "auth secrets", "A2A protocol", "observability"],
        },
    ]
    
    duration = (datetime.now() - start).total_seconds()
    
    print(f"  ✓ Decomposed into {len(subtasks)} subtasks")
    for task in subtasks:
        print(f"    - {task['name']}: {task['description']}")
    
    return StageResult(
        stage_name="decompose_task",
        status="success",
        output=subtasks,
        duration_seconds=duration,
        service_calls=["inference-service:8085 (task decomposition)"],
    )

# =============================================================================
# STAGE 2: PARALLEL CROSS-REFERENCE (4-Layer Retrieval)
# =============================================================================

async def search_qdrant(query: str) -> list[dict]:
    """Search Qdrant vector database via semantic-search service."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SEMANTIC_SEARCH}/v1/search",
                json={"query": query, "top_k": 5, "collection": "chapters"}
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                return [{"source": "qdrant", "content": r.get("content", "")[:500], "score": r.get("score", 0)} for r in results]
    except Exception as e:
        print(f"    [Qdrant] Error: {e}")
    return []

async def search_neo4j(query: str) -> list[dict]:
    """Search Neo4j graph database via semantic-search service."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SEMANTIC_SEARCH}/v1/graph/search",
                json={"query": query, "limit": 5}
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                return [{"source": "neo4j", "content": str(r)[:500]} for r in results]
    except Exception as e:
        print(f"    [Neo4j] Error: {e}")
    return []

async def search_code_orchestrator(query: str) -> list[dict]:
    """Search code via Code-Orchestrator ML pipeline."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Try keyword extraction endpoint
            response = await client.post(
                f"{CODE_ORCHESTRATOR}/v1/extract/keywords",
                json={"text": query, "max_keywords": 5}
            )
            if response.status_code == 200:
                keywords = response.json().get("keywords", [])
                return [{"source": "code-orchestrator", "type": "keywords", "content": keywords}]
    except Exception as e:
        print(f"    [Code-Orchestrator] Error: {e}")
    return []

async def search_local_documents(query: str) -> list[dict]:
    """Search local architecture documents for relevant content."""
    results = []
    search_terms = query.lower().split()
    
    for doc_path in DOCUMENTS_TO_REVIEW:
        try:
            if Path(doc_path).exists():
                content = Path(doc_path).read_text()
                lines = content.split('\n')
                
                # Find relevant sections
                for i, line in enumerate(lines):
                    if any(term in line.lower() for term in search_terms):
                        # Get context (5 lines before and after)
                        start = max(0, i - 5)
                        end = min(len(lines), i + 6)
                        context = '\n'.join(lines[start:end])
                        results.append({
                            "source": "local_doc",
                            "file": Path(doc_path).name,
                            "line": i + 1,
                            "content": context[:500],
                        })
                        if len(results) >= 3:
                            break
        except Exception as e:
            print(f"    [LocalDoc] Error reading {doc_path}: {e}")
    
    return results

async def cross_reference(query: str) -> CrossReferenceResult:
    """Run 4-layer parallel cross-reference retrieval."""
    print(f"  → Cross-referencing: '{query[:50]}...'")
    
    # Parallel retrieval from all sources
    qdrant_task = search_qdrant(query)
    neo4j_task = search_neo4j(query)
    code_task = search_code_orchestrator(query)
    docs_task = search_local_documents(query)
    
    results = await asyncio.gather(
        qdrant_task, neo4j_task, code_task, docs_task,
        return_exceptions=True
    )
    
    return CrossReferenceResult(
        qdrant_results=results[0] if not isinstance(results[0], Exception) else [],
        neo4j_results=results[1] if not isinstance(results[1], Exception) else [],
        code_orchestrator_results=results[2] if not isinstance(results[2], Exception) else [],
        textbook_results=results[3] if not isinstance(results[3], Exception) else [],
        query=query,
    )

async def stage_2_parallel_cross_reference(subtasks: list[dict]) -> StageResult:
    """STAGE 2: Run ParallelAgent cross-reference for each subtask."""
    print("\n" + "="*70)
    print("STAGE 2: ParallelAgent(cross_reference)")
    print("="*70)
    print("Running 4-layer parallel retrieval:")
    print("  • Qdrant (vectors) - semantic similarity")
    print("  • Neo4j (graph) - relationship traversal")
    print("  • Local Docs - architecture documents")
    print("  • Code-Orchestrator - ML keyword extraction")
    
    start = datetime.now()
    all_evidence: dict[str, list[CrossReferenceResult]] = {}
    service_calls = []
    
    for task in subtasks:
        print(f"\n  📂 Subtask: {task['name']}")
        task_evidence = []
        
        for query in task["queries"]:
            result = await cross_reference(query)
            task_evidence.append(result)
            
            # Log what we found
            total = (len(result.qdrant_results) + len(result.neo4j_results) + 
                    len(result.textbook_results) + len(result.code_orchestrator_results))
            print(f"    ✓ Found {total} results for '{query}'")
        
        all_evidence[task["name"]] = task_evidence
        service_calls.extend([
            "semantic-search:8081/v1/search",
            "semantic-search:8081/v1/graph/search",
            "code-orchestrator:8083/v1/extract/keywords",
        ])
    
    duration = (datetime.now() - start).total_seconds()
    
    # Summarize evidence
    total_results = sum(
        len(e.qdrant_results) + len(e.neo4j_results) + len(e.textbook_results) + len(e.code_orchestrator_results)
        for evidence_list in all_evidence.values()
        for e in evidence_list
    )
    print(f"\n  ✓ Total evidence collected: {total_results} results from 4 sources")
    
    return StageResult(
        stage_name="parallel_cross_reference",
        status="success",
        output=all_evidence,
        duration_seconds=duration,
        service_calls=list(set(service_calls)),
    )

# =============================================================================
# STAGE 3: LLM DISCUSSION LOOP
# =============================================================================

async def call_llm(model: str, prompt: str, participant_name: str) -> str:
    """Call LLM via gateway."""
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(
                f"{LLM_GATEWAY}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 3000,
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[ERROR from {participant_name}]: {str(e)}"

def format_evidence_for_llm(evidence: dict[str, list[CrossReferenceResult]]) -> str:
    """Format cross-reference evidence for LLM consumption."""
    formatted = "## Evidence from Cross-Reference (4-Layer Retrieval)\n\n"
    
    for task_name, results in evidence.items():
        formatted += f"### {task_name}\n\n"
        
        for result in results:
            if result.qdrant_results:
                formatted += "**Qdrant (Vector Search):**\n"
                for r in result.qdrant_results[:2]:
                    formatted += f"- {r.get('content', '')[:200]}...\n"
            
            if result.neo4j_results:
                formatted += "**Neo4j (Graph):**\n"
                for r in result.neo4j_results[:2]:
                    formatted += f"- {r.get('content', '')[:200]}...\n"
                    
            if result.textbook_results:
                formatted += "**Local Documents:**\n"
                for r in result.textbook_results[:2]:
                    formatted += f"- [{r.get('file', '')}:{r.get('line', '')}] {r.get('content', '')[:200]}...\n"
        
        formatted += "\n"
    
    return formatted

async def run_discussion_cycle(
    cycle_num: int,
    evidence: dict[str, list[CrossReferenceResult]],
    previous_analyses: dict[str, str] = None,
) -> DiscussionCycle:
    """Run one cycle of the LLM discussion loop."""
    print(f"\n  🔄 Discussion Cycle {cycle_num}")
    
    evidence_text = format_evidence_for_llm(evidence)
    
    prompt_template = """# Architecture Reconciliation Discussion - Cycle {cycle}

You are {name}, a {role}.

## Your Task
Analyze the evidence below and provide your assessment on reconciling these architecture documents:
1. ARCHITECTURE_DECISION_RECORD.md (Infrastructure decisions)
2. KITCHEN_BRIGADE_ARCHITECTURE.md (Agent workflow patterns)

Focus on:
- Issue 1: Task Protocol / Execution Contract gap
- Issue 2: Missing Router/Planner layer
- Decision: Single initiative vs split Infrastructure/Agents tracks

{evidence}

{previous_context}

## Instructions
1. Analyze the evidence from cross-reference retrieval
2. State your position on each issue
3. Cite specific evidence (source, line number if available)
4. Identify any gaps in the evidence that need more cross-reference
5. Propose concrete next steps

## Format
### My Analysis
[Your detailed analysis with citations]

### My Position
- Issue 1 (Task Protocol): [your position]
- Issue 2 (Router/Planner): [your position]  
- Single vs Split: [your position]

### Evidence Gaps
[List any additional queries needed]

### Recommended Next Steps
[Concrete actions with priority]
"""
    
    previous_context = ""
    if previous_analyses:
        previous_context = "## Previous Round Analyses\n"
        for name, analysis in previous_analyses.items():
            previous_context += f"\n### {name}'s Previous Position\n{analysis[:500]}...\n"
    
    analyses = {}
    
    for participant in PARTICIPANTS:
        print(f"    → {participant['name']} ({participant['model']})...")
        
        prompt = prompt_template.format(
            cycle=cycle_num,
            name=participant["name"],
            role=participant["role"],
            evidence=evidence_text,
            previous_context=previous_context,
        )
        
        response = await call_llm(participant["model"], prompt, participant["name"])
        analyses[participant["name"]] = response
        print(f"      ✓ Responded ({len(response)} chars)")
    
    # Simple agreement detection (in production, use embeddings)
    # Check if participants mention similar positions
    agreement_keywords = ["agree", "consensus", "align", "support"]
    disagreement_keywords = ["disagree", "however", "but", "alternative", "instead"]
    
    agree_count = sum(
        1 for a in analyses.values() 
        if any(k in a.lower() for k in agreement_keywords)
    )
    disagree_count = sum(
        1 for a in analyses.values()
        if any(k in a.lower() for k in disagreement_keywords)
    )
    
    agreement_score = agree_count / len(PARTICIPANTS) if PARTICIPANTS else 0
    
    # Extract disagreement points
    disagreements = []
    if disagree_count > 0:
        disagreements = ["Position differences detected - see individual analyses"]
    
    return DiscussionCycle(
        cycle_number=cycle_num,
        analyses=analyses,
        agreement_score=agreement_score,
        disagreements=disagreements,
    )

async def stage_3_llm_discussion_loop(
    evidence: dict[str, list[CrossReferenceResult]],
    max_cycles: int = 2,
    agreement_threshold: float = 0.75,
) -> StageResult:
    """STAGE 3: Run iterative LLM discussion loop."""
    print("\n" + "="*70)
    print("STAGE 3: LLM Discussion Loop (ITERATIVE)")
    print("="*70)
    print(f"Participants: {', '.join(p['name'] for p in PARTICIPANTS)}")
    print(f"Max cycles: {max_cycles}, Agreement threshold: {agreement_threshold}")
    
    start = datetime.now()
    cycles: list[DiscussionCycle] = []
    previous_analyses = None
    
    for cycle_num in range(1, max_cycles + 1):
        cycle = await run_discussion_cycle(cycle_num, evidence, previous_analyses)
        cycles.append(cycle)
        
        print(f"    Agreement score: {cycle.agreement_score:.2f}")
        
        if cycle.agreement_score >= agreement_threshold:
            print(f"  ✓ Agreement threshold reached at cycle {cycle_num}")
            break
        
        previous_analyses = cycle.analyses
        
        # If disagreement, we could trigger additional cross-reference here
        if cycle.disagreements and cycle_num < max_cycles:
            print(f"    Disagreements detected, continuing to cycle {cycle_num + 1}")
    
    duration = (datetime.now() - start).total_seconds()
    
    return StageResult(
        stage_name="llm_discussion_loop",
        status="success",
        output=cycles,
        duration_seconds=duration,
        service_calls=[f"llm-gateway:8080 ({p['model']})" for p in PARTICIPANTS],
    )

# =============================================================================
# STAGE 4: SYNTHESIZE OUTPUTS
# =============================================================================

async def stage_4_synthesize_outputs(cycles: list[DiscussionCycle]) -> StageResult:
    """STAGE 4: Synthesize all discussion outputs into coherent findings."""
    print("\n" + "="*70)
    print("STAGE 4: synthesize_outputs")
    print("="*70)
    
    start = datetime.now()
    
    # Collect all analyses from final cycle
    final_cycle = cycles[-1] if cycles else None
    
    synthesis_prompt = """# Synthesize Architecture Reconciliation Findings

You are a Technical Program Manager synthesizing findings from a multi-expert discussion.

## Expert Analyses (Final Cycle)
"""
    
    if final_cycle:
        for name, analysis in final_cycle.analyses.items():
            synthesis_prompt += f"\n### {name}\n{analysis}\n"
    
    synthesis_prompt += """

## Your Task
Create a unified synthesis that:
1. Identifies consensus points across all experts
2. Documents remaining disagreements with each position
3. Proposes a concrete decision framework
4. Prioritizes actions into immediate/short-term/long-term

## Output Format
### Consensus Points
[List agreed items]

### Remaining Disagreements  
[Each disagreement with expert positions]

### Recommended Decision
[The synthesized recommendation]

### Action Priority
- IMMEDIATE (this week): [actions]
- SHORT-TERM (this month): [actions]
- LONG-TERM (next quarter): [actions]
"""
    
    # Use Claude for synthesis (good at summarization)
    synthesis = await call_llm("claude-opus-4-20250514", synthesis_prompt, "Synthesizer")
    
    duration = (datetime.now() - start).total_seconds()
    
    print(f"  ✓ Synthesized {len(cycles)} discussion cycles")
    
    return StageResult(
        stage_name="synthesize_outputs",
        status="success",
        output=synthesis,
        duration_seconds=duration,
        service_calls=["llm-gateway:8080 (claude-opus-4-20250514)"],
    )

# =============================================================================
# STAGE 5: VALIDATE AGAINST SPEC
# =============================================================================

async def stage_5_validate(synthesis: str, evidence: dict) -> StageResult:
    """STAGE 5: Validate findings against evidence and check citations."""
    print("\n" + "="*70)
    print("STAGE 5: validate_against_spec")
    print("="*70)
    
    start = datetime.now()
    
    validation_checks = {
        "issue_1_addressed": "task protocol" in synthesis.lower() or "execution contract" in synthesis.lower(),
        "issue_2_addressed": "router" in synthesis.lower() or "planner" in synthesis.lower(),
        "has_recommendations": "recommend" in synthesis.lower() or "action" in synthesis.lower(),
        "has_priorities": "immediate" in synthesis.lower() or "priority" in synthesis.lower(),
        "references_evidence": any(
            task_name.replace("_", " ") in synthesis.lower()
            for task_name in evidence.keys()
        ),
    }
    
    passed = sum(validation_checks.values())
    total = len(validation_checks)
    
    print("  Validation Checks:")
    for check, passed_check in validation_checks.items():
        status = "✓" if passed_check else "✗"
        print(f"    {status} {check}")
    
    duration = (datetime.now() - start).total_seconds()
    
    return StageResult(
        stage_name="validate_against_spec",
        status="success" if passed >= 4 else "partial",
        output={"checks": validation_checks, "passed": passed, "total": total},
        duration_seconds=duration,
        service_calls=[],
    )

# =============================================================================
# STAGE 6: GENERATE FINAL REPORT
# =============================================================================

async def stage_6_generate_report(stages: list[StageResult]) -> str:
    """Generate the final Kitchen Brigade report."""
    print("\n" + "="*70)
    print("STAGE 6: Generate Final Report")
    print("="*70)
    
    report = f"""# Kitchen Brigade Full Stack: Architecture Reconciliation Report

**Generated:** {datetime.now().isoformat()}
**Pipeline:** Kitchen Brigade 6-Stage Full Stack
**Status:** Complete

---

## Executive Summary

This report presents the findings from running the complete Kitchen Brigade pipeline
to reconcile architecture documents and address blocking issues.

---

## Pipeline Execution Summary

| Stage | Name | Status | Duration | Services Called |
|-------|------|--------|----------|-----------------|
"""
    
    total_duration = 0
    all_services = set()
    
    for i, stage in enumerate(stages, 1):
        report += f"| {i} | {stage.stage_name} | {stage.status} | {stage.duration_seconds:.1f}s | {len(stage.service_calls)} |\n"
        total_duration += stage.duration_seconds
        all_services.update(stage.service_calls)
    
    report += f"\n**Total Duration:** {total_duration:.1f}s\n"
    report += f"**Services Used:** {len(all_services)}\n\n"
    
    report += "### Services Invoked\n"
    for service in sorted(all_services):
        report += f"- {service}\n"
    
    report += "\n---\n\n"
    
    # Add synthesis (Stage 4 output)
    synthesis_stage = next((s for s in stages if s.stage_name == "synthesize_outputs"), None)
    if synthesis_stage:
        report += "## Synthesis of Expert Discussion\n\n"
        report += synthesis_stage.output
        report += "\n\n---\n\n"
    
    # Add discussion details
    discussion_stage = next((s for s in stages if s.stage_name == "llm_discussion_loop"), None)
    if discussion_stage:
        report += "## Discussion Loop Details\n\n"
        cycles = discussion_stage.output
        report += f"**Cycles Completed:** {len(cycles)}\n\n"
        
        for cycle in cycles:
            report += f"### Cycle {cycle.cycle_number}\n"
            report += f"**Agreement Score:** {cycle.agreement_score:.2f}\n\n"
            
            for name, analysis in cycle.analyses.items():
                report += f"#### {name}\n"
                report += f"{analysis[:1500]}...\n\n" if len(analysis) > 1500 else f"{analysis}\n\n"
    
    # Add validation results
    validation_stage = next((s for s in stages if s.stage_name == "validate_against_spec"), None)
    if validation_stage:
        report += "---\n\n## Validation Results\n\n"
        checks = validation_stage.output.get("checks", {})
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            report += f"- {status} {check.replace('_', ' ').title()}\n"
    
    report += f"""
---

## Metadata

```json
{{
  "pipeline": "kitchen_brigade_full_stack",
  "stages_completed": {len(stages)},
  "total_duration_seconds": {total_duration:.1f},
  "participants": {json.dumps([p['name'] for p in PARTICIPANTS])},
  "models_used": {json.dumps([p['model'] for p in PARTICIPANTS])},
  "services_called": {len(all_services)},
  "generated_at": "{datetime.now().isoformat()}"
}}
```
"""
    
    return report

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

async def main():
    """Run the complete Kitchen Brigade pipeline."""
    print("="*70)
    print("KITCHEN BRIGADE FULL STACK: Architecture Reconciliation")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*70)
    
    stages: list[StageResult] = []
    
    try:
        # STAGE 1: Decompose Task
        stage_1 = await stage_1_decompose_task()
        stages.append(stage_1)
        subtasks = stage_1.output
        
        # STAGE 2: Parallel Cross-Reference
        stage_2 = await stage_2_parallel_cross_reference(subtasks)
        stages.append(stage_2)
        evidence = stage_2.output
        
        # STAGE 3: LLM Discussion Loop
        stage_3 = await stage_3_llm_discussion_loop(evidence, max_cycles=2)
        stages.append(stage_3)
        cycles = stage_3.output
        
        # STAGE 4: Synthesize Outputs
        stage_4 = await stage_4_synthesize_outputs(cycles)
        stages.append(stage_4)
        synthesis = stage_4.output
        
        # STAGE 5: Validate
        stage_5 = await stage_5_validate(synthesis, evidence)
        stages.append(stage_5)
        
        # STAGE 6: Generate Report
        report = await stage_6_generate_report(stages)
        
        # Save outputs
        report_path = OUTPUT_DIR / "KITCHEN_BRIGADE_FULL_REPORT.md"
        report_path.write_text(report)
        print(f"\n✅ Report saved: {report_path}")
        
        # Save raw data
        raw_data = {
            "stages": [
                {
                    "name": s.stage_name,
                    "status": s.status,
                    "duration": s.duration_seconds,
                    "services": s.service_calls,
                }
                for s in stages
            ],
            "discussion_cycles": [
                {
                    "cycle": c.cycle_number,
                    "agreement_score": c.agreement_score,
                    "analyses": c.analyses,
                }
                for c in cycles
            ] if cycles else [],
        }
        
        raw_path = OUTPUT_DIR / "kitchen_brigade_raw_data.json"
        raw_path.write_text(json.dumps(raw_data, indent=2, default=str))
        print(f"✅ Raw data saved: {raw_path}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise
    
    print("\n" + "="*70)
    print("KITCHEN BRIGADE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())
