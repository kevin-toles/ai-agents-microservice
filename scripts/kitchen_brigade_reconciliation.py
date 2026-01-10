#!/usr/bin/env python3
"""
Kitchen Brigade Architecture Reconciliation

Autonomous multi-LLM review to reconcile 4 core architecture documents:
1. ARCHITECTURE_DECISION_RECORD.md - Infrastructure
2. ARCHITECTURE_ROUNDTABLE_FINDINGS.md - Blocking Issues
3. ADK_MIGRATION_GUIDE.md - Agent Migration
4. UNIFIED_KITCHEN_BRIGADE_ARCHITECTURE.md - CORE orchestration

This script runs WITHOUT manual approval for each step.
User only approves:
- Additional research loops
- Final recommendations before applying changes
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import httpx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.table import Table

console = Console()

# Service endpoints
LLM_GATEWAY = "http://localhost:8080"
INFERENCE_SERVICE = "http://localhost:8085"

# Document paths
PLATFORM_DOCS = Path(__file__).parent.parent.parent / "Platform Documentation"
DOCS_TO_RECONCILE = [
    PLATFORM_DOCS / "ARCHITECTURE_DECISION_RECORD.md",
    PLATFORM_DOCS / "ARCHITECTURE_ROUNDTABLE_FINDINGS.md",
    Path(__file__).parent.parent / "docs/architecture/Future/ADK_MIGRATION_GUIDE.md",
    PLATFORM_DOCS / "UNIFIED_KITCHEN_BRIGADE_ARCHITECTURE.md",
]

# LLM assignments (Kitchen Brigade pattern)
BRIGADE_LLMS = {
    "architect": "deepseek-r1-7b",      # Synthesis, consolidation
    "critic": "qwen3-8b",               # Validation, conflict detection
    "implementer": "codellama-13b-instruct",  # Implementation planning
    "reviewer": "deepseek-coder-v2-lite",    # Final review, edge cases
}


class KitchenBrigadeOrchestrator:
    """
    Autonomous orchestrator for Kitchen Brigade architecture review.
    
    Progress is streamed to terminal.
    TODOs are created in VS Code workspace.
    User approval only for additional research loops.
    """
    
    def __init__(self):
        self.console = Console()
        self.client = httpx.AsyncClient(timeout=300.0)
        self.trace_bundle = {
            "start_time": datetime.now().isoformat(),
            "rounds": [],
            "conflicts": [],
            "resolutions": [],
            "todos": [],
        }
        
    async def run(self):
        """Main orchestration loop - runs autonomously."""
        
        self.console.print(Panel.fit(
            "[bold cyan]Kitchen Brigade Architecture Reconciliation[/bold cyan]\n"
            "[dim]Autonomous multi-LLM review of 4 core documents[/dim]",
            border_style="cyan"
        ))
        
        # Phase 1: Load documents
        documents = await self.load_documents()
        
        # Phase 2: Round 1 - Initial analysis (parallel)
        self.console.print("\n[bold yellow]━━━ Round 1: Initial Analysis (Parallel) ━━━[/bold yellow]\n")
        round1_analyses = await self.round1_parallel_analysis(documents)
        
        # Phase 3: Round 2 - Conflict identification
        self.console.print("\n[bold yellow]━━━ Round 2: Conflict Identification ━━━[/bold yellow]\n")
        conflicts = await self.round2_identify_conflicts(round1_analyses)
        
        # Phase 4: Round 3 - Resolution proposals
        self.console.print("\n[bold yellow]━━━ Round 3: Resolution Proposals ━━━[/bold yellow]\n")
        resolutions = await self.round3_propose_resolutions(conflicts)
        
        # Phase 5: Round 4 - Consensus validation
        self.console.print("\n[bold yellow]━━━ Round 4: Consensus Validation ━━━[/bold yellow]\n")
        consensus = await self.round4_validate_consensus(resolutions)
        
        # Phase 6: Check if additional research needed
        needs_research = self.check_needs_additional_research(consensus)
        
        if needs_research:
            if await self.request_user_approval_for_research(needs_research):
                self.console.print("\n[bold green]User approved additional research loop[/bold green]")
                # Recursively run another round with expanded context
                return await self.run_with_expanded_context(needs_research)
            else:
                self.console.print("\n[bold yellow]User declined additional research - proceeding with current findings[/bold yellow]")
        
        # Phase 7: Generate implementation TODOs
        self.console.print("\n[bold yellow]━━━ Generating Implementation Plan ━━━[/bold yellow]\n")
        todos = await self.generate_todos(consensus)
        
        # Phase 8: Create VS Code TODOs
        await self.create_vscode_todos(todos)
        
        # Phase 9: Save trace bundle
        await self.save_trace_bundle()
        
        # Phase 10: Present final recommendations
        await self.present_recommendations(consensus, todos)
        
        return consensus
    
    async def load_documents(self) -> Dict[str, str]:
        """Load all documents to reconcile."""
        documents = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Loading documents...", total=len(DOCS_TO_RECONCILE))
            
            for doc_path in DOCS_TO_RECONCILE:
                if doc_path.exists():
                    documents[doc_path.name] = doc_path.read_text()
                    self.console.print(f"✓ Loaded {doc_path.name} ({len(documents[doc_path.name])} chars)")
                else:
                    self.console.print(f"✗ [red]Missing: {doc_path.name}[/red]")
                progress.update(task, advance=1)
        
        return documents
    
    async def round1_parallel_analysis(self, documents: Dict[str, str]) -> Dict[str, Any]:
        """
        Round 1: Each LLM analyzes all documents in parallel.
        NO user approval - runs autonomously.
        """
        analyses = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            tasks_to_run = []
            for role, model in BRIGADE_LLMS.items():
                tasks_to_run.append(
                    self.llm_analyze_documents(role, model, documents, progress)
                )
            
            # Run all analyses in parallel
            results = await asyncio.gather(*tasks_to_run)
            
            for role, result in zip(BRIGADE_LLMS.keys(), results):
                analyses[role] = result
        
        self.trace_bundle["rounds"].append({
            "round": 1,
            "type": "parallel_analysis",
            "analyses": analyses
        })
        
        return analyses
    
    async def llm_analyze_documents(
        self, 
        role: str, 
        model: str, 
        documents: Dict[str, str],
        progress: Progress
    ) -> Dict[str, Any]:
        """Single LLM analyzes all documents for conflicts/gaps."""
        
        task = progress.add_task(f"[cyan]{role.capitalize()}[/cyan] analyzing...", total=1)
        
        prompt = self.build_analysis_prompt(role, documents)
        
        try:
            response = await self.client.post(
                f"{INFERENCE_SERVICE}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are the {role} in a Kitchen Brigade architecture review."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 4096,
                }
            )
            response.raise_for_status()
            result = response.json()
            
            analysis = result["choices"][0]["message"]["content"]
            
            # Validate JSON is parseable before returning
            parsed_json = self.extract_json_from_response(analysis)
            if parsed_json is None:
                self.console.print(f"[yellow]{role.capitalize()} response (first 500 chars):[/yellow]")
                self.console.print(f"[dim]{analysis[:500]}...[/dim]")
                self.console.print(f"[yellow]Could not extract valid JSON - storing raw response[/yellow]")
            
            progress.update(task, completed=1)
            self.console.print(f"✓ [green]{role.capitalize()}[/green] completed analysis")
            
            return {
                "role": role,
                "model": model,
                "analysis": analysis,
                "parsed_json": parsed_json,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            progress.update(task, completed=1)
            self.console.print(f"✗ [red]{role.capitalize()} failed: {e}[/red]")
            return {
                "role": role,
                "model": model,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def extract_json_from_response(self, content: str) -> dict:
        """
        Robust JSON extraction from LLM response.
        Handles markdown code blocks, extra text, multiple formats.
        """
        import re
        
        # Try 1: Look for ```json code blocks (case insensitive)
        json_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        matches = re.findall(json_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # Try 2: Look for JSON object anywhere in response
        # Find everything between first { and last }
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Try 3: Look for JSON array
        try:
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Failed all attempts
        return None
    
    def build_analysis_prompt(self, role: str, documents: Dict[str, str]) -> str:
        """Build role-specific analysis prompt."""
        
        doc_summaries = "\n\n".join([
            f"## {name}\n```markdown\n{content[:2000]}...\n```"
            for name, content in documents.items()
        ])
        
        prompts = {
            "architect": f"""CRITICAL: You must respond with ONLY valid JSON. No explanatory text before or after.

You are the ARCHITECT reviewing these 4 core platform documents:

{doc_summaries}

Your task:
1. Identify the CORE PRINCIPLE in each document
2. Find where principles OVERLAP or COMPLEMENT each other
3. Spot where principles CONFLICT or CONTRADICT
4. Assess if UNIFIED_KITCHEN_BRIGADE_ARCHITECTURE is truly the CORE or if it conflicts with infrastructure decisions

Output JSON:
{{
  "core_principles": {{"doc_name": "principle"}},
  "overlaps": [list of complementary relationships],
  "conflicts": [list of contradictions],
  "is_unified_doc_core": true/false,
  "reasoning": "explanation"
}}
""",
            
            "critic": f"""
You are the CRITIC reviewing these 4 core platform documents:

{doc_summaries}

Your task:
1. Find CONTRADICTIONS between documents
2. Identify UNDEFINED BEHAVIORS where docs disagree
3. Spot MISSING INTEGRATIONS (e.g., ADK patterns not in unified doc)
4. Flag VIOLATION of established principles

Output JSON:
{{
  "contradictions": [{{
    "doc1": "name",
    "doc2": "name", 
    "conflict": "description",
    "severity": "high/medium/low"
  }}],
  "undefined_behaviors": [list],
  "missing_integrations": [list],
  "principle_violations": [list]
CRITICAL: You must respond with ONLY valid JSON. No explanatory text before or after.

}}
""",
            
            "implementer": f"""
You are the IMPLEMENTER reviewing these 4 core platform documents:

{doc_summaries}

Your task:
1. Identify what CAN be implemented RIGHT NOW
2. Find BLOCKERS that prevent implementation
3. Spot DEPENDENCIES between documents
4. Propose IMPLEMENTATION PHASES

Output JSON:
{{
  "ready_to_implement": [list of actionable items],
  "blockers": [{{"item": "description", "blocked_by": "reason"}}],
  "dependencies": [{{"item": "description", "depends_on": ["deps"]}}],
  "proposed_phases": [{{
    "phase": number,
    "name": "name",
    "items": [list],
    "estimated_effort": "duration"
  }}]
CRITICAL: You must respond with ONLY valid JSON. No explanatory text before or after.

}}
""",
            
            "reviewer": f"""
You are the REVIEWER reviewing these 4 core platform documents:

{doc_summaries}

Your task:
1. Check COMPLETENESS - are all aspects covered?
2. Find EDGE CASES not addressed
3. Validate CONSISTENCY across documents
4. Assess PRODUCTION READINESS

Output JSON:
{{
  "completeness_gaps": [list of missing aspects],
  "edge_cases": [list of unhandled scenarios],
  "consistency_issues": [list of inconsistencies],
  "production_readiness": {{
    "score": 0-10,
    "blockers": [list],
    "recommendations": [list]
  }}
}}
"""
        }
        
        return prompts.get(role, prompts["architect"])
    
    async def round2_identify_conflicts(self, analyses: Dict[str, Any]) -> List[Dict]:
        """
        Round 2: Synthesize analyses to identify concrete conflicts.
        Architect leads, others validate.
        """
        
        self.console.print("Synthesizing conflict list from all analyses...")
        
        # Extract all conflicts/contradictions from analyses
        all_conflicts = []
        
        for role, analysis in analyses.items():
            if "error" in analysis:
                continue
            
            # Use pre-parsed JSON if available
            if "parsed_json" in analysis and analysis["parsed_json"]:
                data = analysis["parsed_json"]
                
                # Extract conflicts based on role
                if role == "critic" and "contradictions" in data:
                    all_conflicts.extend(data["contradictions"])
                elif role == "architect" and "conflicts" in data:
                    all_conflicts.extend([{"conflict": c} for c in data["conflicts"]])
            else:
                self.console.print(f"[yellow]Skipping {role} - no valid JSON parsed[/yellow]")
        
        # Deduplicate and rank by severity
        unique_conflicts = self.deduplicate_conflicts(all_conflicts)
        
        self.console.print(f"\n[bold]Found {len(unique_conflicts)} unique conflicts[/bold]")
        for i, conflict in enumerate(unique_conflicts, 1):
            severity = conflict.get("severity", "unknown")
            color = {"high": "red", "medium": "yellow", "low": "blue"}.get(severity, "white")
            self.console.print(f"  {i}. [{color}]{conflict.get('conflict', conflict)}[/{color}]")
        
        self.trace_bundle["conflicts"] = unique_conflicts
        return unique_conflicts
    
    def deduplicate_conflicts(self, conflicts: List[Dict]) -> List[Dict]:
        """Deduplicate conflicts by semantic similarity."""
        # Simple deduplication - in production, use embedding similarity
        seen = set()
        unique = []
        
        for conflict in conflicts:
            conflict_str = str(conflict.get("conflict", conflict))
            if conflict_str not in seen:
                seen.add(conflict_str)
                unique.append(conflict)
        
        return unique
    
    async def round3_propose_resolutions(self, conflicts: List[Dict]) -> List[Dict]:
        """
        Round 3: Each LLM proposes resolutions for conflicts.
        Runs in parallel.
        """
        
        self.console.print("LLMs proposing resolutions...")
        
        resolutions = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            tasks_to_run = []
            for role, model in BRIGADE_LLMS.items():
                tasks_to_run.append(
                    self.llm_propose_resolution(role, model, conflicts, progress)
                )
            
            results = await asyncio.gather(*tasks_to_run)
            
            for result in results:
                if result and "proposals" in result:
                    resolutions.extend(result["proposals"])
        
        self.trace_bundle["resolutions"] = resolutions
        return resolutions
    
    async def llm_propose_resolution(
        self,
        role: str,
        model: str,
        conflicts: List[Dict],
        progress: Progress
    ) -> Dict[str, Any]:
        """Single LLM proposes resolutions for conflicts."""
        
        task = progress.add_task(f"[cyan]{role.capitalize()}[/cyan] proposing...", total=1)
        
        prompt = f"""
You are the {role} in a Kitchen Brigade architecture review.

CONFLICTS IDENTIFIED:
{json.dumps(conflicts, indent=2)}

Your task: Propose resolutions for each conflict.

Output JSON:
{{
  "proposals": [{{
    "conflict_id": index,
    "resolution": "description of how to resolve",
    "affected_docs": ["list of docs that need updates"],
    "implementation_notes": "notes for implementer"
  }}]
}}
"""
        
        try:
            response = await self.client.post(
                f"{INFERENCE_SERVICE}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": f"You are the {role}."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 4096,
                }
            )
            response.raise_for_status()
            result = response.json()
            
            content = result["choices"][0]["message"]["content"]
            
            # Parse JSON
            data = self.extract_json_from_response(content)
            
            if data is None:
                self.console.print(f"[yellow]{role.capitalize()} response (first 500 chars):[/yellow]")
                self.console.print(f"[dim]{content[:500]}...[/dim]")
                progress.update(task, completed=1)
                return {"proposals": []}
            progress.update(task, completed=1)
            self.console.print(f"✓ [green]{role.capitalize()}[/green] proposed {len(data.get('proposals', []))} resolutions")
            
            return data
            
        except Exception as e:
            progress.update(task, completed=1)
            self.console.print(f"✗ [red]{role.capitalize()} failed: {e}[/red]")
            return {"proposals": []}
    
    async def round4_validate_consensus(self, resolutions: List[Dict]) -> Dict[str, Any]:
        """
        Round 4: Architect synthesizes all proposals into consensus.
        """
        
        self.console.print("Architect building consensus from all proposals...")
        
        prompt = f"""
You are the ARCHITECT synthesizing the final consensus.

ALL RESOLUTION PROPOSALS:
{json.dumps(resolutions, indent=2)}

Your task:
1. Identify which resolutions have CONSENSUS (multiple LLMs agree)
2. Identify DISSENT (LLMs disagree)
3. Make FINAL DECISION for each conflict
4. Prioritize by IMPACT

Output JSON:
{{
  "consensus_resolutions": [{{
    "conflict": "description",
    "resolution": "final decision",
    "supporting_roles": ["list of roles that agreed"],
    "priority": "P0/P1/P2"
  }}],
  "dissenting_opinions": [{{
    "conflict": "description",
    "dissent": "what was disagreed on"
  }}],
  "requires_additional_research": [{{
    "topic": "what needs more research",
    "reason": "why"
  }}]
}}
"""
        
        try:
            response = await self.client.post(
                f"{INFERENCE_SERVICE}/v1/chat/completions",
                json={
                    "model": BRIGADE_LLMS["architect"],
                    "messages": [
                        {"role": "system", "content": "You are the lead architect."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 8192,
                }
            )
            response.raise_for_status()
            result = response.json()
            
            content = result["choices"][0]["message"]["content"]
            
            # Parse JSON
            if "```json" in content:
            consensus = self.extract_json_from_response(content)
            
            if consensus is None:
                self.console.print(f"[red]Failed to parse consensus JSON[/red]")
                self.console.print(f"[yellow]Response (first 500 chars):[/yellow]")
                self.console.print(f"[dim]{content[:500]}...[/dim]")
                return {
                    "consensus_resolutions": [],
                    "dissenting_opinions": [],
                    "requires_additional_research": []
                }
            # Display consensus
            self.display_consensus(consensus)
            
            return consensus
            
        except Exception as e:
            self.console.print(f"[red]Consensus synthesis failed: {e}[/red]")
            return {
                "consensus_resolutions": [],
                "dissenting_opinions": [],
                "requires_additional_research": []
            }
    
    def display_consensus(self, consensus: Dict[str, Any]):
        """Display consensus in rich table."""
        
        table = Table(title="Consensus Resolutions", show_header=True)
        table.add_column("Priority", style="cyan")
        table.add_column("Conflict", style="yellow")
        table.add_column("Resolution", style="green")
        table.add_column("Support", style="blue")
        
        for res in consensus.get("consensus_resolutions", []):
            table.add_row(
                res.get("priority", "P2"),
                res.get("conflict", "")[:50] + "...",
                res.get("resolution", "")[:60] + "...",
                ", ".join(res.get("supporting_roles", []))
            )
        
        self.console.print(table)
    
    def check_needs_additional_research(self, consensus: Dict[str, Any]) -> List[Dict]:
        """Check if consensus indicates need for additional research."""
        return consensus.get("requires_additional_research", [])
    
    async def request_user_approval_for_research(self, research_topics: List[Dict]) -> bool:
        """
        REQUEST USER APPROVAL for additional research loop.
        This is the ONLY place user interaction is required.
        """
        
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold yellow]Additional Research Recommended[/bold yellow]\n\n"
            + "\n".join([f"• {t['topic']}: {t['reason']}" for t in research_topics]),
            border_style="yellow"
        ))
        
        response = self.console.input("\n[bold]Run additional research loop? (y/N): [/bold]")
        return response.lower() in ["y", "yes"]
    
    async def run_with_expanded_context(self, research_topics: List[Dict]) -> Dict:
        """Run another reconciliation loop with expanded context."""
        self.console.print("[cyan]Expanding context with additional research...[/cyan]")
        # In production, this would fetch additional documents, run semantic search, etc.
        # For now, just re-run with same context
        return await self.run()
    
    async def generate_todos(self, consensus: Dict[str, Any]) -> List[Dict]:
        """Generate implementation TODOs from consensus."""
        
        self.console.print("Generating implementation TODOs...")
        
        prompt = f"""
Based on this consensus:

{json.dumps(consensus, indent=2)}

Generate VS Code TODOs for implementation phases.

Output JSON:
{{
  "todos": [{{
    "phase": "Phase 0/1/2/3",
    "title": "TODO title",
    "description": "What needs to be done",
    "files": ["list of files to modify"],
    "estimated_effort": "duration",
    "priority": "P0/P1/P2"
  }}]
}}
"""
        
        try:
            response = await self.client.post(
                f"{INFERENCE_SERVICE}/v1/chat/completions",
                json={
                    "model": BRIGADE_LLMS["implementer"],
                    "messages": [
                        {"role": "system", "content": "You are the implementer."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 4096,
                }
            )
            response.raise_for_status()
            result = response.json()
            data = self.extract_json_from_response(content)
            
            if data is None:
                self.console.print(f"[red]Failed to parse TODOs JSON[/red]")
                self.console.print(f"[yellow]Response (first 500 chars):[/yellow]")
                self.console.print(f"[dim]{content[:500]}...[/dim]")
                return []
            
            todos = data.get("todos", [])
            self.console.print(f"✓ Generated {len(todos)} TODOs")
            
            return todos
            
        except Exception as e:
            self.console.print(f"[red]TODO generation failed: {e}[/red]")
            return []
    
    async def create_vscode_todos(self, todos: List[Dict]):
        """Create .vscode/tasks.json with TODOs."""
        
        vscode_dir = Path(__file__).parent.parent.parent / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        tasks_file = vscode_dir / "kitchen_brigade_todos.json"
        
        tasks_file.write_text(json.dumps({
            "version": "2.0.0",
            "tasks": [
                {
                    "label": f"{todo['phase']}: {todo['title']}",
                    "type": "shell",
                    "command": f"echo 'TODO: {todo['description']}'",
                    "problemMatcher": [],
                    "detail": todo.get("description", ""),
                    "group": "build"
                }
                for todo in todos
            ]
        }, indent=2))
        
        self.console.print(f"✓ Created {tasks_file}")
        self.trace_bundle["todos"] = todos
    
    async def save_trace_bundle(self):
        """Save complete trace bundle."""
        
        trace_file = PLATFORM_DOCS / "architecture_reconciliation" / f"reconciliation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        trace_file.parent.mkdir(exist_ok=True)
        
        self.trace_bundle["end_time"] = datetime.now().isoformat()
        
        trace_file.write_text(json.dumps(self.trace_bundle, indent=2))
        
        self.console.print(f"✓ Saved trace bundle: {trace_file}")
    
    async def present_recommendations(self, consensus: Dict, todos: List[Dict]):
        """Present final recommendations to user."""
        
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold green]Kitchen Brigade Reconciliation Complete[/bold green]\n\n"
            f"• Conflicts identified: {len(self.trace_bundle['conflicts'])}\n"
            f"• Consensus resolutions: {len(consensus.get('consensus_resolutions', []))}\n"
            f"• Implementation TODOs: {len(todos)}\n"
            f"• Trace bundle saved\n\n"
            "[dim]Review .vscode/kitchen_brigade_todos.json for implementation phases[/dim]",
            border_style="green"
        ))


async def main():
    """Main entry point."""
    orchestrator = KitchenBrigadeOrchestrator()
    
    try:
        consensus = await orchestrator.run()
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Reconciliation interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]Reconciliation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
