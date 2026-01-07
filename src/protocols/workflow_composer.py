#!/usr/bin/env python3
"""
Kitchen Brigade Workflow Composer

Chains multiple protocols together with inter-stage feedback loops.
Enables complex workflows like: Round Table → Debate → Pipeline

Usage:
    python -m src.protocols.workflow_composer --workflow "round_table → debate → pipeline"
    python -m src.protocols.workflow_composer --workflow-file workflow.yaml
    python -m src.protocols.workflow_composer --interactive
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import yaml

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import the executor
from .kitchen_brigade_executor import KitchenBrigadeExecutor, PROTOCOLS_DIR

console = Console()

WORKFLOWS_DIR = Path(__file__).parent.parent.parent / "config" / "workflows"


@dataclass
class StageResult:
    """Result from a workflow stage."""
    stage_name: str
    protocol_id: str
    outputs: Dict[str, Any]
    trace: Dict[str, Any]
    needs_more_work: List[Dict] = field(default_factory=list)
    feedback_loops_used: int = 0
    user_injected_context: Optional[str] = None
    cross_reference_evidence: List[Dict] = field(default_factory=list)


@dataclass 
class WorkflowDefinition:
    """Definition of a multi-stage workflow."""
    name: str
    description: str
    stages: List[Dict[str, Any]]
    inter_stage_feedback: bool = True
    max_feedback_loops_per_stage: int = 3
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "WorkflowDefinition":
        """Load workflow from YAML file."""
        data = yaml.safe_load(yaml_path.read_text())
        return cls(
            name=data["name"],
            description=data["description"],
            stages=data["stages"],
            inter_stage_feedback=data.get("inter_stage_feedback", True),
            max_feedback_loops_per_stage=data.get("max_feedback_loops_per_stage", 3)
        )
    
    @classmethod
    def from_shorthand(cls, shorthand: str, inputs: Dict[str, Any] = None) -> "WorkflowDefinition":
        """
        Parse shorthand workflow notation.
        
        Examples:
            "round_table → debate → pipeline"
            "ROUNDTABLE_DISCUSSION -> ARCHITECTURE_RECONCILIATION"
        """
        # Normalize separators
        shorthand = shorthand.replace("->", "→").replace("=>", "→")
        stage_names = [s.strip() for s in shorthand.split("→")]
        
        # Map common names to protocol IDs
        name_map = {
            "round_table": "ROUNDTABLE_DISCUSSION",
            "roundtable": "ROUNDTABLE_DISCUSSION",
            "design_review": "ARCHITECTURE_RECONCILIATION",
            "debate": "DEBATE_PROTOCOL",
            "pipeline": "PIPELINE_PROTOCOL",
            "reconcile": "ARCHITECTURE_RECONCILIATION",
            "explore": "ROUNDTABLE_DISCUSSION",
            "decide": "DEBATE_PROTOCOL",
            "implement": "PIPELINE_PROTOCOL",
        }
        
        stages = []
        for i, name in enumerate(stage_names):
            protocol_id = name_map.get(name.lower(), name.upper())
            stages.append({
                "name": f"Stage {i+1}: {name}",
                "protocol_id": protocol_id,
                "inputs": inputs or {},
                "pass_outputs": True  # Pass outputs to next stage
            })
        
        return cls(
            name=f"Workflow: {' → '.join(stage_names)}",
            description=f"Multi-stage workflow with {len(stages)} protocols",
            stages=stages
        )


class WorkflowComposer:
    """
    Orchestrates multi-stage Kitchen Brigade workflows.
    
    Handles:
    - Sequential protocol execution
    - Output passing between stages
    - Inter-stage feedback loops
    - User injection points
    - Stage 2 cross-reference retrieval (4-layer: Qdrant, Neo4j, Textbooks, Code-Orchestrator)
    """
    
    def __init__(
        self, 
        workflow: WorkflowDefinition, 
        global_inputs: Dict[str, Any] = None,
        enable_cross_reference: bool = True,
        cross_reference_all_stages: bool = False
    ):
        self.workflow = workflow
        self.global_inputs = global_inputs or {}
        self.enable_cross_reference = enable_cross_reference
        self.cross_reference_all_stages = cross_reference_all_stages
        self.stage_results: List[StageResult] = []
        self.workflow_trace = {
            "workflow_name": workflow.name,
            "start_time": datetime.now().isoformat(),
            "stages": [],
            "inter_stage_feedback": []
        }
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the full workflow."""
        console.print(Panel.fit(
            f"[bold cyan]{self.workflow.name}[/bold cyan]\n"
            f"[dim]{self.workflow.description}[/dim]\n"
            f"[dim]Stages: {len(self.workflow.stages)} | Feedback: {'enabled' if self.workflow.inter_stage_feedback else 'disabled'}[/dim]",
            border_style="cyan"
        ))
        
        accumulated_outputs = {}
        
        for i, stage_def in enumerate(self.workflow.stages):
            stage_num = i + 1
            total_stages = len(self.workflow.stages)
            
            console.print(f"\n[bold magenta]{'═' * 60}[/bold magenta]")
            console.print(f"[bold magenta]  STAGE {stage_num}/{total_stages}: {stage_def['name']}[/bold magenta]")
            console.print(f"[bold magenta]{'═' * 60}[/bold magenta]\n")
            
            # Build inputs for this stage
            stage_inputs = {**self.global_inputs, **stage_def.get("inputs", {})}
            
            # Pass outputs from previous stages if configured
            if stage_def.get("pass_outputs", True) and accumulated_outputs:
                stage_inputs["_previous_stage_outputs"] = accumulated_outputs
                stage_inputs["_previous_stage_summary"] = self._summarize_outputs(accumulated_outputs)
            
            # Execute the stage
            result = await self._execute_stage(stage_def, stage_inputs)
            self.stage_results.append(result)
            
            # Accumulate outputs
            accumulated_outputs[stage_def["name"]] = result.outputs
            
            # Record in trace
            self.workflow_trace["stages"].append({
                "stage_num": stage_num,
                "name": stage_def["name"],
                "protocol_id": stage_def["protocol_id"],
                "start_time": result.trace.get("start_time"),
                "end_time": result.trace.get("end_time"),
                "feedback_loops": result.feedback_loops_used
            })
            
            # Inter-stage feedback check (not after last stage)
            if self.workflow.inter_stage_feedback and stage_num < total_stages:
                feedback_decision = await self._inter_stage_feedback(
                    result, 
                    stage_num, 
                    self.workflow.stages[i + 1]
                )
                
                self.workflow_trace["inter_stage_feedback"].append({
                    "after_stage": stage_num,
                    "decision": feedback_decision
                })
                
                if feedback_decision.get("stop_workflow"):
                    console.print("[red]Workflow stopped by user[/red]")
                    break
                
                if feedback_decision.get("repeat_stage"):
                    console.print(f"[yellow]Repeating stage {stage_num}...[/yellow]")
                    # Remove last result and re-execute
                    self.stage_results.pop()
                    del accumulated_outputs[stage_def["name"]]
                    
                    # Add user context if provided
                    if feedback_decision.get("additional_context"):
                        stage_inputs["_user_feedback"] = feedback_decision["additional_context"]
                    
                    result = await self._execute_stage(stage_def, stage_inputs)
                    self.stage_results.append(result)
                    accumulated_outputs[stage_def["name"]] = result.outputs
                
                elif feedback_decision.get("inject_context"):
                    # Add context for next stage
                    self.global_inputs["_injected_context"] = feedback_decision["inject_context"]
        
        self.workflow_trace["end_time"] = datetime.now().isoformat()
        self.workflow_trace["completed_stages"] = len(self.stage_results)
        
        # Final summary
        await self._present_workflow_summary()
        
        return {
            "workflow_name": self.workflow.name,
            "stage_results": [self._stage_result_to_dict(r) for r in self.stage_results],
            "final_outputs": accumulated_outputs,
            "trace": self.workflow_trace
        }
    
    async def _execute_stage(self, stage_def: Dict, inputs: Dict) -> StageResult:
        """Execute a single stage."""
        protocol_id = stage_def["protocol_id"]
        
        # Check if protocol exists
        protocol_file = PROTOCOLS_DIR / f"{protocol_id}.json"
        if not protocol_file.exists():
            console.print(f"[yellow]Warning: Protocol {protocol_id} not found, using ROUNDTABLE_DISCUSSION[/yellow]")
            protocol_id = "ROUNDTABLE_DISCUSSION"
        
        # Determine if cross-reference should run for this stage
        # By default, run cross-reference only for first stage to gather evidence
        # Subsequent stages can inherit evidence or run their own retrieval
        is_first_stage = len(self.stage_results) == 0
        
        # Cross-reference logic:
        # 1. If globally disabled, don't run
        # 2. If cross_reference_all_stages is True, run for every stage
        # 3. Otherwise, only run for first stage
        # 4. Stage can override with explicit enable_cross_reference setting
        stage_cross_ref_override = stage_def.get("enable_cross_reference")
        
        if not self.enable_cross_reference:
            enable_cross_ref = False
            run_cross_ref = False
        elif stage_cross_ref_override is not None:
            enable_cross_ref = stage_cross_ref_override
            run_cross_ref = stage_cross_ref_override
        elif self.cross_reference_all_stages:
            enable_cross_ref = True
            run_cross_ref = True
        else:
            enable_cross_ref = True
            run_cross_ref = is_first_stage
        
        executor = KitchenBrigadeExecutor(
            protocol_id=protocol_id,
            inputs=inputs,
            brigade_override=stage_def.get("brigade_override"),
            enable_cross_reference=enable_cross_ref
        )
        
        # If not first stage and not running fresh cross-reference, inherit evidence
        if not is_first_stage and not run_cross_ref and self.stage_results:
            inherited_evidence = self._get_inherited_cross_reference()
            if inherited_evidence:
                executor.cross_reference_evidence = inherited_evidence
                console.print(f"[dim]Passing {len(inherited_evidence)} cross-reference results to stage[/dim]")
        
        result = await executor.execute(
            max_feedback_loops=self.workflow.max_feedback_loops_per_stage,
            allow_feedback=True,
            run_cross_reference=run_cross_ref
        )
        
        return StageResult(
            stage_name=stage_def["name"],
            protocol_id=protocol_id,
            outputs=result["outputs"],
            trace=result["trace"],
            needs_more_work=result.get("needs_more_work", []),
            feedback_loops_used=result.get("feedback_loops_used", 0),
            cross_reference_evidence=getattr(executor, 'cross_reference_evidence', [])
        )
    
    def _get_inherited_cross_reference(self) -> List[Dict]:
        """Gather cross-reference evidence from all previous stages."""
        all_evidence = []
        for stage_result in self.stage_results:
            if hasattr(stage_result, 'cross_reference_evidence') and stage_result.cross_reference_evidence:
                all_evidence.extend(stage_result.cross_reference_evidence)
        return all_evidence
    
    async def _inter_stage_feedback(
        self, 
        result: StageResult, 
        stage_num: int,
        next_stage: Dict
    ) -> Dict[str, Any]:
        """Handle inter-stage feedback and user injection."""
        
        console.print("\n")
        console.print(Panel.fit(
            f"[bold yellow]Stage {stage_num} Complete[/bold yellow]\n"
            f"[dim]Review before proceeding to: {next_stage['name']}[/dim]",
            border_style="yellow"
        ))
        
        # Show summary of stage results
        console.print("\n[bold]Stage Summary:[/bold]")
        console.print(f"  Protocol: [cyan]{result.protocol_id}[/cyan]")
        console.print(f"  Feedback loops used: [cyan]{result.feedback_loops_used}[/cyan]")
        
        if result.needs_more_work:
            console.print(f"  [yellow]⚠ {len(result.needs_more_work)} items flagged for attention[/yellow]")
        else:
            console.print("  [green]✓ No outstanding issues[/green]")
        
        # Options
        console.print("\n[bold]Options:[/bold]")
        console.print("  1. [cyan]continue[/cyan] - Proceed to next stage")
        console.print("  2. [cyan]inject[/cyan] - Add context for next stage")
        console.print("  3. [cyan]repeat[/cyan] - Repeat this stage with more feedback")
        console.print("  4. [cyan]review[/cyan] - Show detailed outputs")
        console.print("  5. [cyan]stop[/cyan] - Stop workflow here")
        
        choice = Prompt.ask(
            "\nHow to proceed?",
            choices=["continue", "inject", "repeat", "review", "stop", "1", "2", "3", "4", "5"],
            default="continue"
        )
        
        choice_map = {"1": "continue", "2": "inject", "3": "repeat", "4": "review", "5": "stop"}
        choice = choice_map.get(choice, choice)
        
        if choice == "stop":
            return {"stop_workflow": True}
        
        if choice == "review":
            # Show detailed outputs
            console.print("\n[bold]Detailed Outputs:[/bold]")
            console.print(json.dumps(result.outputs, indent=2, default=str)[:2000])
            console.print("[dim]...truncated[/dim]" if len(json.dumps(result.outputs)) > 2000 else "")
            
            # Re-prompt
            return await self._inter_stage_feedback(result, stage_num, next_stage)
        
        if choice == "repeat":
            additional = Prompt.ask(
                "Add guidance for repeat (or empty)",
                default=""
            )
            return {
                "repeat_stage": True,
                "additional_context": additional if additional else None
            }
        
        if choice == "inject":
            context = Prompt.ask(
                "Enter context to inject for next stage"
            )
            return {
                "inject_context": context
            }
        
        return {}  # Continue normally
    
    def _summarize_outputs(self, outputs: Dict) -> str:
        """Create a text summary of accumulated outputs for context."""
        summary_parts = []
        
        for stage_name, stage_outputs in outputs.items():
            summary_parts.append(f"## {stage_name}")
            
            # Extract key findings from each round
            for round_key, round_output in stage_outputs.items():
                if not isinstance(round_output, dict):
                    continue
                
                for role, result in round_output.items():
                    if isinstance(result, dict) and "parsed" in result:
                        parsed = result["parsed"]
                        if parsed:
                            # Extract key elements
                            if "findings" in parsed:
                                summary_parts.append(f"- {role} findings: {parsed['findings'][:3]}")
                            if "final_position" in parsed:
                                summary_parts.append(f"- {role} position: {parsed['final_position'][:200]}...")
                            if "consensus_achieved" in parsed:
                                summary_parts.append(f"- Consensus: {parsed['consensus_achieved']}")
        
        return "\n".join(summary_parts)[:4000]  # Limit size
    
    def _stage_result_to_dict(self, result: StageResult) -> Dict:
        """Convert StageResult to dict for serialization."""
        return {
            "stage_name": result.stage_name,
            "protocol_id": result.protocol_id,
            "outputs": result.outputs,
            "needs_more_work": result.needs_more_work,
            "feedback_loops_used": result.feedback_loops_used
        }
    
    async def _present_workflow_summary(self):
        """Present final workflow summary."""
        console.print("\n")
        console.print(Panel.fit(
            f"[bold green]Workflow Complete: {self.workflow.name}[/bold green]",
            border_style="green"
        ))
        
        table = Table(title="Stage Summary")
        table.add_column("Stage", style="cyan")
        table.add_column("Protocol", style="green")
        table.add_column("Feedback Loops", style="yellow")
        table.add_column("Issues", style="red")
        
        for result in self.stage_results:
            table.add_row(
                result.stage_name,
                result.protocol_id,
                str(result.feedback_loops_used),
                str(len(result.needs_more_work)) if result.needs_more_work else "✓"
            )
        
        console.print(table)


async def prompt_workflow_config() -> Dict[str, Any]:
    """Interactive workflow configuration."""
    console.print(Panel.fit(
        "[bold cyan]Workflow Composer[/bold cyan]\n"
        "[dim]Build a multi-stage Kitchen Brigade workflow[/dim]",
        border_style="cyan"
    ))
    
    # Pre-built workflows
    console.print("\n[bold]Pre-built Workflows:[/bold]")
    console.print("  1. [cyan]explore_decide_implement[/cyan] - Round Table → Debate → Pipeline")
    console.print("  2. [cyan]review_reconcile[/cyan] - Design Review → Round Table consensus")
    console.print("  3. [cyan]custom[/cyan] - Build your own workflow")
    
    choice = Prompt.ask(
        "\nSelect workflow",
        choices=["1", "2", "3", "explore_decide_implement", "review_reconcile", "custom"],
        default="1"
    )
    
    if choice in ["1", "explore_decide_implement"]:
        workflow_str = "round_table → debate → pipeline"
    elif choice in ["2", "review_reconcile"]:
        workflow_str = "design_review → round_table"
    else:
        # Custom workflow
        console.print("\n[bold]Build Custom Workflow[/bold]")
        console.print("[dim]Available protocols: round_table, design_review, debate, pipeline[/dim]")
        console.print("[dim]Use → or -> to chain: round_table → debate → pipeline[/dim]")
        workflow_str = Prompt.ask("Enter workflow")
    
    # Get topic/inputs
    topic = Prompt.ask("\nWhat topic/task is this workflow for?")
    
    # Feedback settings
    enable_feedback = Confirm.ask("Enable inter-stage feedback prompts?", default=True)
    
    return {
        "workflow_str": workflow_str,
        "topic": topic,
        "enable_feedback": enable_feedback
    }


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kitchen Brigade Workflow Composer")
    parser.add_argument("--workflow", help="Workflow shorthand (e.g., 'round_table → debate')")
    parser.add_argument("--workflow-file", help="Path to workflow YAML definition")
    parser.add_argument("--input", action="append", help="Input key=value pairs")
    parser.add_argument("--topic", help="Main topic for the workflow")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive configuration")
    parser.add_argument("--no-feedback", action="store_true", help="Disable inter-stage feedback")
    parser.add_argument("--no-cross-reference", action="store_true", 
                        help="Disable Stage 2 cross-reference retrieval")
    parser.add_argument("--cross-reference-all-stages", action="store_true",
                        help="Run cross-reference at every stage (not just first)")
    
    args = parser.parse_args()
    
    # Parse inputs
    inputs = {}
    if args.input:
        for inp in args.input:
            key, value = inp.split("=", 1)
            try:
                inputs[key] = json.loads(value)
            except json.JSONDecodeError:
                inputs[key] = value
    
    if args.topic:
        inputs["topic"] = args.topic
    
    # Determine workflow
    if args.interactive:
        config = await prompt_workflow_config()
        workflow = WorkflowDefinition.from_shorthand(config["workflow_str"], inputs)
        workflow.inter_stage_feedback = config["enable_feedback"]
        inputs["topic"] = config["topic"]
    elif args.workflow_file:
        workflow = WorkflowDefinition.from_yaml(Path(args.workflow_file))
    elif args.workflow:
        workflow = WorkflowDefinition.from_shorthand(args.workflow, inputs)
    else:
        console.print("[red]Must specify --workflow, --workflow-file, or --interactive[/red]")
        return
    
    if args.no_feedback:
        workflow.inter_stage_feedback = False
    
    # Execute with cross-reference settings
    composer = WorkflowComposer(
        workflow, 
        inputs,
        enable_cross_reference=not args.no_cross_reference,
        cross_reference_all_stages=args.cross_reference_all_stages
    )
    result = await composer.execute()
    
    # Save trace
    trace_file = Path(f"workflow_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    trace_file.write_text(json.dumps(result, indent=2, default=str))
    console.print(f"\n✓ Workflow trace saved to {trace_file}")
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
