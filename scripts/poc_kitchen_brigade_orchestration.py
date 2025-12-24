#!/usr/bin/env python3
"""
POC: Kitchen Brigade Inter-AI Orchestration
============================================

This script validates the hypothesis that multiple AI agents can collaborate
on document cross-referencing tasks following the Kitchen Brigade architecture.

ROLE MAPPING:
- THIS SCRIPT (Copilot) = Executive Chef / Requester - Orchestrates the workflow
- Claude (via llm-gateway) = Internal LLM Worker #1 - Executes tasks
- DeepSeek (via llm-gateway) = Internal LLM Worker #2 - Executes tasks

TASK: Steps 1-3 from the user's requirements
- Step 1: Document Hierarchy Review
- Step 2: Guideline Cross-Reference with textbooks
- Step 3: Conflict Identification & Resolution

Per INTER_AI_ORCHESTRATION.md:
- ALL communication is MEDIATED (no direct AI-to-AI communication)
- ai-agents is the moderator that facilitates conversations between AI participants

Created: 2024-12-24
"""

import asyncio
import httpx
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

LLM_GATEWAY_URL = "http://localhost:8080"
TEXTBOOKS_PATH = Path("/Users/kevintoles/POC/llm-document-enhancer/workflows/pdf_to_json/output/textbooks_json")
GUIDELINES_PATH = Path("/Users/kevintoles/POC/textbooks/Guidelines/GUIDELINES_AI_Engineering_Building_Applications_AIML_LLM_ENHANCED.md")
TAXONOMY_PATH = Path("/Users/kevintoles/POC/textbooks/Taxonomies/AI-ML_taxonomy_20251128.json")
OUTPUT_DIR = Path("/Users/kevintoles/POC/ai-agents/data/kitchen_brigade_poc")

# Per INTER_AI_ORCHESTRATION.md - these are our participant models
WORKER_MODELS = {
    "claude": "claude-opus-4-5-20251101",
    "deepseek": "deepseek-reasoner",
}

# Document Priority Hierarchy (from user requirements)
DOCUMENT_PRIORITY = [
    "GUIDELINES_AI_Engineering_Building_Applications_AIML_LLM_ENHANCED.md",
    "AI_CODING_PLATFORM_ARCHITECTURE.md",
    "llm-gateway/docs/ARCHITECTURE.md",
    "AI-ML_taxonomy_20251128.json",
    "CODING_PATTERNS_ANALYSIS.md",
]


# ============================================================================
# DATA CLASSES (per INTER_AI_ORCHESTRATION.md)
# ============================================================================

class ParticipantType(Enum):
    LLM = "llm"
    ORCHESTRATOR = "orchestrator"


class ConversationStatus(Enum):
    IN_PROGRESS = "in_progress"
    CONSENSUS = "consensus"
    DEADLOCK = "deadlock"
    COMPLETE = "complete"
    TIMEOUT = "timeout"


@dataclass
class ConversationMessage:
    """Single message in an inter-AI conversation."""
    message_id: str
    conversation_id: str
    participant_id: str
    participant_type: ParticipantType
    timestamp: str
    role: str  # "user" | "assistant" | "orchestrator"
    content: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Conversation:
    """Complete inter-AI conversation state."""
    conversation_id: str
    task: str
    participants: list[str]
    messages: list[ConversationMessage] = field(default_factory=list)
    current_turn: str = ""
    status: ConversationStatus = ConversationStatus.IN_PROGRESS
    context: dict = field(default_factory=dict)
    consensus_data: dict = field(default_factory=dict)


# ============================================================================
# ORCHESTRATOR CLASS
# ============================================================================

class KitchenBrigadeOrchestrator:
    """
    Executive Chef / Orchestrator that mediates all communication.
    
    Per INTER_AI_ORCHESTRATION.md:
    "ai-agents acts as the moderator/orchestrator that facilitates 
    conversations between AI participants. No AI talks directly to another - 
    all communication is mediated."
    """
    
    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.conversation: Optional[Conversation] = None
        self.message_counter = 0
        
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=120.0)
        return self
        
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
    
    def _next_message_id(self) -> str:
        self.message_counter += 1
        return f"msg_{self.message_counter:04d}"
    
    def _log_message(self, participant: str, role: str, content: str, 
                     latency_ms: int = 0, tokens: int = 0) -> ConversationMessage:
        """Log a message to the conversation transcript."""
        msg = ConversationMessage(
            message_id=self._next_message_id(),
            conversation_id=self.conversation.conversation_id if self.conversation else "init",
            participant_id=participant,
            participant_type=ParticipantType.ORCHESTRATOR if participant == "orchestrator" else ParticipantType.LLM,
            timestamp=datetime.now().isoformat(),
            role=role,
            content=content[:500] + "..." if len(content) > 500 else content,
            tokens_used=tokens if tokens > 0 else None,
            latency_ms=latency_ms,
        )
        if self.conversation:
            self.conversation.messages.append(msg)
        return msg
    
    async def call_worker(self, worker_id: str, prompt: str, system_prompt: str = "") -> dict:
        """
        Send a task to a worker AI via llm-gateway.
        
        This is the MEDIATED communication pattern from INTER_AI_ORCHESTRATION.md:
        - Orchestrator → llm-gateway → Worker LLM
        - Worker LLM → llm-gateway → Orchestrator
        """
        model = WORKER_MODELS.get(worker_id)
        if not model:
            raise ValueError(f"Unknown worker: {worker_id}")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        start_time = datetime.now()
        
        # Log outbound message
        self._log_message("orchestrator", "user", f"[TO {worker_id}] {prompt[:200]}...")
        print(f"\n{'='*60}")
        print(f"📤 ORCHESTRATOR → {worker_id.upper()}")
        print(f"{'='*60}")
        print(f"Model: {model}")
        print(f"Prompt: {prompt[:300]}...")
        
        try:
            response = await self.client.post(
                f"{LLM_GATEWAY_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 2000,
                    "temperature": 0.3,
                },
            )
            
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("total_tokens", 0)
                
                # Log inbound response
                self._log_message(worker_id, "assistant", content, latency_ms, tokens)
                
                print(f"\n{'='*60}")
                print(f"📥 {worker_id.upper()} → ORCHESTRATOR")
                print(f"{'='*60}")
                print(f"Latency: {latency_ms}ms | Tokens: {tokens}")
                print(f"Response: {content[:500]}...")
                
                return {
                    "success": True,
                    "content": content,
                    "tokens": tokens,
                    "latency_ms": latency_ms,
                }
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                self._log_message(worker_id, "error", error_msg, latency_ms)
                print(f"❌ ERROR from {worker_id}: {error_msg}")
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = str(e)
            self._log_message(worker_id, "error", error_msg)
            print(f"❌ EXCEPTION calling {worker_id}: {error_msg}")
            return {"success": False, "error": error_msg}
    
    async def run_step_1_document_hierarchy_review(self) -> dict:
        """
        STEP 1: Document Hierarchy Review
        
        Ask each worker to review the document hierarchy and identify
        what each document contributes to the architecture.
        """
        print("\n" + "="*80)
        print("🔷 STEP 1: DOCUMENT HIERARCHY REVIEW")
        print("="*80)
        
        hierarchy_context = f"""
DOCUMENT PRIORITY HIERARCHY (highest to lowest):
1. {DOCUMENT_PRIORITY[0]} - Master guidelines for AI/ML engineering
2. {DOCUMENT_PRIORITY[1]} - Kitchen Brigade architecture, service roles
3. {DOCUMENT_PRIORITY[2]} - Gateway routing, tool proxy pattern
4. {DOCUMENT_PRIORITY[3]} - Taxonomy of AI/ML concepts from 201 textbooks
5. {DOCUMENT_PRIORITY[4]} - Anti-patterns and coding standards

Your task: Analyze what each document's PRIMARY responsibility is
in the architecture. What decisions should be resolved by each document?
"""
        
        system_prompt = """You are an expert software architect reviewing a document hierarchy.
Your role is to identify the PRIMARY responsibility of each document in the hierarchy.
Be concise and focus on what decisions each document should govern."""
        
        # Call Claude
        claude_result = await self.call_worker(
            "claude",
            f"{hierarchy_context}\n\nProvide a structured analysis of each document's responsibility.",
            system_prompt
        )
        
        # Call DeepSeek
        deepseek_result = await self.call_worker(
            "deepseek", 
            f"{hierarchy_context}\n\nProvide a structured analysis of each document's responsibility.",
            system_prompt
        )
        
        return {
            "step": 1,
            "task": "Document Hierarchy Review",
            "claude": claude_result,
            "deepseek": deepseek_result,
        }
    
    async def run_step_2_guideline_crossref(self) -> dict:
        """
        STEP 2: Guideline Cross-Reference with Textbooks
        
        Ask workers to cross-reference a specific guideline against
        the textbook taxonomy.
        """
        print("\n" + "="*80)
        print("🔷 STEP 2: GUIDELINE CROSS-REFERENCE")
        print("="*80)
        
        # Load taxonomy for context
        taxonomy = {}
        if TAXONOMY_PATH.exists():
            with open(TAXONOMY_PATH) as f:
                taxonomy = json.load(f)
        
        # Get sample books from taxonomy
        sample_books = list(taxonomy.get("textbooks", {}).keys())[:10]
        
        crossref_context = f"""
TASK: Cross-reference the Kitchen Brigade pattern against textbook sources.

GUIDELINE CONCEPT: "Kitchen Brigade Architecture"
- Router (llm-gateway): Routes requests, manages sessions, does NOT make content decisions
- Expeditor (ai-agents): Orchestrates workflow, coordinates services
- Cookbook (semantic-search): DUMB retrieval only, returns ALL matches
- Sous Chef (Code-Orchestrator): SMART - hosts ML models, generates content
- Auditor (audit-service): Validates code, detects anti-patterns

SAMPLE TEXTBOOKS FROM TAXONOMY:
{json.dumps(sample_books, indent=2)}

QUESTION: Which textbooks likely support or contradict this pattern?
Provide specific textbook names and explain the connection.
"""
        
        system_prompt = """You are a technical analyst cross-referencing architecture patterns.
Identify which textbooks from the taxonomy would support or contradict the pattern.
Be specific about book names and reasoning."""
        
        claude_result = await self.call_worker(
            "claude",
            crossref_context,
            system_prompt
        )
        
        deepseek_result = await self.call_worker(
            "deepseek",
            crossref_context,
            system_prompt
        )
        
        return {
            "step": 2,
            "task": "Guideline Cross-Reference",
            "claude": claude_result,
            "deepseek": deepseek_result,
        }
    
    async def run_step_3_conflict_resolution(self, step1_results: dict, step2_results: dict) -> dict:
        """
        STEP 3: Conflict Identification & Resolution
        
        Ask workers to identify conflicts between their analyses
        and propose resolutions.
        """
        print("\n" + "="*80)
        print("🔷 STEP 3: CONFLICT RESOLUTION")
        print("="*80)
        
        # Summarize prior results for context
        claude_step1 = step1_results.get("claude", {}).get("content", "N/A")[:500]
        deepseek_step1 = step1_results.get("deepseek", {}).get("content", "N/A")[:500]
        claude_step2 = step2_results.get("claude", {}).get("content", "N/A")[:500]
        deepseek_step2 = step2_results.get("deepseek", {}).get("content", "N/A")[:500]
        
        conflict_context = f"""
TASK: Identify conflicts and propose resolutions.

STEP 1 ANALYSES:
- Claude's analysis: {claude_step1}
- DeepSeek's analysis: {deepseek_step1}

STEP 2 ANALYSES:
- Claude's cross-reference: {claude_step2}
- DeepSeek's cross-reference: {deepseek_step2}

QUESTIONS:
1. Do Claude and DeepSeek agree on the key points?
2. Where do they disagree?
3. How should conflicts be resolved using the document hierarchy?
"""
        
        system_prompt = """You are a conflict resolution specialist.
Compare the two AI analyses and identify:
1. Points of agreement
2. Points of disagreement  
3. Proposed resolution using the document hierarchy as tiebreaker"""
        
        # Ask Claude to analyze the conflict
        claude_result = await self.call_worker(
            "claude",
            conflict_context,
            system_prompt
        )
        
        # Ask DeepSeek to analyze the conflict
        deepseek_result = await self.call_worker(
            "deepseek",
            conflict_context,
            system_prompt
        )
        
        return {
            "step": 3,
            "task": "Conflict Resolution",
            "claude": claude_result,
            "deepseek": deepseek_result,
        }
    
    async def run_full_orchestration(self) -> dict:
        """
        Execute the full Kitchen Brigade orchestration workflow.
        
        This is the POC to validate:
        1. Inter-LLM communication plumbing works
        2. Kitchen Brigade orchestration functions
        3. AI agents can perform document cross-referencing
        """
        # Initialize conversation
        self.conversation = Conversation(
            conversation_id=f"kb_poc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task="Document Cross-Reference via Kitchen Brigade",
            participants=["orchestrator", "claude", "deepseek"],
        )
        
        print("\n" + "🍽️"*20)
        print("KITCHEN BRIGADE ORCHESTRATION POC")
        print("🍽️"*20)
        print(f"Conversation ID: {self.conversation.conversation_id}")
        print(f"Participants: {self.conversation.participants}")
        print(f"Task: {self.conversation.task}")
        
        # Execute Steps 1-3
        step1_results = await self.run_step_1_document_hierarchy_review()
        step2_results = await self.run_step_2_guideline_crossref()
        step3_results = await self.run_step_3_conflict_resolution(step1_results, step2_results)
        
        # Mark complete
        self.conversation.status = ConversationStatus.COMPLETE
        
        # Compile final results
        results = {
            "conversation_id": self.conversation.conversation_id,
            "status": self.conversation.status.value,
            "participants": self.conversation.participants,
            "task": self.conversation.task,
            "timestamp": datetime.now().isoformat(),
            "steps": {
                "step_1": step1_results,
                "step_2": step2_results,
                "step_3": step3_results,
            },
            "message_count": len(self.conversation.messages),
            "transcript": [asdict(m) for m in self.conversation.messages],
        }
        
        # Save results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / f"{self.conversation.conversation_id}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n" + "="*80)
        print("✅ ORCHESTRATION COMPLETE")
        print("="*80)
        print(f"Messages exchanged: {len(self.conversation.messages)}")
        print(f"Results saved to: {output_file}")
        
        return results


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Entry point for the Kitchen Brigade POC."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║     KITCHEN BRIGADE INTER-AI ORCHESTRATION POC                                ║
║     ─────────────────────────────────────────                                ║
║     Role: Executive Chef (Orchestrator/Requester)                            ║
║     Workers: Claude + DeepSeek (via llm-gateway)                             ║
║     Task: Steps 1-3 Document Cross-Reference                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    async with KitchenBrigadeOrchestrator() as orchestrator:
        results = await orchestrator.run_full_orchestration()
        
        # Print summary
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        
        for step_key, step_data in results["steps"].items():
            step_num = step_data.get("step", "?")
            task = step_data.get("task", "Unknown")
            claude_ok = step_data.get("claude", {}).get("success", False)
            deepseek_ok = step_data.get("deepseek", {}).get("success", False)
            
            claude_status = "✅" if claude_ok else "❌"
            deepseek_status = "✅" if deepseek_ok else "❌"
            
            print(f"Step {step_num}: {task}")
            print(f"  Claude: {claude_status} | DeepSeek: {deepseek_status}")
        
        # Overall status
        all_success = all(
            step_data.get("claude", {}).get("success", False) and
            step_data.get("deepseek", {}).get("success", False)
            for step_data in results["steps"].values()
        )
        
        if all_success:
            print("\n🎉 POC VALIDATED: Inter-AI orchestration plumbing works!")
        else:
            print("\n⚠️ POC PARTIAL: Some steps failed - check logs")
        
        return results


if __name__ == "__main__":
    asyncio.run(main())
