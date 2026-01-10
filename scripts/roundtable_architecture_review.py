#!/usr/bin/env python3
"""
Round Table Architecture Review

A collaborative multi-LLM architecture review where 4 LLMs act as Principal Engineers
in a round-table discussion format. Each LLM can:
- Present their analysis
- Challenge other participants
- Request additional research from the knowledge base
- Build on or refine others' recommendations

Features:
- Multiple discussion rounds until consensus or max rounds
- Dynamic research requests via semantic search + graph RAG
- All participants are equal peers (no hierarchy)
- Chicago-style citations for references

WBS: Multi-LLM Architecture Validation - Round Table Pattern
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import httpx

# Configuration
LLM_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "http://localhost:8080")
SEMANTIC_SEARCH_URL = os.getenv("SEMANTIC_SEARCH_URL", "http://localhost:8081")
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8085")

# Output configuration
OUTPUT_DIR = Path("/Users/kevintoles/POC/Platform Documentation/architecture_roundtable")
TIMESTAMP = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# Document paths
DOCS_BASE = Path("/Users/kevintoles/POC/Platform Documentation")
ARCHITECTURE_DOC = DOCS_BASE / "ARCHITECTURE_DECISION_RECORD.md"
CODE_QUALITY_DOC = DOCS_BASE / "code_quality_assessment" / "CODE_QUALITY_ASSESSMENT.md"
MULTI_LLM_DOC = DOCS_BASE / "multi_llm_assessment" / "MULTI_LLM_ASSESSMENT.md"
GEMINI_DOC = DOCS_BASE / "gemini_review" / "GEMINI_ASSESSMENT.md"

# Round Table Configuration
MAX_DISCUSSION_ROUNDS = 4
MIN_DISCUSSION_ROUNDS = 2

# Participant Configuration - All are Principal Engineers
PARTICIPANTS = {
    "qwen": {
        "name": "Qwen (Principal Engineer - Distributed Systems)",
        "model": "qwen3-8b",
        "provider": "local",
        "expertise": "Distributed systems, service mesh, container orchestration",
        "context_length": 32768
    },
    "deepseek": {
        "name": "DeepSeek (Principal Engineer - Platform Architecture)", 
        "model": "deepseek-r1-7b",
        "provider": "local",
        "expertise": "Platform architecture, API design, system integration",
        "context_length": 32768
    },
    "gpt": {
        "name": "GPT (Principal Engineer - Security & Reliability)",
        "model": "gpt-5.2",
        "provider": "openai",
        "expertise": "Security architecture, reliability engineering, risk assessment",
        "context_length": 128000
    },
    "claude": {
        "name": "Claude (Principal Engineer - Developer Experience)",
        "model": "claude-opus-4-20250514",
        "provider": "anthropic",
        "expertise": "Developer experience, observability, documentation",
        "context_length": 200000
    }
}


class SemanticSearchClient:
    """Client for semantic search + graph RAG queries."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.base_url = SEMANTIC_SEARCH_URL
    
    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """Execute hybrid search (vector + graph)."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/search/hybrid",
                json={
                    "query": query,
                    "limit": limit,
                    "include_graph": True
                }
            )
            if response.status_code == 200:
                return response.json().get("results", [])
            return []
        except Exception as e:
            print(f"   ⚠️ Search error: {e}")
            return []
    
    async def close(self):
        await self.client.aclose()


class LLMClient:
    """Unified client for LLM calls through gateway and inference service."""
    
    def __init__(self):
        self.gateway_url = LLM_GATEWAY_URL
        self.inference_url = INFERENCE_SERVICE_URL
        self.client = httpx.AsyncClient(timeout=180.0)
    
    async def complete(
        self,
        model: str,
        messages: list[dict],
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> dict:
        """Send completion request."""
        try:
            # Route local models to inference-service
            if provider == "local":
                url = f"{self.inference_url}/v1/chat/completions"
            else:
                url = f"{self.gateway_url}/v1/chat/completions"
            
            response = await self.client.post(
                url,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text[:500]}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def close(self):
        await self.client.aclose()


def load_document(path: Path) -> str:
    """Load document content."""
    if path.exists():
        return path.read_text()
    return f"[Document not found: {path}]"


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max characters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated for context limits ...]"


def extract_research_requests(response: str) -> list[str]:
    """Extract research requests from LLM response."""
    # Look for patterns like:
    # RESEARCH_REQUEST: <topic>
    # [NEED_RESEARCH: <topic>]
    # I need more information about <topic>
    patterns = [
        r'RESEARCH_REQUEST:\s*(.+?)(?:\n|$)',
        r'\[NEED_RESEARCH:\s*(.+?)\]',
        r'(?:need|want|require)(?:s)?\s+(?:more\s+)?(?:information|context|research|data)\s+(?:about|on|regarding)\s+["\']?([^"\'.\n]+)["\']?',
    ]
    
    requests = []
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        requests.extend(matches)
    
    return list(set(requests))  # Deduplicate


def check_consensus(responses: list[dict]) -> tuple[bool, list[str]]:
    """
    Check if participants have reached consensus.
    Returns (has_consensus, disagreement_points)
    """
    # Look for explicit consensus/disagreement markers
    consensus_phrases = ["i agree", "consensus reached", "we all agree", "aligns with"]
    disagreement_phrases = ["i disagree", "however", "but i think", "on the contrary", "challenge this"]
    
    disagreements = []
    agreement_count = 0
    
    for resp in responses:
        content_lower = resp.get("content", "").lower()
        
        if any(phrase in content_lower for phrase in consensus_phrases):
            agreement_count += 1
        
        for phrase in disagreement_phrases:
            if phrase in content_lower:
                # Extract the disagreement context
                idx = content_lower.find(phrase)
                context = resp["content"][max(0, idx-50):idx+200]
                disagreements.append(f"{resp['name']}: {context[:150]}...")
    
    # Consensus if majority agrees and few explicit disagreements
    has_consensus = agreement_count >= len(responses) - 1 and len(disagreements) <= 1
    
    return has_consensus, disagreements


class RoundTableDiscussion:
    """Orchestrates the round table architecture review."""
    
    def __init__(self, search_client: SemanticSearchClient, llm_client: LLMClient):
        self.search = search_client
        self.llm = llm_client
        self.discussion_history: list[dict] = []
        self.research_cache: dict[str, list[dict]] = {}
        self.round_number = 0
    
    async def conduct_research(self, topics: list[str]) -> dict:
        """Conduct research on requested topics."""
        results = {}
        for topic in topics:
            if topic in self.research_cache:
                results[topic] = self.research_cache[topic]
            else:
                print(f"      📚 Researching: {topic}")
                search_results = await self.search.search(topic, limit=3)
                self.research_cache[topic] = search_results
                results[topic] = search_results
        return results
    
    def format_research_context(self, research: dict) -> str:
        """Format research results for context."""
        if not research:
            return ""
        
        parts = ["\n=== ADDITIONAL RESEARCH FINDINGS ==="]
        for topic, results in research.items():
            parts.append(f"\n--- Research on: {topic} ---")
            for r in results[:3]:
                source = r.get("metadata", {}).get("source", "Unknown")
                content = r.get("content", "")[:500]
                parts.append(f"Source: {source}\n{content}\n")
        
        return "\n".join(parts)
    
    def build_system_prompt(self, participant_key: str) -> str:
        """Build system prompt for a participant."""
        p = PARTICIPANTS[participant_key]
        
        return f"""You are {p['name']}, participating in a Round Table Architecture Review.

Your expertise: {p['expertise']}

DISCUSSION RULES:
1. You are an EQUAL PEER with other Principal Engineers - challenge ideas respectfully
2. Reference specific sections of documents when making points
3. Build on others' insights or respectfully disagree with reasoning
4. If you need more information, use: RESEARCH_REQUEST: <specific topic>
5. When you agree with a consensus forming, explicitly state it
6. Focus on actionable, specific recommendations

FORMAT YOUR RESPONSE:
## My Assessment
<your analysis>

## Response to Other Engineers
<agreements, challenges, or refinements to others' points>

## Key Recommendations
<prioritized list>

## Research Requests (if any)
RESEARCH_REQUEST: <topic> (only if you need more context)

## Consensus Status
<do you agree with emerging consensus? what's still unresolved?>"""

    def build_user_prompt(
        self, 
        participant_key: str, 
        context: dict, 
        is_initial: bool,
        additional_research: str = ""
    ) -> str:
        """Build user prompt with context and discussion history."""
        p = PARTICIPANTS[participant_key]
        is_local = p["provider"] == "local"
        
        # Adjust sizes based on model context
        doc_size = 3000 if is_local else 5000
        history_size = 2000 if is_local else 4000
        
        parts = []
        
        if is_initial:
            parts.append("=== ROUND TABLE ARCHITECTURE REVIEW ===")
            parts.append("You are reviewing the AI Platform architecture. Initial documents:\n")
            parts.append("--- ARCHITECTURE DECISION RECORD (under review) ---")
            parts.append(truncate_text(context["architecture_doc"], doc_size))
            parts.append("\n--- CODE QUALITY ASSESSMENT ---")
            parts.append(truncate_text(context["code_quality_doc"], doc_size // 2))
            parts.append("\n--- MULTI-LLM ASSESSMENT ---")
            parts.append(truncate_text(context["multi_llm_doc"], doc_size // 2))
            parts.append("\n--- GEMINI ASSESSMENT ---")
            parts.append(truncate_text(context["gemini_doc"], doc_size // 2))
            parts.append("\n--- INITIAL RESEARCH CONTEXT ---")
            parts.append(truncate_text(context.get("initial_research", ""), 2000))
            parts.append("\n\nProvide your initial assessment. Other engineers will respond after you.")
        else:
            parts.append(f"=== ROUND {self.round_number} DISCUSSION ===\n")
            parts.append("Previous discussion:")
            
            # Include recent discussion history
            recent_history = self.discussion_history[-8:]  # Last 8 responses
            for entry in recent_history:
                parts.append(f"\n--- {entry['name']} (Round {entry['round']}) ---")
                parts.append(truncate_text(entry['content'], history_size // len(recent_history)))
            
            if additional_research:
                parts.append(additional_research)
            
            parts.append("\n\nRespond to your fellow engineers. Build on agreements, address disagreements.")
        
        return "\n".join(parts)
    
    async def get_participant_response(
        self, 
        participant_key: str, 
        context: dict,
        is_initial: bool = False,
        additional_research: str = ""
    ) -> dict:
        """Get response from a participant."""
        p = PARTICIPANTS[participant_key]
        
        print(f"\n{'='*60}")
        print(f"🎙️ {p['name']}")
        print(f"   Model: {p['model']} via {p['provider']}")
        print(f"{'='*60}")
        
        system_prompt = self.build_system_prompt(participant_key)
        user_prompt = self.build_user_prompt(participant_key, context, is_initial, additional_research)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.llm.complete(
            model=p["model"],
            messages=messages,
            provider=p["provider"],
            temperature=0.7,
            max_tokens=4096
        )
        
        if "error" in response:
            print(f"   ❌ Error: {response['error']}")
            content = f"[Error: {response['error']}]"
        else:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"   ✅ Response received ({len(content)} chars)")
        
        result = {
            "participant": participant_key,
            "name": p["name"],
            "model": p["model"],
            "provider": p["provider"],
            "content": content,
            "round": self.round_number,
            "timestamp": datetime.now().isoformat()
        }
        
        self.discussion_history.append(result)
        return result
    
    async def run_discussion_round(self, context: dict, is_initial: bool = False) -> list[dict]:
        """Run one round of discussion with all participants."""
        self.round_number += 1
        round_responses = []
        
        print(f"\n{'━'*70}")
        print(f"ROUND {self.round_number}" + (" (Initial Assessments)" if is_initial else " (Discussion)"))
        print(f"{'━'*70}")
        
        # Collect all research requests from previous round
        if not is_initial and self.discussion_history:
            all_research_requests = []
            for entry in self.discussion_history[-4:]:  # Last round's responses
                requests = extract_research_requests(entry.get("content", ""))
                all_research_requests.extend(requests)
            
            # Conduct research if requested
            additional_research = ""
            if all_research_requests:
                print(f"\n   📚 Processing {len(all_research_requests)} research requests...")
                research_results = await self.conduct_research(all_research_requests)
                additional_research = self.format_research_context(research_results)
        else:
            additional_research = ""
        
        # Get responses from all participants
        for participant_key in PARTICIPANTS.keys():
            response = await self.get_participant_response(
                participant_key, 
                context, 
                is_initial=is_initial,
                additional_research=additional_research
            )
            round_responses.append(response)
        
        return round_responses
    
    async def run_full_discussion(self, context: dict) -> dict:
        """Run the full round table discussion."""
        
        # Round 1: Initial assessments
        initial_responses = await self.run_discussion_round(context, is_initial=True)
        
        # Continue discussion rounds
        for round_num in range(2, MAX_DISCUSSION_ROUNDS + 1):
            round_responses = await self.run_discussion_round(context, is_initial=False)
            
            # Check for consensus after minimum rounds
            if round_num >= MIN_DISCUSSION_ROUNDS:
                has_consensus, disagreements = check_consensus(round_responses)
                
                if has_consensus:
                    print(f"\n   ✅ Consensus reached in round {round_num}")
                    break
                elif disagreements:
                    print(f"\n   ⚠️ Ongoing disagreements: {len(disagreements)} points")
        
        return {
            "total_rounds": self.round_number,
            "history": self.discussion_history,
            "research_conducted": list(self.research_cache.keys())
        }


def generate_final_report(discussion_result: dict) -> str:
    """Generate the final architecture review report."""
    
    history = discussion_result["history"]
    
    report = [
        "# Round Table Architecture Review Report",
        f"**Generated:** {TIMESTAMP}",
        f"**Total Discussion Rounds:** {discussion_result['total_rounds']}",
        "",
        "## Participating Engineers",
        "",
        "| Engineer | Expertise | Model |",
        "|----------|-----------|-------|",
    ]
    
    for key, p in PARTICIPANTS.items():
        report.append(f"| {p['name']} | {p['expertise']} | {p['model']} |")
    
    report.extend([
        "",
        "---",
        "",
        "## Discussion Transcript",
        ""
    ])
    
    # Group by round
    rounds: dict[int, list] = {}
    for entry in history:
        r = entry.get("round", 0)
        if r not in rounds:
            rounds[r] = []
        rounds[r].append(entry)
    
    for round_num in sorted(rounds.keys()):
        report.append(f"### Round {round_num}")
        report.append("")
        
        for entry in rounds[round_num]:
            report.append(f"#### {entry['name']}")
            report.append("")
            report.append(entry["content"])
            report.append("")
            report.append("---")
            report.append("")
    
    # Research conducted
    if discussion_result.get("research_conducted"):
        report.extend([
            "## Research Topics Queried",
            ""
        ])
        for topic in discussion_result["research_conducted"]:
            report.append(f"- {topic}")
        report.append("")
    
    report.extend([
        "---",
        "",
        "*This review was conducted using the Round Table multi-LLM pattern,*",
        "*with all participants acting as equal Principal Engineer peers.*",
        "*Research conducted via semantic search (Qdrant) + graph RAG (Neo4j).*"
    ])
    
    return "\n".join(report)


async def main():
    """Main execution flow."""
    print("╔" + "═"*70 + "╗")
    print("║" + " ROUND TABLE ARCHITECTURE REVIEW ".center(70) + "║")
    print("║" + f" Generated: {TIMESTAMP} ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize clients
    search_client = SemanticSearchClient()
    llm_client = LLMClient()
    
    try:
        # Load documents
        print("📄 Loading review documents...")
        context = {
            "architecture_doc": load_document(ARCHITECTURE_DOC),
            "code_quality_doc": load_document(CODE_QUALITY_DOC),
            "multi_llm_doc": load_document(MULTI_LLM_DOC),
            "gemini_doc": load_document(GEMINI_DOC)
        }
        print(f"   ✅ Loaded 4 documents")
        
        # Initial research
        print("\n📚 Conducting initial research...")
        initial_topics = [
            "microservices architecture patterns",
            "LLM orchestration best practices",
            "service mesh security",
            "observability distributed systems"
        ]
        
        initial_research = []
        for topic in initial_topics:
            print(f"   📚 {topic}")
            results = await search_client.search(topic, limit=3)
            for r in results:
                source = r.get("metadata", {}).get("source", "Unknown")
                content = r.get("content", "")[:400]
                initial_research.append(f"[{source}]: {content}")
        
        context["initial_research"] = "\n\n".join(initial_research)
        print(f"   ✅ Initial research complete")
        
        # Run round table discussion
        discussion = RoundTableDiscussion(search_client, llm_client)
        result = await discussion.run_full_discussion(context)
        
        # Generate and save report
        print("\n📝 Generating final report...")
        report = generate_final_report(result)
        report_path = OUTPUT_DIR / "ARCHITECTURE_ROUNDTABLE_REVIEW.md"
        report_path.write_text(report)
        print(f"   ✅ Report saved to: {report_path}")
        
        # Save raw discussion data
        (OUTPUT_DIR / "discussion_data.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        
        print("\n" + "═"*70)
        print("REVIEW COMPLETE")
        print(f"Total rounds: {result['total_rounds']}")
        print(f"Research topics: {len(result['research_conducted'])}")
        print(f"Output: {OUTPUT_DIR}")
        print("═"*70)
        
    finally:
        await search_client.close()
        await llm_client.close()


if __name__ == "__main__":
    asyncio.run(main())
