#!/usr/bin/env python3
"""
Kitchen Brigade Architecture Reassessment Script

This script orchestrates 4 LLMs to reassess the ARCHITECTURE_DECISION_RECORD.md
against the findings from:
- CODE_QUALITY_ASSESSMENT.md
- MULTI_LLM_ASSESSMENT.md  
- GEMINI_ASSESSMENT.md

Features:
- Routes through LLM Gateway → Inference Service for local models
- Uses semantic search (Qdrant) + graph RAG (Neo4j) for research
- Generates Chicago-style citations
- Kitchen Brigade pattern: Proposer → Architect → Critic → Integrator

WBS: Multi-LLM Architecture Validation
"""

import asyncio
import json
import os
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
OUTPUT_DIR = Path("/Users/kevintoles/POC/Platform Documentation/architecture_reassessment")
TIMESTAMP = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# Document paths
DOCS_BASE = Path("/Users/kevintoles/POC/Platform Documentation")
ARCHITECTURE_DOC = DOCS_BASE / "ARCHITECTURE_DECISION_RECORD.md"
CODE_QUALITY_DOC = DOCS_BASE / "code_quality_assessment" / "CODE_QUALITY_ASSESSMENT.md"
MULTI_LLM_DOC = DOCS_BASE / "multi_llm_assessment" / "MULTI_LLM_ASSESSMENT.md"
GEMINI_DOC = DOCS_BASE / "gemini_review" / "GEMINI_ASSESSMENT.md"

# LLM Configuration for Kitchen Brigade
# Note: Using available local models + external via LLM Gateway
LLM_CONFIG = {
    "proposer": {
        "name": "Qwen3 8B",
        "model": "qwen3-8b",
        "provider": "local",  # Routes through inference-service
        "role": "Proposer - Initial synthesis and improvement proposals",
        "context_length": 32768  # Qwen3 has 32K context
    },
    "architect": {
        "name": "DeepSeek R1 7B", 
        "model": "deepseek-r1-7b",
        "provider": "local",
        "role": "Architect - Technical validation and architecture patterns",
        "context_length": 32768
    },
    "critic": {
        "name": "GPT-5.2",
        "model": "gpt-5.2",
        "provider": "openai",
        "role": "Critic - Risk identification and gap analysis",
        "context_length": 128000
    },
    "integrator": {
        "name": "Claude Opus 4.5",
        "model": "claude-opus-4-20250514",
        "provider": "anthropic",
        "role": "Integrator - Final synthesis with Chicago citations",
        "context_length": 200000
    }
}


class ChicagoCitationGenerator:
    """Generates Chicago-style citations for textbook references."""
    
    # Known textbook metadata for citation generation
    TEXTBOOK_METADATA = {
        "Machine Learning Engineering": {
            "author": "Burkov, Andriy",
            "year": "2020",
            "publisher": "True Positive Inc.",
            "location": "Quebec City"
        },
        "Designing Data-Intensive Applications": {
            "author": "Kleppmann, Martin",
            "year": "2017",
            "publisher": "O'Reilly Media",
            "location": "Sebastopol, CA"
        },
        "Generative AI with LangChain": {
            "author": "Auffarth, Ben",
            "year": "2024",
            "publisher": "Packt Publishing",
            "location": "Birmingham, UK"
        },
        "Game Programming Gems 8": {
            "author": "Lake, Adam",
            "year": "2010",
            "publisher": "Course Technology PTR",
            "location": "Boston"
        },
        "AntiPatterns": {
            "author": "Brown, William J., et al.",
            "year": "1998",
            "publisher": "Wiley",
            "location": "New York"
        }
    }
    
    @classmethod
    def generate_citation(cls, source: str, chapter: str = None, page: str = None) -> str:
        """Generate Chicago-style citation."""
        metadata = cls.TEXTBOOK_METADATA.get(source, {})
        
        if not metadata:
            # Fallback for unknown sources
            return f'{source}. Accessed {datetime.now().strftime("%B %d, %Y")}.'
        
        citation = f'{metadata["author"]}. {source}'
        if chapter:
            citation += f', chap. "{chapter}"'
        citation += f'. {metadata["location"]}: {metadata["publisher"]}, {metadata["year"]}'
        if page:
            citation += f', {page}'
        citation += '.'
        
        return citation

    @classmethod
    def format_bibliography(cls, references: list[dict]) -> str:
        """Format a list of references as Chicago bibliography."""
        bibliography = ["## Bibliography", ""]
        
        for ref in sorted(references, key=lambda x: x.get("author", x.get("source", ""))):
            citation = cls.generate_citation(
                ref.get("source", "Unknown"),
                ref.get("chapter"),
                ref.get("page")
            )
            bibliography.append(f"- {citation}")
        
        return "\n".join(bibliography)


class SemanticSearchClient:
    """Client for hybrid semantic search (Qdrant + Neo4j)."""
    
    def __init__(self, base_url: str = SEMANTIC_SEARCH_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def hybrid_search(self, query: str, limit: int = 10) -> dict:
        """
        Perform hybrid search combining vector (Qdrant) and graph (Neo4j).
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            Search results with scores and metadata
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/search/hybrid",
                json={
                    "query": query,
                    "limit": limit,
                    "collection": "chapters",
                    "include_graph": True,
                    "tier_boost": True
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"⚠️  Semantic search error: {e}")
            return {"results": [], "error": str(e)}
    
    async def graph_query(self, concept: str) -> dict:
        """
        Query Neo4j graph for related concepts and patterns.
        
        Args:
            concept: Concept to search for
            
        Returns:
            Graph relationships and connected concepts
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/search/graph",
                json={
                    "concept": concept,
                    "depth": 2,
                    "include_patterns": True
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"⚠️  Graph query error: {e}")
            return {"nodes": [], "relationships": [], "error": str(e)}
    
    async def close(self):
        await self.client.aclose()


class LLMGatewayClient:
    """Client for LLM Gateway with routing to local and cloud providers."""
    
    def __init__(self, gateway_url: str = LLM_GATEWAY_URL):
        self.gateway_url = gateway_url
        self.inference_url = INFERENCE_SERVICE_URL
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def complete(
        self,
        model: str,
        messages: list[dict],
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> dict:
        """
        Send completion request through LLM Gateway.
        
        Routes to:
        - inference-service for local models (qwen, deepseek)
        - LLM Gateway for cloud providers (openai, anthropic)
        """
        try:
            # Route local models directly to inference-service
            if provider == "local":
                return await self._local_completion(model, messages, temperature, max_tokens)
            
            # Route cloud models through LLM Gateway
            response = await self.client.post(
                f"{self.gateway_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _local_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int
    ) -> dict:
        """Route to local inference service for GGUF models."""
        try:
            response = await self.client.post(
                f"{self.inference_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Local inference error: {e}"}
    
    async def close(self):
        await self.client.aclose()


def load_document(path: Path) -> str:
    """Load document content from file."""
    try:
        return path.read_text()
    except Exception as e:
        print(f"⚠️  Error loading {path}: {e}")
        return f"[Error loading document: {e}]"


def truncate_for_context(text: str, max_chars: int = 15000) -> str:
    """Truncate text to fit context window while preserving structure."""
    if len(text) <= max_chars:
        return text
    
    # Keep beginning and end, truncate middle
    half = max_chars // 2
    return text[:half] + "\n\n[... content truncated for context limits ...]\n\n" + text[-half:]


async def research_topics(search_client: SemanticSearchClient, topics: list[str]) -> dict:
    """
    Research relevant topics using hybrid search.
    
    Returns research context with citations.
    """
    all_results = []
    citations = []
    
    for topic in topics:
        print(f"  📚 Researching: {topic}")
        
        # Hybrid search (vector + graph)
        results = await search_client.hybrid_search(topic, limit=5)
        
        if "results" in results and results["results"]:
            for result in results["results"][:3]:
                content = result.get("content", result.get("text", ""))[:500]
                metadata = result.get("metadata", {})
                
                all_results.append({
                    "topic": topic,
                    "source": metadata.get("book", metadata.get("source", "Unknown")),
                    "chapter": metadata.get("chapter", metadata.get("title", "")),
                    "score": result.get("score", 0),
                    "content": content
                })
                
                citations.append({
                    "source": metadata.get("book", metadata.get("source", "Unknown")),
                    "chapter": metadata.get("chapter", metadata.get("title", "")),
                    "page": metadata.get("page")
                })
    
    return {
        "results": all_results,
        "citations": citations,
        "formatted_context": _format_research_context(all_results)
    }


def _format_research_context(results: list[dict]) -> str:
    """Format research results for LLM context."""
    if not results:
        return "No additional research available."
    
    lines = ["=== RESEARCH CONTEXT FROM TEXTBOOK CORPUS ===\n"]
    
    for r in results:
        lines.append(f"**Topic:** {r['topic']}")
        lines.append(f"**Source:** {r['source']} - {r['chapter']}")
        lines.append(f"**Relevance Score:** {r['score']:.3f}")
        lines.append(f"**Content:** {r['content']}")
        lines.append("---\n")
    
    return "\n".join(lines)


async def run_kitchen_brigade_round(
    llm_client: LLMGatewayClient,
    role_key: str,
    context: dict,
    previous_responses: list[dict] = None
) -> dict:
    """
    Run a single Kitchen Brigade round with specified role.
    
    Args:
        llm_client: LLM Gateway client
        role_key: One of proposer, architect, critic, integrator
        context: Assessment context including documents and research
        previous_responses: Responses from previous rounds
        
    Returns:
        LLM response with role metadata
    """
    config = LLM_CONFIG[role_key]
    
    # Build role-specific prompt
    system_prompt = build_system_prompt(role_key, config)
    user_prompt = build_user_prompt(role_key, context, previous_responses)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print(f"\n{'='*60}")
    print(f"🎭 {config['name']} ({config['role']})")
    print(f"   Model: {config['model']} via {config['provider']}")
    print(f"{'='*60}")
    
    response = await llm_client.complete(
        model=config["model"],
        messages=messages,
        provider=config["provider"],
        temperature=0.7,
        max_tokens=4096
    )
    
    if "error" in response:
        print(f"   ❌ Error: {response['error']}")
        content = f"[Error from {config['name']}: {response['error']}]"
    else:
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"   ✅ Response received ({len(content)} chars)")
    
    return {
        "role": role_key,
        "name": config["name"],
        "model": config["model"],
        "provider": config["provider"],
        "content": content,
        "timestamp": datetime.now().isoformat()
    }


def build_system_prompt(role_key: str, config: dict) -> str:
    """Build role-specific system prompt."""
    
    base_context = """You are part of a Kitchen Brigade multi-LLM assessment team evaluating 
an AI platform architecture. Your task is to assess the ARCHITECTURE_DECISION_RECORD.md 
against findings from three independent assessments (Code Quality, Multi-LLM, and Gemini).

IMPORTANT: Use the research context from the textbook corpus to ground your recommendations
in established software engineering principles. Cite sources using Chicago style when referencing
specific patterns or recommendations from the research."""
    
    role_prompts = {
        "proposer": f"""{base_context}

You are the PROPOSER ({config['name']}). Your responsibilities:
1. Synthesize the key findings from all three assessment documents
2. Identify the TOP 5 architectural concerns that need addressing
3. Propose specific improvements for each concern
4. Reference relevant patterns from the research context
5. Flag any areas needing further research with "RESEARCH_NEEDED: <topic>"

Format your response with clear sections:
## Synthesized Findings
## Top 5 Concerns
## Proposed Improvements
## Research References""",

        "architect": f"""{base_context}

You are the ARCHITECT ({config['name']}). Your responsibilities:
1. Validate the Proposer's technical recommendations
2. Assess each proposed improvement for:
   - Technical feasibility
   - Implementation complexity
   - Alignment with architecture patterns from research
3. Suggest refinements or alternatives where needed
4. Prioritize improvements using MoSCoW (Must/Should/Could/Won't)
5. Identify any architectural anti-patterns

Format your response with:
## Technical Validation
## Prioritized Recommendations (MoSCoW)
## Anti-Patterns Identified
## Architecture Pattern References""",

        "critic": f"""{base_context}

You are the CRITIC ({config['name']}). Your responsibilities:
1. Challenge assumptions in both Proposer and Architect responses
2. Identify risks and potential failure modes
3. Play devil's advocate on proposed changes
4. Highlight what's MISSING from the analysis
5. Assess production readiness gaps

Be constructively critical. Your goal is to strengthen the final recommendations.

Format your response with:
## Challenged Assumptions
## Risk Analysis
## Missing Considerations
## Production Readiness Gaps
## Final Verdict""",

        "integrator": f"""{base_context}

You are the INTEGRATOR ({config['name']}). Your responsibilities:
1. Synthesize inputs from Proposer, Architect, and Critic
2. Produce a FINAL ASSESSMENT with:
   - Executive summary
   - Consolidated recommendations (prioritized)
   - Implementation roadmap
   - Risk mitigation plan
3. Generate Chicago-style bibliography for all research references
4. Provide a final production readiness score (1-10)

Your output is the FINAL DELIVERABLE. Be comprehensive and actionable.

Format your response with:
## Executive Summary
## Consolidated Recommendations
## Implementation Roadmap
## Risk Mitigation Plan
## Production Readiness Score
## Bibliography (Chicago Style)"""
    }
    
    return role_prompts.get(role_key, base_context)


def build_user_prompt(role_key: str, context: dict, previous_responses: list[dict] = None) -> str:
    """Build role-specific user prompt with context."""
    
    # Smaller context for local models (32K context ~ 24K tokens max)
    # Reserve ~8K tokens for response
    is_local = LLM_CONFIG[role_key]["provider"] == "local"
    doc_size = 4000 if is_local else 6000
    research_size = 2000 if is_local else 3000
    prev_size = 2000 if is_local else 3000
    
    # Base context from documents
    prompt_parts = [
        "=== ARCHITECTURE DECISION RECORD (Target Document) ===",
        truncate_for_context(context["architecture_doc"], doc_size),
        "",
        "=== CODE QUALITY ASSESSMENT FINDINGS ===",
        truncate_for_context(context["code_quality_doc"], doc_size // 2),
        "",
        "=== MULTI-LLM ASSESSMENT FINDINGS ===", 
        truncate_for_context(context["multi_llm_doc"], doc_size // 2),
        "",
        "=== GEMINI ASSESSMENT FINDINGS ===",
        truncate_for_context(context["gemini_doc"], doc_size // 2),
        "",
        "=== RESEARCH CONTEXT ===",
        truncate_for_context(context.get("research_context", "No research available."), research_size),
    ]
    
    # Add previous responses for later rounds
    if previous_responses:
        prompt_parts.append("\n=== PREVIOUS ROUND RESPONSES ===")
        for resp in previous_responses:
            prompt_parts.append(f"\n--- {resp['name']} ({resp['role'].upper()}) ---")
            prompt_parts.append(truncate_for_context(resp['content'], prev_size))
    
    prompt_parts.append("\n\nProvide your assessment now.")
    
    return "\n".join(prompt_parts)


async def main():
    """Main execution flow."""
    print("╔" + "═"*70 + "╗")
    print("║" + " KITCHEN BRIGADE ARCHITECTURE REASSESSMENT ".center(70) + "║")
    print("║" + f" Generated: {TIMESTAMP} ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize clients
    search_client = SemanticSearchClient()
    llm_client = LLMGatewayClient()
    
    try:
        # =====================================================================
        # STEP 1: Load Documents
        # =====================================================================
        print("📄 Loading assessment documents...")
        context = {
            "architecture_doc": load_document(ARCHITECTURE_DOC),
            "code_quality_doc": load_document(CODE_QUALITY_DOC),
            "multi_llm_doc": load_document(MULTI_LLM_DOC),
            "gemini_doc": load_document(GEMINI_DOC)
        }
        print(f"   ✅ Loaded 4 documents")
        
        # =====================================================================
        # STEP 2: Research Phase - Query semantic search + graph RAG
        # =====================================================================
        print("\n📚 Conducting research via Semantic Search + Graph RAG...")
        
        research_topics_list = [
            "service discovery patterns hybrid environments",
            "microservices security architecture",
            "LLM orchestration patterns",
            "configuration management distributed systems",
            "observability monitoring microservices",
            "single point of failure mitigation",
            "local development environment best practices"
        ]
        
        research = await research_topics(search_client, research_topics_list)
        context["research_context"] = research["formatted_context"]
        context["citations"] = research["citations"]
        
        print(f"   ✅ Retrieved {len(research['results'])} research results")
        
        # Save research context
        (OUTPUT_DIR / "research_context.json").write_text(
            json.dumps(research, indent=2, default=str)
        )
        
        # =====================================================================
        # STEP 3: Kitchen Brigade Rounds
        # =====================================================================
        responses = []
        
        # Round 1: Proposer (Qwen 2.5 7B - Local)
        print("\n" + "━"*70)
        print("ROUND 1: PROPOSER")
        print("━"*70)
        resp = await run_kitchen_brigade_round(llm_client, "proposer", context)
        responses.append(resp)
        (OUTPUT_DIR / "round1_proposer.md").write_text(resp["content"])
        
        # Round 2: Architect (DeepSeek R1 7B - Local)
        print("\n" + "━"*70)
        print("ROUND 2: ARCHITECT")
        print("━"*70)
        resp = await run_kitchen_brigade_round(llm_client, "architect", context, responses)
        responses.append(resp)
        (OUTPUT_DIR / "round2_architect.md").write_text(resp["content"])
        
        # Round 3: Critic (GPT-5.2 - Cloud)
        print("\n" + "━"*70)
        print("ROUND 3: CRITIC")
        print("━"*70)
        resp = await run_kitchen_brigade_round(llm_client, "critic", context, responses)
        responses.append(resp)
        (OUTPUT_DIR / "round3_critic.md").write_text(resp["content"])
        
        # Round 4: Integrator (Claude Opus 4.5 - Cloud)
        print("\n" + "━"*70)
        print("ROUND 4: INTEGRATOR (Final Synthesis)")
        print("━"*70)
        resp = await run_kitchen_brigade_round(llm_client, "integrator", context, responses)
        responses.append(resp)
        
        # =====================================================================
        # STEP 4: Generate Final Report
        # =====================================================================
        print("\n📝 Generating final report...")
        
        final_report = generate_final_report(responses, research["citations"])
        report_path = OUTPUT_DIR / "ARCHITECTURE_REASSESSMENT.md"
        report_path.write_text(final_report)
        
        print(f"\n✅ Final report saved to: {report_path}")
        
        # Save all responses as JSON for reference
        (OUTPUT_DIR / "all_responses.json").write_text(
            json.dumps(responses, indent=2, default=str)
        )
        
        print("\n" + "═"*70)
        print("ASSESSMENT COMPLETE")
        print(f"Output directory: {OUTPUT_DIR}")
        print("═"*70)
        
    finally:
        await search_client.close()
        await llm_client.close()


def generate_final_report(responses: list[dict], citations: list[dict]) -> str:
    """Generate the final assessment report with Chicago citations."""
    
    report_parts = [
        "# Architecture Reassessment Report",
        f"**Generated:** {TIMESTAMP}",
        f"**Assessment Type:** Kitchen Brigade Multi-LLM Review",
        "",
        "## Participating Models",
        "",
        "| Role | Model | Provider |",
        "|------|-------|----------|",
    ]
    
    for resp in responses:
        report_parts.append(f"| {resp['role'].title()} | {resp['name']} | {resp['provider']} |")
    
    report_parts.extend([
        "",
        "---",
        "",
        "## Assessment Rounds",
        ""
    ])
    
    # Add each round's response
    for i, resp in enumerate(responses, 1):
        report_parts.extend([
            f"### Round {i}: {resp['role'].title()} ({resp['name']})",
            "",
            resp["content"],
            "",
            "---",
            ""
        ])
    
    # Add bibliography
    report_parts.extend([
        "",
        ChicagoCitationGenerator.format_bibliography(citations),
        "",
        "---",
        "",
        "*This assessment was generated using the Kitchen Brigade multi-LLM framework,*",
        "*routing through LLM Gateway → Inference Service with semantic search + graph RAG.*"
    ])
    
    return "\n".join(report_parts)


if __name__ == "__main__":
    asyncio.run(main())
