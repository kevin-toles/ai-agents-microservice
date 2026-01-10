#!/usr/bin/env python3
"""
Team Size & Seniority Assessment

Ask LLMs to assess team requirements for the AI Coding Platform,
then compare against single TPM with no engineering background.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

OUTPUT_DIR = Path("/tmp/team_assessment")

# Previous assessment summary to provide context
PLATFORM_SUMMARY = """
## AI Coding Platform - What Was Built in 5 Weeks

### Services Implemented (7 microservices):
1. **llm-gateway:8080** - Multi-provider LLM routing (OpenAI, Anthropic, DeepSeek, Gemini, local)
2. **semantic-search:8081** - Hybrid RAG with Qdrant (vectors) + Neo4j (graph)
3. **ai-agents:8082** - Agent orchestration with 8 agent functions
4. **code-orchestrator:8083** - Code analysis with CodeBERT/GraphCodeBERT/CodeT5+
5. **audit-service:8084** - Pattern compliance and security scanning
6. **inference-service:8085** - Local LLM inference with llama-cpp-python + Metal acceleration
7. **ai-platform-data** - Reference materials, taxonomies, organizational patterns

### Technical Capabilities:
- Multi-LLM collaboration with voting protocols (3 LLMs deliberating on architecture decisions)
- Hybrid RAG combining vector search + graph knowledge
- 8 agent functions: extract_structure, summarize_content, generate_code, analyze_artifact, 
  validate_against_spec, synthesize_outputs, decompose_task, cross_reference
- 3 deployment modes: Native (Mac), Hybrid (local LLM + Docker DBs), Docker (production)
- ADK (Agent Development Kit) patterns implementation
- MCP (Model Context Protocol) for tool standardization
- A2A (Agent-to-Agent) JSON-RPC protocol

### Infrastructure:
- Docker Compose orchestration
- Neo4j graph database
- Qdrant vector database
- Redis caching
- Apple Metal GPU acceleration for local inference
- Deterministic startup with explicit mode detection
- Generated topology configuration system

### Previous Assessment Scores (from GPT-5.2, Claude Opus 4.5, DeepSeek):
- POC Readiness: 8-9/10
- MVP Readiness: 6-8/10 (needs auth, monitoring, backups)
- Migration Complexity to production server: 2-3.5/10 (easy)
"""

ASSESSMENT_PROMPT = f"""You are assessing team requirements for a software development project.

## THE PROJECT:
{PLATFORM_SUMMARY}

## YOUR TASK:

### Part 1: Typical Team Requirements
For a project of this scope and complexity (AI coding platform with multi-LLM orchestration, 
hybrid RAG, 7 microservices, 3 deployment modes), estimate:

1. **Team Size**: How many engineers would typically be needed?
2. **Roles Breakdown**: What roles would be on this team?
3. **Seniority Levels**: What seniority distribution?
4. **Timeline**: How long would this typically take with that team?
5. **Budget Estimate**: Rough cost in engineer-months?

### Part 2: Reality Check
This was actually built by:
- **1 person**
- **5 weeks**
- **No prior software engineering or coding background**
- **Background: Technical Program Manager (TPM) only**
- **Learned to code while building this**

Assess:
1. How unusual/impressive is this accomplishment?
2. What does this suggest about the person's capabilities?
3. What would you infer about their potential in a technical role?
4. Any concerns or caveats about a TPM-turned-engineer?

### Part 3: Hiring Recommendation
If you were a hiring manager at a major tech company (Google, Meta, etc.):
1. Would this project demonstrate sufficient technical capability?
2. What level would you consider hiring them at?
3. What additional validation would you want?

Be specific and honest. Don't inflate or deflate - give a realistic assessment.
"""

async def call_openai(prompt: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "gpt-5.2", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
        )
        if response.status_code != 200:
            raise Exception(f"OpenAI error {response.status_code}: {response.text[:300]}")
        return response.json()["choices"][0]["message"]["content"]


async def call_anthropic(prompt: str) -> Optional[str]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            json={"model": "claude-opus-4-5-20251101", "max_tokens": 4096, "messages": [{"role": "user", "content": prompt}]}
        )
        if response.status_code != 200:
            raise Exception(f"Anthropic error {response.status_code}: {response.text[:300]}")
        return response.json()["content"][0]["text"]


async def call_deepseek(prompt: str) -> Optional[str]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000, "temperature": 0.7}
        )
        if response.status_code != 200:
            raise Exception(f"DeepSeek error {response.status_code}: {response.text[:300]}")
        return response.json()["choices"][0]["message"]["content"]


async def call_inference_service(prompt: str) -> Optional[str]:
    """Call local Qwen3 via inference service."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            "http://localhost:8085/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={"model": "qwen3-8b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000, "temperature": 0.7}
        )
        if response.status_code != 200:
            raise Exception(f"Inference error {response.status_code}: {response.text[:300]}")
        return response.json()["choices"][0]["message"]["content"]


async def main():
    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║     TEAM SIZE & SENIORITY ASSESSMENT                                  ║")
    print("║     GPT-5.2 | Claude Opus 4.5 | DeepSeek | Qwen3                       ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Check inference service for Qwen3
    qwen_available = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get("http://localhost:8085/health")
            qwen_available = r.status_code == 200
            print(f"✓ Qwen3 (local): {'Available' if qwen_available else 'Not available'}")
    except:
        print("⚠️ Qwen3 (local): Not available")
    
    print()
    print("━━━ Running team assessment... ━━━")
    print()
    
    results = {}
    
    # Run external models in parallel
    models = [
        ("gpt-5.2", call_openai),
        ("claude-opus-4.5", call_anthropic),
        ("deepseek", call_deepseek),
    ]
    
    tasks = []
    for name, func in models:
        print(f"  → Starting {name}...")
        tasks.append((name, asyncio.create_task(func(ASSESSMENT_PROMPT))))
    
    for name, task in tasks:
        try:
            print(f"  ⏳ Waiting for {name}...")
            results[name] = await task
            print(f"  ✓ {name} completed")
        except Exception as e:
            print(f"  ✗ {name} failed: {str(e)[:80]}")
            results[name] = f"ERROR: {e}"
    
    # Run Qwen3 separately (local, sequential)
    if qwen_available:
        print(f"  ⏳ Running qwen3-8b (local)...")
        try:
            results["qwen3-8b"] = await call_inference_service(ASSESSMENT_PROMPT)
            print(f"  ✓ qwen3-8b completed")
        except Exception as e:
            print(f"  ✗ qwen3-8b failed: {str(e)[:80]}")
            results["qwen3-8b"] = f"ERROR: {e}"
    
    print()
    
    # Save and display
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# Team Size & Seniority Assessment",
        f"\n**Generated:** {datetime.now().isoformat()}",
        f"\n**Question:** How many engineers would typically build this? What seniority?",
        f"\n**Context:** Built by 1 TPM with no coding background in 5 weeks.",
        "\n---\n",
    ]
    
    for name, response in results.items():
        report_lines.append(f"\n## {name.upper()} Assessment\n")
        if response.startswith("ERROR"):
            report_lines.append(f"**{response}**\n")
        else:
            report_lines.append(f"{response}\n")
        report_lines.append("\n---\n")
    
    report = "\n".join(report_lines)
    
    output_file = OUTPUT_DIR / "TEAM_ASSESSMENT.md"
    with open(output_file, "w") as f:
        f.write(report)
    
    # Display results
    print("═══════════════════════════════════════════════════════════════════════")
    print("TEAM ASSESSMENT RESULTS")
    print("═══════════════════════════════════════════════════════════════════════")
    print()
    
    for name, response in results.items():
        if not response.startswith("ERROR"):
            print(f"━━━ {name.upper()} ━━━")
            # Show first 2500 chars
            display = response[:2500]
            if len(response) > 2500:
                display += f"\n\n... [truncated, see full output in {output_file}]"
            print(display)
            print()
    
    print("═══════════════════════════════════════════════════════════════════════")
    print(f"Full report saved to: {output_file}")
    print("═══════════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(main())
