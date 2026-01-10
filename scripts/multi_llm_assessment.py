#!/usr/bin/env python3
"""
Multi-LLM Platform Assessment

Runs the system design assessment across multiple LLMs:
- External: GPT-5.2, Claude Opus 4.5, DeepSeek
- Internal: Qwen3, Llama Coder (via inference-service)
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import httpx


DOCS_DIR = Path(__file__).parent.parent.parent / "Platform Documentation"
OUTPUT_DIR = Path("/tmp/multi_llm_assessment")


# Model configurations
EXTERNAL_MODELS = {
    "gpt-5.2": {
        "provider": "openai",
        "model_id": "gpt-5.2",
    },
    "claude-opus-4.5": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-5-20251101",
    },
    "deepseek": {
        "provider": "deepseek",
        "model_id": "deepseek-chat",
    },
}

INTERNAL_MODELS = {
    # Skip local models for now - context length too large for 8B models
    # "qwen3-8b": {
    #     "provider": "inference-service",
    #     "model_id": "qwen3-8b",
    # },
}


async def load_documents() -> tuple[str, str]:
    """Load the two main platform documents."""
    overview_path = DOCS_DIR / "GOOGLE_INTERVIEW_OVERVIEW.md"
    adr_path = DOCS_DIR / "ARCHITECTURE_DECISION_RECORD.md"
    
    with open(overview_path, "r") as f:
        overview = f.read()
    
    with open(adr_path, "r") as f:
        adr = f.read()
    
    return overview, adr


def build_assessment_prompt(overview: str, adr: str) -> str:
    """Build the assessment prompt."""
    return f"""You are a systems architect conducting a system design review for a **POC/MVP platform**. 

## CRITICAL CONTEXT - READ CAREFULLY:

1. **Scale**: POC (5-10 users) → MVP (25-50 users). NOT enterprise/Google-scale.

2. **Built by**: Single developer in 5 weeks. This is a demonstration of viability.

3. **The 3 Deployment Modes Are INTENTIONAL**: 
   - **Native mode**: Maximum performance for development/debugging on Apple Metal (M2 Ultra)
   - **Hybrid mode**: Mix of local LLMs + containerized databases for balanced dev experience  
   - **Docker mode**: Designed for DROP-IN MIGRATION to target MVP server hardware

4. **TARGET MVP HARDWARE** (TETRA R Mini-Server):
   | Component | Spec |
   |-----------|------|
   | CPU | AMD Ryzen 9 7950X |
   | GPU (Internal) | NVIDIA RTX 6000 Ada (48 GB GDDR6 ECC) |
   | GPU (External/eGPU) | NVIDIA RTX 4090 |
   | Memory | 128 GB DDR5 (2×64 GB) |
   | Storage | 4 TB + 2 TB PCIe 4.0 NVMe SSD |
   | PSU | 1000W SFX-L Platinum |

   The Docker architecture is specifically designed so the POC can be deployed to this server 
   with minimal changes. The "operational complexity" of 3 modes is a FEATURE, not a bug.

5. **Previous Review Context**: Gemini 2.0 Flash gave this 4/10 for "production readiness" 
   criticizing "Operational Complexity - Managing 3 deployment modes + ADK/MCP/A2A overhead"
   as over-engineering. This missed the point - it's a POC→MVP migration strategy.

## YOUR TASK:

Given this context, provide a FAIR assessment:

## 1. POC/MVP ARCHITECTURE ASSESSMENT
- Is this architecture appropriate for 5-50 users on the target hardware?
- Does the 3-mode strategy (Native/Hybrid/Docker) make sense for POC→MVP migration?
- What's genuinely over-engineered vs what's good forward planning?

## 2. MVP MIGRATION PATH
- How smooth will the Docker deployment be to the TETRA R server?
- What needs to change for the RTX 6000 Ada + 4090 dual-GPU setup?
- Is the architecture ready for 128GB RAM and fast NVMe storage?

## 3. DESIGN DECISIONS RE-EVALUATION
- ADK/MCP/A2A: Over-engineering for POC, or good foundation for team growth?
- Explicit mode detection: Smart for a drop-in migration scenario?
- Service decomposition: Right-sized or needs adjustment?

## 4. WHAT GEMINI GOT WRONG
- Which criticisms don't apply given this context?
- What concerns are still valid even at POC/MVP scale?

## 5. WHAT'S GENUINELY IMPRESSIVE
- What stands out for a 5-week single-developer POC?

## 6. TOP 3 ACTUAL PRIORITIES
- What should actually be addressed before MVP deployment?

## 7. REVISED SCORES
- POC Readiness (1-10): Ready to demo?
- MVP Readiness (1-10): Ready for 25-50 users on TETRA R?
- Migration Complexity (1-10): How hard is POC→MVP deployment?

Be fair. Evaluate against ACTUAL requirements, not enterprise standards.

--- PLATFORM OVERVIEW ---

{overview}

--- ARCHITECTURE DECISION RECORD ---

{adr}

--- PREVIOUS GEMINI ASSESSMENT (context) ---

Gemini gave 4/10 with top concerns:
1. Security Vulnerabilities (valid)
2. Single Points of Failure (less relevant for single-server MVP)
3. Operational Complexity - 3 deployment modes (THIS IS THE MIGRATION STRATEGY)

Re-evaluate: "Over-engineered" and "ADK/MCP/A2A adds complexity" given POC→MVP→TETRA R plan."""


async def call_openrouter(model_id: str, prompt: str) -> Optional[str]:
    """Call OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000,
        "temperature": 0.7,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/kevin-toles/ai-platform",
        "X-Title": "AI Platform Multi-LLM Assessment",
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            error_text = response.text[:500]
            raise Exception(f"OpenRouter error {response.status_code}: {error_text}")
        
        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
    
    return None


async def call_openai(model_id: str, prompt: str) -> Optional[str]:
    """Call OpenAI API directly."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    url = "https://api.openai.com/v1/chat/completions"
    
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            error_text = response.text[:500]
            raise Exception(f"OpenAI error {response.status_code}: {error_text}")
        
        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
    
    return None


async def call_anthropic(model_id: str, prompt: str) -> Optional[str]:
    """Call Anthropic API directly."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    url = "https://api.anthropic.com/v1/messages"
    
    payload = {
        "model": model_id,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            error_text = response.text[:500]
            raise Exception(f"Anthropic error {response.status_code}: {error_text}")
        
        data = response.json()
        content = data.get("content", [])
        if content:
            return content[0].get("text", "")
    
    return None


async def call_deepseek(model_id: str, prompt: str) -> Optional[str]:
    """Call DeepSeek API directly."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")
    
    url = "https://api.deepseek.com/v1/chat/completions"
    
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000,
        "temperature": 0.7,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            error_text = response.text[:500]
            raise Exception(f"DeepSeek error {response.status_code}: {error_text}")
        
        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
    
    return None


async def call_inference_service(model_id: str, prompt: str) -> Optional[str]:
    """Call local inference service."""
    url = "http://localhost:8085/v1/chat/completions"
    
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.7,
    }
    
    headers = {"Content-Type": "application/json"}
    
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            error_text = response.text[:500]
            raise Exception(f"Inference service error {response.status_code}: {error_text}")
        
        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
    
    return None


async def check_inference_service() -> bool:
    """Check if inference service is running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8085/health")
            return response.status_code == 200
    except:
        return False


async def run_assessment(model_name: str, config: dict, prompt: str) -> dict:
    """Run assessment for a single model."""
    result = {
        "model": model_name,
        "provider": config["provider"],
        "status": "pending",
        "response": None,
        "error": None,
    }
    
    try:
        provider = config["provider"]
        model_id = config["model_id"]
        
        if provider == "openai":
            response = await call_openai(model_id, prompt)
        elif provider == "anthropic":
            response = await call_anthropic(model_id, prompt)
        elif provider == "deepseek":
            response = await call_deepseek(model_id, prompt)
        elif provider == "openrouter":
            response = await call_openrouter(model_id, prompt)
        elif provider == "inference-service":
            response = await call_inference_service(model_id, prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        result["status"] = "success"
        result["response"] = response
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


async def main():
    """Run multi-LLM assessment."""
    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║     MULTI-LLM PLATFORM ASSESSMENT                                     ║")
    print("║     External: GPT-5.2, Claude Opus 4.5, DeepSeek                      ║")
    print("║     Internal: Qwen3-8B, CodeLlama-7B                                  ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Check prerequisites
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  OPENROUTER_API_KEY not set - external models will fail")
    
    inference_available = await check_inference_service()
    if not inference_available:
        print("⚠️  Inference service not running at localhost:8085")
        print("   Run: ./run_native.sh or start via VS Code task")
    else:
        print("✓ Inference service is running")
    
    print()
    
    # Load documents
    print("━━━ Loading platform documentation... ━━━")
    overview, adr = await load_documents()
    print(f"  ✓ Loaded {len(overview) + len(adr)} chars total")
    print()
    
    # Build prompt
    prompt = build_assessment_prompt(overview, adr)
    
    # Determine which models to run
    models_to_run = {}
    
    # External models (always try)
    models_to_run.update(EXTERNAL_MODELS)
    
    # Internal models (only if inference service is available)
    if inference_available:
        models_to_run.update(INTERNAL_MODELS)
    
    print(f"━━━ Running assessment on {len(models_to_run)} models... ━━━")
    print()
    
    results = {}
    
    # Run external models in parallel
    external_tasks = []
    for name, config in EXTERNAL_MODELS.items():
        print(f"  → Starting {name}...")
        external_tasks.append((name, config, run_assessment(name, config, prompt)))
    
    # Wait for external models
    for name, config, task in external_tasks:
        print(f"  ⏳ Waiting for {name}...")
        result = await task
        results[name] = result
        if result["status"] == "success":
            print(f"  ✓ {name} completed")
        else:
            print(f"  ✗ {name} failed: {result['error'][:100]}")
    
    # Run internal models sequentially (they share GPU)
    if inference_available:
        for name, config in INTERNAL_MODELS.items():
            print(f"  ⏳ Running {name} (local)...")
            result = await run_assessment(name, config, prompt)
            results[name] = result
            if result["status"] == "success":
                print(f"  ✓ {name} completed")
            else:
                print(f"  ✗ {name} failed: {result['error'][:100]}")
    
    print()
    
    # Save and display results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create combined report
    report_lines = [
        "# Multi-LLM Platform Assessment",
        f"\n**Generated:** {datetime.now().isoformat()}",
        f"\n**Models:** {', '.join(results.keys())}",
        "\n---\n",
    ]
    
    successful_count = 0
    for name, result in results.items():
        report_lines.append(f"\n## {name.upper()} Assessment\n")
        report_lines.append(f"**Provider:** {result['provider']}\n")
        report_lines.append(f"**Status:** {result['status']}\n")
        
        if result["status"] == "success":
            successful_count += 1
            report_lines.append(f"\n{result['response']}\n")
            report_lines.append("\n---\n")
        else:
            report_lines.append(f"\n**Error:** {result['error']}\n")
            report_lines.append("\n---\n")
    
    report = "\n".join(report_lines)
    
    # Save report
    output_file = OUTPUT_DIR / "MULTI_LLM_ASSESSMENT.md"
    with open(output_file, "w") as f:
        f.write(report)
    
    # Display results
    print("═══════════════════════════════════════════════════════════════════════")
    print(f"ASSESSMENT COMPLETE: {successful_count}/{len(results)} models succeeded")
    print("═══════════════════════════════════════════════════════════════════════")
    print()
    
    for name, result in results.items():
        if result["status"] == "success":
            print(f"━━━ {name.upper()} ━━━")
            print(result["response"][:2000])  # Truncate for display
            if len(result["response"]) > 2000:
                print(f"\n... [truncated, see full output in {output_file}]")
            print()
    
    print("═══════════════════════════════════════════════════════════════════════")
    print(f"Full report saved to: {output_file}")
    print("═══════════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(main())
