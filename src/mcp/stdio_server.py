#!/usr/bin/env python3
"""FastMCP stdio server exposing all 8 agent functions.

Production-grade MCP server using stdio transport for VS Code/Claude Desktop.

Usage:
    # Direct execution
    python -m src.mcp.stdio_server
    
    # Via FastMCP CLI
    fastmcp run src/mcp/stdio_server.py

VS Code Configuration (~/.config/mcp/settings.json):
    {
      "mcpServers": {
        "ai-platform": {
          "command": "python",
          "args": ["-m", "src.mcp.stdio_server"],
          "cwd": "/Users/kevintoles/POC/ai-agents"
        }
      }
    }

Reference: WBS-PI5, ADK_MIGRATION_GUIDE.md, PROTOCOL_INTEGRATION_ARCHITECTURE.md
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastmcp import FastMCP

# Configure logging to stderr (stdout is for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP(
    name="ai-platform-agent-functions",
    instructions="Kitchen Brigade agent functions for AI-powered code understanding. "
                 "Use these tools to extract structure, summarize content, generate code, "
                 "analyze artifacts, validate against specs, decompose tasks, synthesize "
                 "outputs, and cross-reference across knowledge sources.",
)


# =============================================================================
# Agent Function Tools
# =============================================================================


@mcp.tool()
async def extract_structure(
    content: str,
    extraction_type: str = "outline",
) -> dict:
    """Extract structured data from unstructured content.
    
    Parses JSON/Markdown/Code into hierarchical structure with headings,
    sections, and code blocks.
    
    Args:
        content: The content to extract structure from
        extraction_type: Type of extraction - "outline", "entities", "keywords"
    
    Returns:
        Structured output with headings, sections, code_blocks
    """
    from src.functions.extract_structure import ExtractStructureFunction
    
    func = ExtractStructureFunction()
    result = await func.run(content=content, extraction_type=extraction_type)
    return result.model_dump() if hasattr(result, 'model_dump') else result


@mcp.tool()
async def summarize_content(
    content: str,
    detail_level: str = "standard",
    max_length: int = 500,
) -> dict:
    """Compress content while preserving key information.
    
    Generates summaries with citation markers for traceability.
    
    Args:
        content: The content to summarize
        detail_level: Level of detail - "brief", "standard", "detailed"
        max_length: Maximum length of summary in words
    
    Returns:
        Summary with citations and metadata
    """
    from src.functions.summarize_content import SummarizeContentFunction
    
    func = SummarizeContentFunction()
    result = await func.run(
        content=content,
        detail_level=detail_level,
        max_length=max_length,
    )
    return result.model_dump() if hasattr(result, 'model_dump') else result


@mcp.tool()
async def generate_code(
    specification: str,
    target_language: str = "python",
    include_tests: bool = False,
    context: str = "",
) -> dict:
    """Generate code from natural language specification.
    
    Args:
        specification: Natural language description of what to generate
        target_language: Programming language for output
        include_tests: Whether to include test stubs
        context: Additional context about the codebase
    
    Returns:
        Generated code with explanation
    """
    from src.functions.generate_code import GenerateCodeFunction
    
    func = GenerateCodeFunction()
    result = await func.run(
        specification=specification,
        target_language=target_language,
        include_tests=include_tests,
        context=context,
    )
    return result.model_dump() if hasattr(result, 'model_dump') else result


@mcp.tool()
async def analyze_artifact(
    artifact: str,
    analysis_type: str = "quality",
    context: str = "",
) -> dict:
    """Analyze code or document for patterns, issues, and quality.
    
    Args:
        artifact: The code or document to analyze
        analysis_type: Type of analysis - "quality", "security", "patterns"
        context: Additional context about the artifact
    
    Returns:
        Analysis results with findings and recommendations
    """
    from src.functions.analyze_artifact import AnalyzeArtifactFunction
    
    func = AnalyzeArtifactFunction()
    result = await func.run(
        artifact=artifact,
        analysis_type=analysis_type,
        context=context,
    )
    return result.model_dump() if hasattr(result, 'model_dump') else result


@mcp.tool()
async def validate_against_spec(
    artifact: str,
    specification: str,
    acceptance_criteria: list[str] | None = None,
) -> dict:
    """Validate an artifact against its specification.
    
    Compares code/content against requirements and acceptance criteria.
    
    Args:
        artifact: The artifact to validate
        specification: The specification to validate against
        acceptance_criteria: List of specific criteria to check
    
    Returns:
        Validation results with compliance percentage and issues
    """
    from src.functions.validate_against_spec import ValidateAgainstSpecFunction
    
    func = ValidateAgainstSpecFunction()
    result = await func.run(
        artifact=artifact,
        specification=specification,
        acceptance_criteria=acceptance_criteria or [],
    )
    return result.model_dump() if hasattr(result, 'model_dump') else result


@mcp.tool()
async def decompose_task(
    task: str,
    constraints: list[str] | None = None,
    available_agents: list[str] | None = None,
    context: str = "",
) -> dict:
    """Decompose a complex task into subtasks.
    
    Breaks down high-level objectives into executable subtasks with
    dependencies, forming a valid DAG for pipeline execution.
    
    Args:
        task: The task to decompose
        constraints: Constraints to consider during decomposition
        available_agents: List of available agent functions
        context: Additional context about the task
    
    Returns:
        List of subtasks with dependencies
    """
    from src.functions.decompose_task import DecomposeTaskFunction
    
    func = DecomposeTaskFunction()
    result = await func.run(
        task=task,
        constraints=constraints or [],
        available_agents=available_agents or [],
        context=context,
    )
    return result.model_dump() if hasattr(result, 'model_dump') else result


@mcp.tool()
async def synthesize_outputs(
    outputs: list[str],
    synthesis_strategy: str = "merge",
    conflict_resolution: str = "latest",
) -> dict:
    """Combine multiple outputs into a coherent result.
    
    Merges outputs from multiple agents while tracking provenance.
    
    Args:
        outputs: List of outputs to synthesize
        synthesis_strategy: Strategy - "merge", "chain", "vote"
        conflict_resolution: How to resolve conflicts - "latest", "vote", "manual"
    
    Returns:
        Synthesized output with provenance tracking
    """
    from src.functions.summarize_content import SummarizeContentFunction
    
    # synthesize_outputs currently reuses SummarizeContentFunction
    func = SummarizeContentFunction()
    combined = "\n\n---\n\n".join(outputs)
    result = await func.run(
        content=combined,
        detail_level="detailed",
    )
    return result.model_dump() if hasattr(result, 'model_dump') else result


@mcp.tool()
async def cross_reference(
    query: str,
    sources: list[str] | None = None,
    top_k: int = 5,
    include_code: bool = True,
    include_books: bool = True,
) -> dict:
    """Find related content across knowledge sources.
    
    Queries semantic search to find related content across code,
    documentation, and textbooks.
    
    Args:
        query: The query to search for
        sources: Specific sources to search (default: all)
        top_k: Number of results to return
        include_code: Whether to search code repositories
        include_books: Whether to search textbooks
    
    Returns:
        Cross-reference results with relevance scores and citations
    """
    from src.functions.cross_reference import CrossReferenceFunction
    
    func = CrossReferenceFunction()
    result = await func.run(
        query_artifact=query,
        sources=sources,
        top_k=top_k,
        include_code=include_code,
        include_books=include_books,
    )
    return result.model_dump() if hasattr(result, 'model_dump') else result


# =============================================================================
# Tiered LLM Fallback Tool (NEW - Interview Feature)
# =============================================================================


@mcp.tool()
async def llm_complete(
    prompt: str,
    model_preference: str = "auto",
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> dict:
    """Generate LLM completion with tiered fallback.
    
    Tier 1: Local inference-service (if available)
    Tier 2: Cloud LLM via llm-gateway (OpenAI/Anthropic)
    Tier 3: Return work package for client to handle
    
    Args:
        prompt: The prompt to complete
        model_preference: Model preference - "auto", "local", "cloud", or specific model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Completion result with tier used and metadata
    """
    import httpx
    
    messages = [{"role": "user", "content": prompt}]
    
    # Tier 1: Try local inference-service
    if model_preference in ("auto", "local"):
        try:
            inference_url = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8085")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    inference_url + "/v1/chat/completions",
                    json={
                        "model": "auto",
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "tier": "local",
                        "model": data.get("model", "local"),
                        "content": data["choices"][0]["message"]["content"],
                        "usage": data.get("usage", {}),
                    }
        except Exception as e:
            logger.warning(f"Tier 1 (local) failed: {e}")
    
    # Tier 2: Try cloud via llm-gateway
    if model_preference in ("auto", "cloud"):
        try:
            # Get model from environment, with fallback chain
            cloud_model = os.getenv("LLM_GATEWAY_DEFAULT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    os.getenv("LLM_GATEWAY_URL", "http://localhost:8081") + "/v1/chat/completions",
                    json={
                        "model": cloud_model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "tier": "cloud",
                        "model": data.get("model", cloud_model),
                        "content": data["choices"][0]["message"]["content"],
                        "usage": data.get("usage", {}),
                    }
        except Exception as e:
            logger.warning(f"Tier 2 (cloud) failed: {e}")
    
    # Tier 3: Return work package for client to handle
    return {
        "tier": "deferred",
        "model": None,
        "content": None,
        "work_package": {
            "type": "llm_completion",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reason": "All LLM tiers unavailable",
        },
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting AI Platform MCP Server (stdio transport)")
    mcp.run()
