#!/usr/bin/env python3
"""
Generic Kitchen Brigade Protocol Executor

Runs multi-LLM discussion loops based on protocol definitions (JSON).
Agents can invoke this with different protocols for different purposes.

Usage:
    python -m protocols.kitchen_brigade_executor --protocol ARCHITECTURE_RECONCILIATION
    python -m protocols.kitchen_brigade_executor --protocol WBS_GENERATION --input "project=Phase2"
    python -m protocols.kitchen_brigade_executor --interactive  # Ask user for configuration
    
Stage 2 Cross-Reference Integration:
    When enable_cross_reference=True, runs 4-layer parallel retrieval:
    • Qdrant (vectors via semantic-search-service) - /v1/search/hybrid
    • Neo4j (graph via semantic-search-service) - /v1/graph/query
    • Textbooks (JSON files) - configured via infrastructure_config
    • Books Enriched (JSON files) - configured via infrastructure_config
    • Code-Orchestrator (ML Stack) - /v1/search, /api/v1/keywords
    
Infrastructure Modes (ARCHITECTURE_DECISION_RECORD.md):
    - docker: All services in Docker containers (uses Docker DNS)
    - hybrid: Databases in Docker, Python services native (uses localhost)
    - native: All services running natively (uses localhost)
    
    Set INFRASTRUCTURE_MODE env var or let it auto-detect.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import httpx
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

# Import infrastructure configuration
from src.infrastructure_config import (
    get_platform_config,
    get_cross_reference_config,
    get_infrastructure_mode,
    PlatformConfig,
)

console = Console()

# =============================================================================
# Dynamic Configuration (from infrastructure_config.py)
# =============================================================================

def _get_platform_endpoints() -> Dict[str, str]:
    """Get service endpoints based on infrastructure mode.
    
    Reference: ARCHITECTURE_DECISION_RECORD.md - Generated config artifact
    Reference: ARCHITECTURE_ROUNDTABLE_FINDINGS.md - Mode parity
    """
    config = get_platform_config()
    return {
        "inference_service": config.services.get("inference-service", "http://localhost:8085"),
        "llm_gateway": config.services.get("llm-gateway", "http://localhost:8080"),
        "semantic_search": config.services.get("semantic-search", "http://localhost:8081"),
        "code_orchestrator": config.services.get("code-orchestrator", "http://localhost:8083"),
    }

def _get_data_directories() -> Dict[str, Path]:
    """Get data directories for cross-reference Stage 2.
    
    These paths are resolved from infrastructure_config and can be
    overridden via environment variables.
    """
    xref_config = get_cross_reference_config()
    return {
        "textbooks": xref_config.get("textbooks_dir", Path(".")),
        "books_raw": xref_config.get("books_raw_dir", Path(".")),
        "books_enriched": xref_config.get("books_enriched_dir", Path(".")),
        "books_metadata": xref_config.get("books_metadata_dir", Path(".")),
    }

# Initialize endpoints and paths (will be refreshed per-execution if needed)
_ENDPOINTS = _get_platform_endpoints()
_DATA_DIRS = _get_data_directories()

# Convenience accessors (for backward compatibility)
INFERENCE_SERVICE = _ENDPOINTS["inference_service"]
LLM_GATEWAY = _ENDPOINTS["llm_gateway"]
SEMANTIC_SEARCH = _ENDPOINTS["semantic_search"]
CODE_ORCHESTRATOR = _ENDPOINTS["code_orchestrator"]

TEXTBOOKS_DIR = _DATA_DIRS["textbooks"]
BOOKS_RAW_DIR = _DATA_DIRS["books_raw"]
BOOKS_ENRICHED_DIR = _DATA_DIRS["books_enriched"]
BOOKS_METADATA_DIR = _DATA_DIRS["books_metadata"]

# =============================================================================
# Static Configuration
# =============================================================================

PROTOCOLS_DIR = Path(__file__).parent.parent.parent / "config" / "protocols"
PROMPTS_DIR = Path(__file__).parent.parent.parent / "config" / "prompts" / "kitchen_brigade"
RECOMMENDATIONS_FILE = Path(__file__).parent.parent.parent / "config" / "brigade_recommendations.yaml"

# External model prefixes that route through gateway
EXTERNAL_MODELS = {
    "claude-opus-4.5", "claude-sonnet-4.5",
    "gpt-5.2", "gpt-5.2-pro", "gpt-5-mini", "gpt-5-nano",
    "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"
}


class KitchenBrigadeExecutor:
    """Generic executor for Kitchen Brigade protocols."""
    
    def __init__(self, protocol_id: str, inputs: Dict[str, Any] = None, 
                 resume_from: Optional[str] = None, brigade_override: Optional[Dict] = None,
                 enable_cross_reference: bool = True, show_infra_status: bool = True):
        self.protocol_id = protocol_id
        self.inputs = inputs or {}
        self.protocol = self.load_protocol()
        self.brigade_override = brigade_override  # Custom model assignments
        self.enable_cross_reference = enable_cross_reference
        self.cross_reference_evidence = {}  # Stores retrieved evidence
        self.client = httpx.AsyncClient(timeout=300.0)
        self.resume_from = resume_from
        self.resumed_outputs = {}
        self.recommendations = self.load_recommendations()
        
        # Refresh infrastructure config (in case env changed)
        self._refresh_platform_config()
        
        if show_infra_status:
            self._display_infrastructure_status()
        
        if resume_from:
            existing_trace = self.load_trace(resume_from)
            self.resumed_outputs = existing_trace.get("outputs", {})
            self.trace = {
                "protocol_id": protocol_id,
                "start_time": existing_trace.get("start_time", datetime.now().isoformat()),
                "resumed_from": resume_from,
                "rounds": existing_trace.get("rounds", []),
            }
            console.print(f"[dim]Resuming from {resume_from} (starting at round {len(self.resumed_outputs) + 1})[/dim]")
        else:
            self.trace = {
                "protocol_id": protocol_id,
                "start_time": datetime.now().isoformat(),
                "rounds": [],
            }
    
    # =========================================================================
    # Infrastructure Configuration
    # =========================================================================
    
    def _refresh_platform_config(self) -> None:
        """Refresh endpoints and data directories from infrastructure config.
        
        This allows the executor to adapt to different deployment modes:
        - docker: All services in Docker containers
        - hybrid: Infrastructure in Docker, Python services native  
        - native: All services running natively
        
        Reference: ARCHITECTURE_DECISION_RECORD.md - D1, D2
        """
        global INFERENCE_SERVICE, LLM_GATEWAY, SEMANTIC_SEARCH, CODE_ORCHESTRATOR
        global TEXTBOOKS_DIR, BOOKS_RAW_DIR, BOOKS_ENRICHED_DIR, BOOKS_METADATA_DIR
        
        endpoints = _get_platform_endpoints()
        INFERENCE_SERVICE = endpoints["inference_service"]
        LLM_GATEWAY = endpoints["llm_gateway"]
        SEMANTIC_SEARCH = endpoints["semantic_search"]
        CODE_ORCHESTRATOR = endpoints["code_orchestrator"]
        
        data_dirs = _get_data_directories()
        TEXTBOOKS_DIR = data_dirs["textbooks"]
        BOOKS_RAW_DIR = data_dirs["books_raw"]
        BOOKS_ENRICHED_DIR = data_dirs["books_enriched"]
        BOOKS_METADATA_DIR = data_dirs["books_metadata"]
    
    def _display_infrastructure_status(self) -> None:
        """Display current infrastructure configuration.
        
        Shows the active deployment mode and configured endpoints.
        """
        mode = get_infrastructure_mode()
        config = get_platform_config()
        
        table = Table(title=f"🍳 Kitchen Brigade - Infrastructure Mode: [{mode.upper()}]", 
                      show_header=True, header_style="bold cyan")
        table.add_column("Service", style="dim")
        table.add_column("Endpoint", style="green")
        table.add_column("Status", justify="center")
        
        # Service endpoints
        for service, url in config.services.items():
            status = "🔗" if url else "❌"
            table.add_row(service, url or "Not configured", status)
        
        console.print(table)
        
        # Data directories
        console.print("\n[bold cyan]📁 Data Directories:[/bold cyan]")
        xref_config = get_cross_reference_config()
        for key in ["textbooks_dir", "books_raw_dir", "books_enriched_dir"]:
            path = xref_config.get(key, Path("."))
            exists = "✓" if path.exists() else "✗"
            console.print(f"  {exists} {key}: {path}")
        console.print()
    
    # =========================================================================
    # Stage 2: Cross-Reference (4-Layer Parallel Retrieval)
    # =========================================================================
    
    async def run_cross_reference(self, queries: List[str] = None) -> Dict[str, Any]:
        """
        Execute Stage 2: ParallelAgent(cross_reference) - 4-layer retrieval.
        
        Retrieves evidence from:
        • Qdrant (vectors) - semantic similarity search
        • Neo4j (graph) - relationship traversal
        • Textbooks (JSON) - reference material lookup
        • Code-Orchestrator (ML Stack): SBERT, CodeT5+, GraphCodeBERT, CodeBERT
        
        Args:
            queries: Optional list of queries. If not provided, extracts from inputs.
            
        Returns:
            Dict with evidence from all sources.
        """
        if not self.enable_cross_reference:
            return {}
        
        console.print("\n[bold cyan]━━━ Stage 2: Cross-Reference (4-Layer Parallel Retrieval) ━━━[/bold cyan]")
        console.print("[dim]  • Qdrant (vectors) - semantic similarity[/dim]")
        console.print("[dim]  • Neo4j (graph) - relationship traversal[/dim]")
        console.print("[dim]  • Textbooks - reference material[/dim]")
        console.print("[dim]  • Code-Orchestrator - ML keyword extraction[/dim]\n")
        
        # Extract queries from inputs if not provided
        if not queries:
            queries = self._extract_cross_reference_queries()
        
        if not queries:
            console.print("[yellow]No queries for cross-reference, skipping[/yellow]")
            return {}
        
        all_evidence = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running cross-reference...", total=len(queries))
            
            for query in queries:
                evidence = await self._cross_reference_single_query(query)
                all_evidence[query] = evidence
                
                total_results = sum(len(v) for v in evidence.values() if isinstance(v, list))
                console.print(f"  ✓ '{query[:40]}...' → {total_results} results")
                progress.update(task, advance=1)
        
        self.cross_reference_evidence = all_evidence
        self.trace["cross_reference"] = {
            "queries": queries,
            "evidence_counts": {q: sum(len(v) for v in e.values() if isinstance(v, list)) 
                               for q, e in all_evidence.items()}
        }
        
        return all_evidence
    
    def _extract_cross_reference_queries(self) -> List[str]:
        """Extract search queries from inputs for cross-reference."""
        queries = []
        
        # Check for explicit queries
        if "cross_reference_queries" in self.inputs:
            queries.extend(self.inputs["cross_reference_queries"])
        
        # Extract from topic
        if "topic" in self.inputs:
            queries.append(self.inputs["topic"])
        
        # Extract from documents (get key terms from first 500 chars)
        if "documents" in self.inputs:
            for doc_path in self.inputs["documents"][:3]:  # Limit to 3 docs
                path = Path(doc_path)
                if path.exists():
                    content = path.read_text()[:500]
                    # Extract potential query from document title/header
                    lines = content.split('\n')
                    for line in lines[:5]:
                        if line.startswith('#') or len(line) > 20:
                            queries.append(line.strip('#').strip()[:100])
                            break
        
        # Extract from previous stage outputs (for workflows)
        if "_previous_stage_outputs" in self.inputs:
            prev = self.inputs["_previous_stage_outputs"]
            # Look for key findings to query
            for stage_name, outputs in prev.items():
                for round_key, round_output in outputs.items():
                    if isinstance(round_output, dict):
                        for role, result in round_output.items():
                            if isinstance(result, dict) and "parsed" in result:
                                parsed = result["parsed"]
                                if isinstance(parsed, dict):
                                    # Extract findings/questions as queries
                                    if "key_points" in parsed:
                                        queries.extend(parsed["key_points"][:2])
                                    if "remaining_questions" in parsed:
                                        queries.extend(parsed["remaining_questions"][:2])
        
        # Deduplicate and limit
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower and q_lower not in seen and len(q_lower) > 5:
                seen.add(q_lower)
                unique_queries.append(q)
        
        return unique_queries[:10]  # Max 10 queries
    
    async def _cross_reference_single_query(self, query: str) -> Dict[str, List[Dict]]:
        """Run 4-layer parallel retrieval for a single query."""
        tasks = [
            self._search_qdrant(query),
            self._search_neo4j(query),
            self._search_textbooks(query),
            self._search_code_orchestrator(query),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "qdrant": results[0] if not isinstance(results[0], Exception) else [],
            "neo4j": results[1] if not isinstance(results[1], Exception) else [],
            "textbooks": results[2] if not isinstance(results[2], Exception) else [],
            "code_orchestrator": results[3] if not isinstance(results[3], Exception) else [],
        }
    
    async def _search_qdrant(self, query: str) -> List[Dict]:
        """Search Qdrant vector database via semantic-search-service hybrid endpoint.
        
        Uses /v1/search/hybrid which combines vector similarity with graph relationships.
        Collection: 'chapters' (indexed book chapters from ai-platform-data)
        """
        try:
            response = await self.client.post(
                f"{SEMANTIC_SEARCH}/v1/search/hybrid",
                json={
                    "query": query, 
                    "limit": 5, 
                    "collection": "chapters",
                    "include_graph": True,
                    "alpha": 0.7  # 70% vector, 30% graph
                }
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                return [{"source": "qdrant_hybrid", 
                        "content": r.get("payload", {}).get("content", "")[:500], 
                        "score": r.get("score", 0),
                        "vector_score": r.get("vector_score", 0),
                        "graph_score": r.get("graph_score"),
                        "metadata": r.get("payload", {})} 
                       for r in results]
        except Exception as e:
            console.print(f"[dim]    Qdrant/Hybrid: {e}[/dim]")
        return []
    
    async def _search_neo4j(self, query: str) -> List[Dict]:
        """Search Neo4j graph database via semantic-search-service graph query.
        
        Uses /v1/graph/query with a Cypher query to find concepts/chapters
        related to the search query terms.
        """
        try:
            # Search for concepts matching query terms
            cypher = """
            MATCH (c:Concept)-[r:MENTIONED_IN]->(ch:Chapter)
            WHERE toLower(c.name) CONTAINS toLower($query)
               OR toLower(ch.title) CONTAINS toLower($query)
            RETURN c.name as concept, ch.title as chapter, 
                   ch.book as book, type(r) as relationship
            LIMIT 10
            """
            response = await self.client.post(
                f"{SEMANTIC_SEARCH}/v1/graph/query",
                json={"cypher": cypher, "parameters": {"query": query}}
            )
            if response.status_code == 200:
                records = response.json().get("records", [])
                return [{"source": "neo4j_graph", 
                        "concept": r.get("concept", ""),
                        "chapter": r.get("chapter", ""),
                        "book": r.get("book", ""),
                        "content": f"{r.get('concept', '')} in {r.get('chapter', '')} ({r.get('book', '')})",
                        "relationship": r.get("relationship", "")} 
                       for r in records]
        except Exception as e:
            console.print(f"[dim]    Neo4j: {e}[/dim]")
        return []
    
    async def _search_textbooks(self, query: str) -> List[Dict]:
        """Search textbook JSON files directly.
        
        Searches two locations:
        1. /Users/kevintoles/POC/textbooks/JSON Texts/ - technical reference books
        2. /Users/kevintoles/POC/ai-platform-data/books/raw/ - raw book JSONs
        3. /Users/kevintoles/POC/ai-platform-data/books/enriched/ - enriched metadata
        
        Uses simple keyword matching (could be enhanced with SBERT embeddings).
        """
        results = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        try:
            # Search textbooks JSON directory
            await self._search_json_directory(TEXTBOOKS_DIR, query_terms, results, "textbook")
            
            # Search ai-platform-data books/raw
            await self._search_json_directory(BOOKS_RAW_DIR, query_terms, results, "book_raw")
            
            # Search enriched metadata for concepts/keywords
            await self._search_enriched_metadata(query_terms, results)
            
        except Exception as e:
            console.print(f"[dim]    Textbooks: {e}[/dim]")
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return results[:5]
    
    async def _search_json_directory(self, directory: Path, query_terms: set, 
                                     results: List[Dict], source_type: str) -> None:
        """Search JSON files in a directory for matching content."""
        if not directory.exists():
            return
            
        for json_file in directory.glob("*.json"):
            try:
                content = json_file.read_text(encoding="utf-8")
                data = json.loads(content)
                
                # Search chapters if available
                chapters = data.get("chapters", [])
                if isinstance(chapters, list):
                    for chapter in chapters[:20]:  # Limit chapters per book
                        chapter_content = chapter.get("content", "")
                        chapter_title = chapter.get("title", "")
                        
                        # Simple relevance scoring
                        content_lower = (chapter_content + " " + chapter_title).lower()
                        matches = sum(1 for term in query_terms if term in content_lower)
                        
                        if matches > 0:
                            relevance = matches / len(query_terms)
                            results.append({
                                "source": source_type,
                                "book": data.get("title", json_file.stem),
                                "chapter": chapter_title,
                                "chapter_number": chapter.get("chapter_number", chapter.get("number")),
                                "content": chapter_content[:400] if chapter_content else "",
                                "relevance": relevance,
                                "file_path": str(json_file)
                            })
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    
    async def _search_enriched_metadata(self, query_terms: set, results: List[Dict]) -> None:
        """Search enriched metadata for concepts and keywords."""
        if not BOOKS_ENRICHED_DIR.exists():
            return
            
        for json_file in BOOKS_ENRICHED_DIR.glob("*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                
                # Search enrichment_metadata if present
                enrichment = data.get("enrichment_metadata", {})
                concepts = enrichment.get("concepts", [])
                keywords = enrichment.get("keywords", [])
                
                # Check for matches in concepts/keywords
                all_terms = " ".join(concepts + keywords).lower()
                matches = sum(1 for term in query_terms if term in all_terms)
                
                if matches > 0:
                    results.append({
                        "source": "enriched_metadata",
                        "book": data.get("book", {}).get("title", json_file.stem),
                        "concepts": concepts[:10],
                        "keywords": keywords[:10],
                        "content": f"Concepts: {', '.join(concepts[:5])}. Keywords: {', '.join(keywords[:5])}",
                        "relevance": matches / len(query_terms),
                        "file_path": str(json_file)
                    })
                    
                # Also search chapter-level enrichments
                for chapter in data.get("chapters", [])[:10]:
                    ch_keywords = chapter.get("keywords", [])
                    ch_concepts = chapter.get("concepts", [])
                    ch_terms = " ".join(ch_keywords + ch_concepts).lower()
                    ch_matches = sum(1 for term in query_terms if term in ch_terms)
                    
                    if ch_matches > 0:
                        results.append({
                            "source": "enriched_chapter",
                            "book": data.get("book", {}).get("title", json_file.stem),
                            "chapter": chapter.get("title", ""),
                            "concepts": ch_concepts[:5],
                            "keywords": ch_keywords[:5],
                            "content": chapter.get("summary", "")[:300],
                            "relevance": ch_matches / len(query_terms)
                        })
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    
    async def _search_code_orchestrator(self, query: str) -> List[Dict]:
        """Search via Code-Orchestrator ML pipeline.
        
        Available endpoints:
        - POST /api/v1/keywords - TF-IDF keyword extraction
        - POST /v1/search - Full pipeline search (extract → search → curate)
        - POST /v1/extract/terms - Term extraction with BERT models
        
        ML Models:
        - SBERT: Sentence embeddings for semantic similarity
        - CodeT5+: Code understanding and keyword extraction  
        - GraphCodeBERT: Code-aware representations
        - CodeBERT: Code-NL understanding
        """
        results = []
        try:
            # 1. Full pipeline search via /v1/search
            response = await self.client.post(
                f"{CODE_ORCHESTRATOR}/v1/search",
                json={"query": query, "options": {"top_k": 5}}
            )
            if response.status_code == 200:
                search_results = response.json().get("results", [])
                for r in search_results:
                    results.append({
                        "source": "code_orchestrator", 
                        "type": "search",
                        "model": "SBERT+TF-IDF", 
                        "book": r.get("book", ""),
                        "chapter": r.get("chapter"),
                        "content": r.get("content", "")[:300],
                        "relevance_score": r.get("relevance_score", 0)
                    })
            
            # 2. Keyword extraction via /api/v1/keywords
            response = await self.client.post(
                f"{CODE_ORCHESTRATOR}/api/v1/keywords",
                json={"corpus": [query], "top_k": 10}
            )
            if response.status_code == 200:
                keywords_list = response.json().get("keywords", [[]])
                if keywords_list and keywords_list[0]:
                    results.append({
                        "source": "code_orchestrator", 
                        "type": "keywords",
                        "model": "TF-IDF", 
                        "content": keywords_list[0],
                        "keywords": keywords_list[0]
                    })
            
            # 3. Term extraction via /v1/extract/terms (if available)
            try:
                response = await self.client.post(
                    f"{CODE_ORCHESTRATOR}/v1/extract/terms",
                    json={"text": query, "options": {"use_bert": True}}
                )
                if response.status_code == 200:
                    terms = response.json().get("terms", [])
                    if terms:
                        results.append({
                            "source": "code_orchestrator",
                            "type": "term_extraction",
                            "model": "BERT",
                            "content": terms[:10],
                            "terms": terms[:10]
                        })
            except Exception:
                pass  # Term extraction endpoint may not be available
                
        except Exception as e:
            console.print(f"[dim]    Code-Orchestrator: {e}[/dim]")
        return results
    
    def format_evidence_for_prompt(self) -> str:
        """Format cross-reference evidence for inclusion in LLM prompts.
        
        Formats results from:
        - Qdrant hybrid search (vectors + graph)
        - Neo4j graph queries (concept relationships)
        - Textbooks/Books JSON files (chapter content)
        - Enriched metadata (concepts, keywords, summaries)
        - Code-Orchestrator (search results, keywords, terms)
        """
        if not self.cross_reference_evidence:
            return ""
        
        formatted = "\n## Cross-Reference Evidence (Stage 2 - 4-Layer Retrieval)\n\n"
        
        for query, evidence in self.cross_reference_evidence.items():
            formatted += f"### Query: \"{query[:80]}\"\n"
            
            # Qdrant/Hybrid results
            if evidence.get("qdrant"):
                formatted += "\n**QDRANT HYBRID SEARCH** (Vector + Graph):\n"
                for item in evidence["qdrant"][:3]:
                    score = item.get("score", 0)
                    content = item.get("content", "")[:180]
                    formatted += f"- [Score: {score:.3f}] {content}...\n"
            
            # Neo4j graph results
            if evidence.get("neo4j"):
                formatted += "\n**NEO4J KNOWLEDGE GRAPH**:\n"
                for item in evidence["neo4j"][:3]:
                    concept = item.get("concept", "")
                    chapter = item.get("chapter", "")
                    book = item.get("book", "")
                    formatted += f"- Concept: \"{concept}\" → Chapter: \"{chapter}\" (Book: {book})\n"
            
            # Textbooks results
            if evidence.get("textbooks"):
                formatted += "\n**TEXTBOOKS/BOOKS** (JSON Sources):\n"
                for item in evidence["textbooks"][:3]:
                    book = item.get("book", "Unknown")
                    chapter = item.get("chapter", "")
                    source = item.get("source", "textbook")
                    relevance = item.get("relevance", 0)
                    content = item.get("content", "")[:150]
                    formatted += f"- [{source}] \"{book}\" Ch: \"{chapter}\" (rel: {relevance:.2f})\n"
                    if content:
                        formatted += f"  > {content}...\n"
            
            # Code-Orchestrator results
            if evidence.get("code_orchestrator"):
                formatted += "\n**CODE-ORCHESTRATOR ML PIPELINE**:\n"
                for item in evidence["code_orchestrator"][:3]:
                    model = item.get("model", "")
                    item_type = item.get("type", "")
                    if item_type == "keywords":
                        keywords = item.get("keywords", [])[:7]
                        formatted += f"- [{model}] Keywords: {', '.join(str(k) for k in keywords)}\n"
                    elif item_type == "search":
                        book = item.get("book", "")
                        score = item.get("relevance_score", 0)
                        formatted += f"- [{model}] Book: \"{book}\" (score: {score:.3f})\n"
                    elif item_type == "term_extraction":
                        terms = item.get("terms", [])[:5]
                        formatted += f"- [{model}] Terms: {', '.join(str(t) for t in terms)}\n"
                    else:
                        content = item.get("content", str(item))[:100]
                        formatted += f"- [{model}] {content}\n"
            
            formatted += "\n"
        
        return formatted[:5000]  # Limit total size
    
    def load_recommendations(self) -> Optional[Dict]:
        """Load brigade recommendations configuration."""
        if RECOMMENDATIONS_FILE.exists():
            return yaml.safe_load(RECOMMENDATIONS_FILE.read_text())
        return None
    
    def get_endpoint_for_model(self, model_id: str) -> str:
        """Route model to correct endpoint (local inference vs external gateway)."""
        if model_id in EXTERNAL_MODELS:
            return LLM_GATEWAY
        return INFERENCE_SERVICE
    
    def detect_scenario(self) -> str:
        """Detect scenario type from inputs for recommendation selection."""
        inputs_str = json.dumps(self.inputs).lower()
        
        if any(k in inputs_str for k in ["architecture", "reconcil", "design"]):
            return "architecture_discussion"
        elif any(k in inputs_str for k in ["code", "review", "pr", "pull request"]):
            return "code_review"
        elif any(k in inputs_str for k in ["wbs", "work breakdown", "task", "project"]):
            return "wbs_generation"
        elif any(k in inputs_str for k in ["document", "enhance", "refine"]):
            return "document_reconciliation"
        elif any(k in inputs_str for k in ["external", "validate", "expert"]):
            return "external_validation"
        else:
            return "quick_analysis"
    
    def get_recommendation(self, scenario: str, tier: str = "balanced") -> Dict:
        """Get recommended brigade configuration for a scenario."""
        if not self.recommendations:
            return {}
        
        scenarios = self.recommendations.get("scenarios", {})
        if scenario not in scenarios:
            return {}
        
        return scenarios[scenario].get(tier, scenarios[scenario].get("local_only", {}))
    
    async def prompt_user_decision(self) -> Dict[str, Any]:
        """Interactive mode: Ask user for configuration decisions."""
        console.print(Panel.fit(
            "[bold cyan]Kitchen Brigade Configuration[/bold cyan]\n"
            "[dim]Let's configure how to run this protocol[/dim]",
            border_style="cyan"
        ))
        
        # Detect scenario and show recommendation
        scenario = self.detect_scenario()
        recommendation = self.get_recommendation(scenario, "balanced")
        
        console.print(f"\n[yellow]Detected scenario:[/yellow] {scenario.replace('_', ' ').title()}")
        
        if recommendation:
            console.print(f"[green]Recommendation:[/green] {recommendation.get('description', 'N/A')}")
            if "brigade" in recommendation:
                table = Table(title="Recommended Brigade")
                table.add_column("Role", style="cyan")
                table.add_column("Model", style="green")
                table.add_column("Endpoint", style="dim")
                
                for role, model in recommendation["brigade"].items():
                    endpoint = "external" if model in EXTERNAL_MODELS else "local"
                    table.add_row(role, model, endpoint)
                
                console.print(table)
        
        # Protocol type selection
        protocol_types = ["round_table", "design_review", "debate", "pipeline"]
        console.print("\n[bold]Protocol Types:[/bold]")
        console.print("  1. [cyan]round_table[/cyan] - All LLMs participate every round (exploration)")
        console.print("  2. [cyan]design_review[/cyan] - Structured with synthesis phases (validation)")
        console.print("  3. [cyan]debate[/cyan] - Adversarial discussion (decision-making)")
        console.print("  4. [cyan]pipeline[/cyan] - Sequential handoff (code generation)")
        
        choice = Prompt.ask(
            "\nSelect protocol type",
            choices=["1", "2", "3", "4", "default", "you_decide"],
            default="default"
        )
        
        if choice == "you_decide":
            # Agent decides based on scenario
            protocol_map = {
                "architecture_discussion": "round_table",
                "code_review": "design_review",
                "wbs_generation": "pipeline",
                "document_reconciliation": "design_review",
                "external_validation": "debate",
                "quick_analysis": "design_review"
            }
            selected_protocol = protocol_map.get(scenario, "design_review")
            console.print(f"[green]Auto-selected:[/green] {selected_protocol}")
        elif choice == "default":
            selected_protocol = self.protocol_id
        else:
            selected_protocol = protocol_types[int(choice) - 1]
        
        # Brigade tier selection
        console.print("\n[bold]Brigade Options:[/bold]")
        console.print("  1. [cyan]premium[/cyan] - External LLMs (Claude, GPT - API costs)")
        console.print("  2. [cyan]balanced[/cyan] - Mix of external + local")
        console.print("  3. [cyan]local_only[/cyan] - All local models (zero cost)")
        console.print("  4. [cyan]custom[/cyan] - Specify your own models")
        
        tier_choice = Prompt.ask(
            "\nSelect brigade tier",
            choices=["1", "2", "3", "4", "default"],
            default="default"
        )
        
        brigade_config = None
        if tier_choice == "4":
            # Custom mode - ask for each role
            brigade_config = await self._prompt_custom_brigade()
        elif tier_choice != "default":
            tier_map = {
                "1": "premium",
                "2": "balanced", 
                "3": "local_only"
            }
            tier = tier_map.get(tier_choice, "balanced")
            rec = self.get_recommendation(scenario, tier)
            if rec:
                brigade_config = rec.get("brigade")
        
        # Confirmation
        console.print("\n[bold]Configuration Summary:[/bold]")
        console.print(f"  Protocol: [cyan]{selected_protocol}[/cyan]")
        console.print(f"  Scenario: [cyan]{scenario}[/cyan]")
        if brigade_config:
            console.print(f"  Brigade: [cyan]{list(brigade_config.keys())}[/cyan]")
        
        if not Confirm.ask("\nProceed with this configuration?", default=True):
            console.print("[yellow]Cancelled by user[/yellow]")
            return {"cancelled": True}
        
        return {
            "protocol": selected_protocol,
            "scenario": scenario,
            "brigade_override": brigade_config,
            "cancelled": False
        }
    
    async def _prompt_custom_brigade(self) -> Dict[str, str]:
        """Prompt user to specify custom brigade models."""
        console.print("\n[bold]Custom Brigade Configuration[/bold]")
        console.print("[dim]Enter model ID for each role (or 'skip' to use default)[/dim]")
        
        available = []
        if self.recommendations:
            local = self.recommendations.get("available_models", {}).get("local", {})
            external = self.recommendations.get("available_models", {}).get("external", {})
            available = list(local.keys()) + list(external.keys())
            console.print(f"\n[dim]Available: {', '.join(available[:10])}...[/dim]")
        
        roles = ["analyst", "critic", "synthesizer", "validator"]
        brigade = {}
        
        for role in roles:
            model = Prompt.ask(f"  {role}", default="skip")
            if model != "skip":
                brigade[role] = model
        
        return brigade if brigade else None
    
    def load_protocol(self) -> Dict:
        """Load protocol definition from JSON."""
        protocol_file = PROTOCOLS_DIR / f"{self.protocol_id}.json"
        if not protocol_file.exists():
            raise FileNotFoundError(f"Protocol not found: {protocol_file}")
        
        return json.loads(protocol_file.read_text())
    
    def load_trace(self, trace_file: str) -> Dict:
        """Load existing trace file for resumption."""
        trace_path = Path(trace_file)
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file}")
        
        return json.loads(trace_path.read_text())
    
    async def execute(self, max_feedback_loops: int = 3, allow_feedback: bool = True,
                      run_cross_reference: bool = True) -> Dict[str, Any]:
        """Execute the protocol with feedback loop support.
        
        Args:
            max_feedback_loops: Maximum feedback iterations per round
            allow_feedback: Whether to prompt for feedback
            run_cross_reference: Whether to run Stage 2 cross-reference before Round 1
        """
        
        console.print(Panel.fit(
            f"[bold cyan]{self.protocol['name']}[/bold cyan]\n"
            f"[dim]{self.protocol['description']}[/dim]",
            border_style="cyan"
        ))
        
        # Stage 2: Cross-Reference (4-layer parallel retrieval)
        if run_cross_reference and self.enable_cross_reference:
            await self.run_cross_reference()
        
        # Execute each round (skip already completed rounds if resuming)
        round_outputs = self.resumed_outputs.copy()
        feedback_loop_count = 0
        
        for round_def in self.protocol["rounds"]:
            round_num = round_def["round"]
            round_type = round_def["type"]
            round_key = f"round_{round_num}"
            
            # Skip rounds that already completed successfully
            if round_key in round_outputs and not self._round_has_errors(round_outputs[round_key]):
                console.print(f"\n[dim]━━━ Round {round_num}: {round_type.upper()} (already completed) ━━━[/dim]\n")
                continue
            
            console.print(f"\n[bold yellow]━━━ Round {round_num}: {round_type.upper()} ━━━[/bold yellow]\n")
            
            if round_type == "parallel":
                output = await self.execute_parallel_round(round_def, round_outputs)
            elif round_type == "synthesis" or round_type == "consensus":
                output = await self.execute_synthesis_round(round_def, round_outputs)
            else:
                raise ValueError(f"Unknown round type: {round_type}")
            
            round_outputs[round_key] = output
            self.trace["rounds"].append({
                "round": round_num,
                "type": round_type,
                "output": output
            })
            
            # Check for feedback needs after each round
            if allow_feedback and feedback_loop_count < max_feedback_loops:
                feedback_needs = self.check_feedback_needs(output)
                if feedback_needs:
                    user_decision = await self.prompt_feedback_decision(feedback_needs, round_num)
                    if user_decision.get("run_additional"):
                        feedback_loop_count += 1
                        console.print(f"[cyan]Running feedback loop {feedback_loop_count}/{max_feedback_loops}[/cyan]")
                        
                        # Inject additional context if provided
                        if user_decision.get("additional_context"):
                            self.inputs["_feedback_context"] = user_decision["additional_context"]
                        
                        # Re-run this round with expanded context
                        if round_type == "parallel":
                            output = await self.execute_parallel_round(round_def, round_outputs)
                        else:
                            output = await self.execute_synthesis_round(round_def, round_outputs)
                        
                        round_outputs[round_key] = output
                        self.trace["rounds"].append({
                            "round": round_num,
                            "type": f"{round_type}_feedback_{feedback_loop_count}",
                            "output": output
                        })
        
        self.trace["end_time"] = datetime.now().isoformat()
        self.trace["feedback_loops"] = feedback_loop_count
        
        # Final feedback check
        final_feedback = self.check_feedback_needs(round_outputs.get(f"round_{len(self.protocol['rounds'])}", {}))
        
        return {
            "outputs": round_outputs,
            "trace": self.trace,
            "needs_more_work": final_feedback,
            "feedback_loops_used": feedback_loop_count
        }
    
    def check_feedback_needs(self, round_output: Dict) -> List[Dict]:
        """Check if LLM outputs indicate need for additional research/rounds."""
        feedback_needs = []
        
        for role, result in round_output.items():
            if not isinstance(result, dict):
                continue
            
            parsed = result.get("parsed", {})
            if not parsed:
                continue
            
            # Check for explicit research requests
            if parsed.get("requires_additional_research"):
                for item in parsed["requires_additional_research"]:
                    feedback_needs.append({
                        "type": "additional_research",
                        "from_role": role,
                        "topic": item if isinstance(item, str) else item.get("topic", str(item)),
                        "reason": item.get("reason", "Requested by LLM") if isinstance(item, dict) else "Requested by LLM"
                    })
            
            # Check for unresolved conflicts
            if parsed.get("unresolved_conflicts") or parsed.get("remaining_disagreements"):
                conflicts = parsed.get("unresolved_conflicts", parsed.get("remaining_disagreements", []))
                for conflict in conflicts:
                    feedback_needs.append({
                        "type": "unresolved_conflict",
                        "from_role": role,
                        "topic": conflict if isinstance(conflict, str) else str(conflict),
                        "reason": "Conflict not resolved in this round"
                    })
            
            # Check for low confidence
            confidence = parsed.get("confidence_level", "").lower()
            if confidence in ["low", "very_low"]:
                feedback_needs.append({
                    "type": "low_confidence",
                    "from_role": role,
                    "topic": "Overall confidence",
                    "reason": f"LLM reported {confidence} confidence in conclusions"
                })
            
            # Check for explicit "needs more" flags
            if parsed.get("needs_cross_reference") or parsed.get("needs_more_context"):
                items = parsed.get("needs_cross_reference", parsed.get("needs_more_context", []))
                for item in (items if isinstance(items, list) else [items]):
                    feedback_needs.append({
                        "type": "needs_context",
                        "from_role": role,
                        "topic": item if isinstance(item, str) else str(item),
                        "reason": "Additional context requested"
                    })
            
            # Check remaining questions
            if parsed.get("remaining_questions"):
                for q in parsed["remaining_questions"][:3]:  # Limit to top 3
                    feedback_needs.append({
                        "type": "open_question",
                        "from_role": role,
                        "topic": q if isinstance(q, str) else str(q),
                        "reason": "Question remains unanswered"
                    })
        
        return feedback_needs
    
    async def prompt_feedback_decision(self, feedback_needs: List[Dict], round_num: int) -> Dict[str, Any]:
        """Prompt user for decision on feedback loop."""
        console.print("\n")
        console.print(Panel.fit(
            f"[bold yellow]Feedback Needed After Round {round_num}[/bold yellow]\n"
            f"[dim]LLMs have identified {len(feedback_needs)} items requiring attention[/dim]",
            border_style="yellow"
        ))
        
        # Show feedback items in table
        table = Table(title="Feedback Items")
        table.add_column("Type", style="cyan")
        table.add_column("From", style="green")
        table.add_column("Topic", style="white")
        table.add_column("Reason", style="dim")
        
        for item in feedback_needs[:10]:  # Show max 10
            table.add_row(
                item["type"].replace("_", " ").title(),
                item["from_role"],
                item["topic"][:50] + "..." if len(item["topic"]) > 50 else item["topic"],
                item["reason"][:40] + "..." if len(item["reason"]) > 40 else item["reason"]
            )
        
        console.print(table)
        
        if len(feedback_needs) > 10:
            console.print(f"[dim]...and {len(feedback_needs) - 10} more items[/dim]")
        
        console.print("\n[bold]Options:[/bold]")
        console.print("  1. [cyan]run[/cyan] - Run additional round with current context")
        console.print("  2. [cyan]add[/cyan] - Add context and run additional round")
        console.print("  3. [cyan]skip[/cyan] - Continue without additional round")
        console.print("  4. [cyan]stop[/cyan] - Stop execution here")
        
        choice = Prompt.ask(
            "\nHow to proceed?",
            choices=["run", "add", "skip", "stop", "1", "2", "3", "4"],
            default="skip"
        )
        
        choice_map = {"1": "run", "2": "add", "3": "skip", "4": "stop"}
        choice = choice_map.get(choice, choice)
        
        if choice == "stop":
            console.print("[red]Execution stopped by user[/red]")
            raise KeyboardInterrupt("User stopped execution")
        
        if choice == "skip":
            return {"run_additional": False}
        
        additional_context = None
        if choice == "add":
            console.print("\n[bold]Add Context[/bold]")
            console.print("[dim]Enter additional context, document paths, or guidance (empty to skip):[/dim]")
            additional_context = Prompt.ask("Context", default="")
            if not additional_context:
                additional_context = None
        
        return {
            "run_additional": True,
            "additional_context": additional_context,
            "feedback_items": feedback_needs
        }
    
    def _round_has_errors(self, round_output: Dict) -> bool:
        """Check if a round has any errors in its outputs."""
        for result in round_output.values():
            if isinstance(result, dict) and "error" in result:
                return True
        return False
    
    async def execute_parallel_round(self, round_def: Dict, prev_outputs: Dict) -> Dict:
        """Execute a parallel round where multiple LLMs run simultaneously."""
        
        roles = round_def["roles"]
        prompt_template = self.load_prompt_template(round_def["prompt_template"])
        
        tasks = []
        for role in roles:
            tasks.append(self.call_llm(role, prompt_template, prev_outputs))
        
        results = await asyncio.gather(*tasks)
        
        return {role: result for role, result in zip(roles, results)}
    
    async def execute_synthesis_round(self, round_def: Dict, prev_outputs: Dict) -> Dict:
        """Execute a synthesis round where one LLM synthesizes previous outputs."""
        
        role = round_def["role"]
        prompt_template = self.load_prompt_template(round_def["prompt_template"])
        
        result = await self.call_llm(role, prompt_template, prev_outputs)
        
        return {role: result}
    
    def load_prompt_template(self, template_name: str) -> str:
        """Load prompt template from file."""
        template_file = PROMPTS_DIR / template_name
        if not template_file.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_file}")
        
        return template_file.read_text()
    
    async def call_llm(self, role: str, prompt_template: str, prev_outputs: Dict) -> Dict:
        """Call LLM with role-specific config, routing to correct endpoint."""
        
        role_config = self.protocol["brigade_roles"][role]
        
        # Check for brigade override
        model = role_config["model"]
        if self.brigade_override and role in self.brigade_override:
            model = self.brigade_override[role]
        
        # Route to correct endpoint
        endpoint = self.get_endpoint_for_model(model)
        
        # Build prompt from template with variable substitution
        prompt = self.build_prompt(prompt_template, role, prev_outputs)
        
        endpoint_label = "[external]" if endpoint == LLM_GATEWAY else "[local]"
        console.print(f"[cyan]{role.capitalize()}[/cyan] {endpoint_label} thinking with {model}...")
        
        try:
            response = await self.client.post(
                f"{endpoint}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": role_config["system_prompt"]},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": role_config.get("temperature", 0.3),
                    "max_tokens": role_config.get("max_tokens", 4096),
                }
            )
            response.raise_for_status()
            result = response.json()
            
            content = result["choices"][0]["message"]["content"]
            parsed = self.extract_json(content)
            
            console.print(f"✓ [green]{role.capitalize()}[/green] completed via {endpoint_label}")
            
            return {
                "role": role,
                "model": model,
                "endpoint": endpoint,
                "content": content,
                "parsed": parsed,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            console.print(f"✗ [red]{role.capitalize()} failed: {e}[/red]")
            return {"role": role, "error": str(e)}
    
    def build_prompt(self, template: str, role: str, prev_outputs: Dict) -> str:
        """Build prompt from template with variable substitution."""
        
        # Available variables for substitution
        variables = {
            "role": role,
            "inputs": json.dumps(self.inputs, indent=2),
            "previous_rounds": json.dumps(prev_outputs, indent=2),
        }
        
        # Add input documents if provided
        if "documents" in self.inputs:
            doc_content = ""
            for doc_path in self.inputs["documents"]:
                path = Path(doc_path)
                if path.exists():
                    doc_content += f"\n\n## {path.name}\n```markdown\n{path.read_text()[:2000]}...\n```"
            variables["documents"] = doc_content
        
        # Add cross-reference evidence from Stage 2 (if available)
        if self.cross_reference_evidence:
            variables["cross_reference_evidence"] = self.format_evidence_for_prompt()
            # Also inject into inputs for template access
            variables["inputs"] = json.dumps({
                **self.inputs,
                "_cross_reference_summary": f"Stage 2 retrieved {len(self.cross_reference_evidence)} query results from Qdrant, Neo4j, Textbooks, Code-Orchestrator"
            }, indent=2)
        
        # Add user feedback if injected
        if "_feedback_context" in self.inputs:
            variables["user_feedback"] = self.inputs["_feedback_context"]
        
        # Add previous stage outputs for workflows
        if "_previous_stage_summary" in self.inputs:
            variables["previous_stage_summary"] = self.inputs["_previous_stage_summary"]
        
        # Simple template substitution (in production, use jinja2)
        prompt = template
        for key, value in variables.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
        
        # If cross-reference evidence exists but template doesn't have placeholder, append it
        if self.cross_reference_evidence and "{{cross_reference_evidence}}" not in template:
            evidence_section = self.format_evidence_for_prompt()
            if evidence_section:
                prompt += f"\n\n{evidence_section}"
        
        return prompt
    
    def extract_json(self, content: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        import re
        
        # Try markdown code blocks
        matches = re.findall(r'```(?:json)?\s*\n(.*?)\n```', content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # Try finding JSON object
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                return json.loads(content[start:end+1])
        except json.JSONDecodeError:
            pass
        
        return None


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Execute Kitchen Brigade protocol")
    parser.add_argument("--protocol", required=True, help="Protocol ID")
    parser.add_argument("--input", action="append", help="Input key=value pairs")
    parser.add_argument("--resume", help="Resume from existing trace file")
    parser.add_argument("--interactive", "-i", action="store_true", 
                        help="Interactive mode: prompt for configuration")
    parser.add_argument("--tier", choices=["premium", "balanced", "local_only"],
                        help="Brigade tier (overrides protocol defaults)")
    parser.add_argument("--brigade", help="JSON string of role:model mappings")
    parser.add_argument("--no-cross-reference", action="store_true",
                        help="Disable Stage 2 cross-reference retrieval")
    
    args = parser.parse_args()
    
    # Parse inputs
    inputs = {}
    if args.input:
        for inp in args.input:
            key, value = inp.split("=", 1)
            # Try to parse as JSON, otherwise use as string
            try:
                inputs[key] = json.loads(value)
            except json.JSONDecodeError:
                inputs[key] = value
    
    # Parse brigade override
    brigade_override = None
    if args.brigade:
        try:
            brigade_override = json.loads(args.brigade)
        except json.JSONDecodeError:
            console.print(f"[red]Invalid --brigade JSON: {args.brigade}[/red]")
            return
    
    executor = KitchenBrigadeExecutor(
        args.protocol, 
        inputs, 
        resume_from=args.resume,
        brigade_override=brigade_override,
        enable_cross_reference=not args.no_cross_reference
    )
    
    # Interactive mode
    if args.interactive:
        config = await executor.prompt_user_decision()
        if config.get("cancelled"):
            return {"cancelled": True}
        
        # Apply user decisions
        if config.get("brigade_override"):
            executor.brigade_override = config["brigade_override"]
        
        # If user selected different protocol, reload
        if config.get("protocol") and config["protocol"] != args.protocol:
            console.print(f"[yellow]Switching to protocol: {config['protocol']}[/yellow]")
            executor.protocol_id = config["protocol"]
            executor.protocol = executor.load_protocol()
    
    # Apply tier if specified via CLI
    if args.tier and not brigade_override:
        scenario = executor.detect_scenario()
        rec = executor.get_recommendation(scenario, args.tier)
        if rec and "brigade" in rec:
            console.print(f"[cyan]Using {args.tier} tier for {scenario}[/cyan]")
            executor.brigade_override = rec["brigade"]
    
    result = await executor.execute()
    
    # Save trace
    trace_file = Path(f"trace_{args.protocol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    trace_file.write_text(json.dumps(result, indent=2))
    
    console.print(f"\n✓ Trace saved to {trace_file}")
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
