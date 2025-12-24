"""
Tool Participant Adapter - Calls BERT tools via Code-Orchestrator-Service.

This adapter routes tool requests to the Code-Orchestrator-Service,
which hosts BERT models (BERTopic, SBERT, GraphCodeBERT, etc.).

Reference: docs/INTER_AI_ORCHESTRATION.md
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import httpx

from src.conversation.models import Conversation, Participant
from src.participants.base import BaseParticipant

logger = logging.getLogger(__name__)


class ToolParticipantAdapter(BaseParticipant):
    """Adapter for BERT tool participants via Code-Orchestrator-Service.
    
    Routes tool calls to appropriate endpoints on the orchestrator service.
    
    Supported tools:
    - bertopic: Topic modeling and clustering
    - sbert: Semantic similarity computation
    - graphcodebert: Code-aware term validation
    - codet5: Code understanding
    - hdbscan: Density-based clustering
    - concept_validator: Concept vs keyword filtering
    
    Attributes:
        orchestrator_url: URL of the Code-Orchestrator-Service.
        timeout: Request timeout in seconds.
    """
    
    # Tool ID to endpoint mapping (matches Code-Orchestrator-Service routes)
    TOOL_ENDPOINTS = {
        "bertopic": "/api/v1/topics",           # BERTopic clustering
        "sbert": "/v1/similarity",              # SBERT similarity
        "graphcodebert": "/api/v1/codebert/similarity",  # Code similarity
        "codet5": "/api/v1/codebert/embed",     # Code embeddings
        "hdbscan": "/api/v1/cluster",           # HDBSCAN clustering
        "concept_validator": "/api/v1/concepts", # Concept extraction
    }
    
    def __init__(
        self,
        orchestrator_url: str = "http://localhost:8083",
        timeout: float = 60.0,
    ) -> None:
        """Initialize the tool participant adapter.
        
        Args:
            orchestrator_url: URL of Code-Orchestrator-Service.
            timeout: Request timeout in seconds.
        """
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    async def respond(
        self,
        conversation: Conversation,
        participant: Participant,
    ) -> dict[str, Any]:
        """Get analysis from a BERT tool.
        
        Routes to the appropriate tool endpoint based on participant ID.
        
        Args:
            conversation: Current conversation state.
            participant: The tool participant definition.
            
        Returns:
            Dict with 'content', 'latency_ms', 'metadata'.
        """
        start_time = datetime.utcnow()
        tool_id = participant.id
        
        # Get endpoint for this tool
        endpoint = self.TOOL_ENDPOINTS.get(tool_id)
        if not endpoint:
            return {
                "content": f"Unknown tool: {tool_id}. Available: {list(self.TOOL_ENDPOINTS.keys())}",
                "latency_ms": 0,
                "metadata": {"error": "unknown_tool"},
            }
        
        # Build request based on tool type
        request_body = self._build_tool_request(conversation, participant)
        
        try:
            response = await self._client.post(
                f"{self.orchestrator_url}{endpoint}",
                json=request_body,
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Format response as conversational content
            content = self._format_tool_response(tool_id, data)
            
            elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return {
                "content": content,
                "tokens_used": None,  # Tools don't use tokens
                "latency_ms": elapsed_ms,
                "metadata": {
                    "tool": tool_id,
                    "endpoint": endpoint,
                    "raw_response": data,
                },
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Tool error: {e.response.status_code} - {e.response.text}")
            elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return {
                "content": f"Error calling {tool_id}: {e.response.text}",
                "latency_ms": elapsed_ms,
                "metadata": {"error": str(e)},
            }
        except Exception as e:
            logger.error(f"Error calling tool {tool_id}: {e}")
            elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return {
                "content": f"Error: {str(e)}",
                "latency_ms": elapsed_ms,
                "metadata": {"error": str(e)},
            }
    
    def _build_tool_request(
        self,
        conversation: Conversation,
        participant: Participant,
    ) -> dict[str, Any]:
        """Build request body for the tool.
        
        Args:
            conversation: Current conversation.
            participant: The tool participant.
            
        Returns:
            Request body dictionary.
        """
        tool_id = participant.id
        context = conversation.context
        
        # Get terms from various possible field names
        terms = (
            context.get("terms", []) or
            context.get("top_terms_for_analysis", []) or
            context.get("all_terms", [])
        )
        
        if tool_id == "bertopic":
            # BERTopic/cluster endpoint expects corpus (list of docs) and chapter_index
            # Convert terms to documents for clustering
            return {
                "corpus": terms if terms else ["placeholder"],
                "chapter_index": list(range(len(terms))) if terms else [0],
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            }
        
        elif tool_id == "sbert":
            # Get last cluster/terms to compute similarity
            return {
                "texts": terms[:100] if terms else [],
            }
        
        elif tool_id == "graphcodebert":
            return {
                "terms": terms,
                "threshold": 0.28,
            }
        
        elif tool_id == "concept_validator":
            return {
                "terms": terms,
                "min_word_count": 2,
            }
        
        elif tool_id == "hdbscan":
            return {
                "embeddings": context.get("embeddings", []),
                "min_cluster_size": 5,
            }
        
        else:
            # Generic request with all context
            return {"context": context}
    
    def _format_tool_response(
        self,
        tool_id: str,
        data: dict[str, Any],
    ) -> str:
        """Format tool response as conversational content.
        
        Args:
            tool_id: The tool that produced the response.
            data: Raw response data.
            
        Returns:
            Formatted string for conversation.
        """
        if tool_id == "bertopic":
            clusters = data.get("concepts", data.get("clusters", []))
            noise = data.get("noise_terms", [])
            
            lines = [
                f"BERTopic Analysis Complete:",
                f"- Discovered {len(clusters)} clusters",
                f"- {len(noise)} terms classified as noise (keywords)",
                "",
                "Top clusters:",
            ]
            
            for cluster in clusters[:10]:
                name = cluster.get("name", "Unknown")
                terms = cluster.get("representative_terms", [])[:5]
                score = cluster.get("quality_score", 0)
                lines.append(f"  [{score:.2f}] {name}: {terms}")
            
            return "\n".join(lines)
        
        elif tool_id == "sbert":
            scores = data.get("similarity_scores", data.get("scores", []))
            return f"SBERT Similarity Analysis:\n- Computed {len(scores)} similarity scores\n- Average: {sum(scores)/len(scores) if scores else 0:.3f}"
        
        elif tool_id == "graphcodebert":
            validated = data.get("validated_terms", [])
            rejected = data.get("rejected_terms", [])
            return f"GraphCodeBERT Validation:\n- {len(validated)} terms validated as code-related\n- {len(rejected)} terms rejected"
        
        elif tool_id == "concept_validator":
            concepts = data.get("concepts", [])
            keywords = data.get("keywords", [])
            return f"Concept Validator:\n- {len(concepts)} true concepts identified\n- {len(keywords)} classified as keywords"
        
        else:
            # Generic JSON formatting
            return f"{tool_id} Response:\n{json.dumps(data, indent=2, default=str)}"
    
    async def health_check(self) -> bool:
        """Check if Code-Orchestrator-Service is healthy.
        
        Returns:
            True if healthy, False otherwise.
        """
        try:
            response = await self._client.get(f"{self.orchestrator_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
