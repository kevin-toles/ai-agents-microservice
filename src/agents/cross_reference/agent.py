"""Cross-Reference Agent implementation.

LangGraph-based agent for taxonomy-aware cross-referencing.
Implements the 9-step workflow from TIER_RELATIONSHIP_DIAGRAM.md.

Pattern: LangGraph StateGraph Workflow
Source: ARCHITECTURE.md (ai-agents), TIER_RELATIONSHIP_DIAGRAM.md
"""

from typing import Any

from langgraph.graph import StateGraph, END

from src.agents.base import BaseAgent
from src.agents.cross_reference.state import (
    CrossReferenceState,
    SourceChapter,
    TraversalConfig,
    CrossReferenceResult,
    ChapterMatch,
    Citation,
    TierCoverage,
)
from src.agents.cross_reference.nodes import (
    analyze_source,
    search_taxonomy,
    traverse_graph,
    retrieve_content,
    synthesize,
)
from src.core.exceptions import AgentExecutionError, AgentValidationError


class CrossReferenceAgent(BaseAgent[dict, CrossReferenceResult]):
    """Cross-Reference Agent for taxonomy-aware scholarly annotation.
    
    This agent:
    1. Analyzes source chapter content and metadata
    2. Searches taxonomy for related books/chapters
    3. Traverses the graph following PARALLEL/PERPENDICULAR/SKIP_TIER edges
    4. Retrieves relevant chapter content for validation
    5. Synthesizes scholarly annotations with Chicago-style citations
    
    Workflow states (LangGraph):
    - analyze_source: Understand source chapter key concepts
    - search_taxonomy: Find related books in taxonomy
    - traverse_graph: Follow spider web paths
    - retrieve_content: Get relevant chapter texts
    - synthesize: Generate annotation with citations
    
    Source: ARCHITECTURE.md (ai-agents), TIER_RELATIONSHIP_DIAGRAM.md
    """
    
    def __init__(self) -> None:
        """Initialize Cross-Reference Agent."""
        self._name = "cross_reference_agent"
        self._workflow = self._build_workflow()
        self._compiled_workflow = self._workflow.compile()
    
    @property
    def name(self) -> str:
        """Return agent name."""
        return self._name
    
    @property
    def description(self) -> str:
        """Return agent description for tool registration."""
        return (
            "Cross-Reference Agent for taxonomy-aware scholarly annotation. "
            "Given a source chapter, traverses the spider web taxonomy to find "
            "related content across tiers and generates annotations with citations."
        )
    
    async def validate_input(self, input_data: dict) -> bool:
        """Validate input before execution.
        
        Args:
            input_data: Dict with book, chapter, title, tier, etc.
            
        Returns:
            True if input is valid, False otherwise
        """
        try:
            book = input_data.get("book")
            chapter = input_data.get("chapter")
            title = input_data.get("title")
            tier = input_data.get("tier")
            
            if not book:
                return False
            if not chapter or chapter < 1:
                return False
            if not title:
                return False
            if tier not in (1, 2, 3):
                return False
            return True
        except Exception:
            return False
    
    async def run(self, input_data: dict) -> CrossReferenceResult:
        """Execute the cross-reference workflow.
        
        Args:
            input_data: Dict with book, chapter, title, tier, content, keywords, config
            
        Returns:
            Cross-reference result with annotation and citations
            
        Raises:
            AgentExecutionError: If execution fails
        """
        # Validate input
        if not await self.validate_input(input_data):
            raise AgentValidationError(
                "Invalid input data",
                agent_name=self.name,
            )
        
        # Build SourceChapter
        source = SourceChapter(
            book=input_data["book"],
            chapter=input_data["chapter"],
            title=input_data["title"],
            tier=input_data["tier"],
            content=input_data.get("content"),
            keywords=input_data.get("keywords", []),
            concepts=input_data.get("concepts", []),
        )
        
        # Build TraversalConfig
        config_data = input_data.get("config", {})
        config = TraversalConfig(
            max_hops=config_data.get("max_hops", 3),
            min_similarity=config_data.get("min_similarity", 0.7),
            include_tier1=config_data.get("include_tier1", True),
            include_tier2=config_data.get("include_tier2", True),
            include_tier3=config_data.get("include_tier3", True),
        )
        
        # Initialize state
        state = CrossReferenceState(
            source=source,
            config=config,
        )
        
        # Execute workflow
        return await self._run_workflow(state)
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph StateGraph workflow.
        
        Returns:
            StateGraph with all nodes and edges configured
        """
        # Create StateGraph with CrossReferenceState
        workflow = StateGraph(CrossReferenceState)
        
        # Add nodes
        workflow.add_node("analyze_source", analyze_source)
        workflow.add_node("search_taxonomy", search_taxonomy)
        workflow.add_node("traverse_graph", traverse_graph)
        workflow.add_node("retrieve_content", retrieve_content)
        workflow.add_node("synthesize", synthesize)
        
        # Add edges (linear workflow)
        workflow.set_entry_point("analyze_source")
        workflow.add_edge("analyze_source", "search_taxonomy")
        workflow.add_edge("search_taxonomy", "traverse_graph")
        workflow.add_edge("traverse_graph", "retrieve_content")
        workflow.add_edge("retrieve_content", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow
    
    async def _run_workflow(
        self,
        state: CrossReferenceState,
    ) -> CrossReferenceResult:
        """Execute the compiled workflow.
        
        Args:
            state: Initial workflow state
            
        Returns:
            CrossReferenceResult from final state
            
        Raises:
            AgentExecutionError: If workflow fails
        """
        try:
            # Run the compiled workflow
            final_state = await self._compiled_workflow.ainvoke(state)
            
            # Extract result
            if isinstance(final_state, dict):
                result = final_state.get("result")
            else:
                result = final_state.result
            
            if result is None:
                # Create empty result if synthesis didn't produce one
                result = CrossReferenceResult(
                    annotation="No cross-references found.",
                    citations=[],
                    traversal_paths=[],
                    tier_coverage=[
                        TierCoverage(tier=1, tier_name="Architecture Spine"),
                        TierCoverage(tier=2, tier_name="Implementation"),
                        TierCoverage(tier=3, tier_name="Engineering Practices"),
                    ],
                    matches=[],
                    processing_time_ms=0.0,
                    model_used="",
                )
            
            return result
            
        except Exception as e:
            raise AgentExecutionError(
                f"Workflow execution failed: {e}",
                agent_name=self.name,
            ) from e
