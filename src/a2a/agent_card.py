"""A2A Agent Card Generation.

WBS-PI2: A2A Agent Card & Discovery - Agent Card Generation

Implements:
- generate_agent_card(): Build AgentCard from FUNCTION_REGISTRY

Acceptance Criteria:
- AC-PI2.3: generate_agent_card() builds card from AgentFunctionRegistry
- AC-PI2.6: Card includes protocolVersion: "0.3.0"
- AC-PI2.7: Card includes capabilities.streaming based on feature flag

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → Agent Card Schema
"""

from src.a2a.models import AgentCard, Capability, Skill
from src.api.routes.functions import FUNCTION_REGISTRY
from src.config.feature_flags import ProtocolFeatureFlags, get_feature_flags


# =============================================================================
# Skill Metadata Mapping
# =============================================================================

# Maps function names to their metadata for skill generation
SKILL_METADATA = {
    "extract-structure": {
        "id": "extract_structure",
        "name": "Extract Structure",
        "description": "Extract structured data (keywords, concepts, entities, outline) from unstructured content",
        "tags": ["extraction", "nlp", "structure"],
        "examples": [
            "Extract keywords from this chapter",
            "Identify the main concepts in this document",
        ],
    },
    "summarize-content": {
        "id": "summarize_content",
        "name": "Summarize Content",
        "description": "Compress content while preserving key invariants",
        "tags": ["summarization", "compression", "nlp"],
        "examples": [
            "Summarize this chapter in 500 words",
            "Create an executive summary preserving these key points",
        ],
    },
    "generate-code": {
        "id": "generate_code",
        "name": "Generate Code",
        "description": "Generate code from specification with context awareness",
        "tags": ["code", "generation", "development"],
        "examples": [
            "Generate a Python class implementing the Repository pattern",
            "Create a FastAPI endpoint for user authentication",
        ],
    },
    "analyze-artifact": {
        "id": "analyze_artifact",
        "name": "Analyze Artifact",
        "description": "Analyze code or documents for quality, security, and patterns",
        "tags": ["analysis", "quality", "security"],
        "examples": [
            "Analyze this code for security vulnerabilities",
            "Check code quality and complexity metrics",
        ],
    },
    "validate-against-spec": {
        "id": "validate_against_spec",
        "name": "Validate Against Spec",
        "description": "Validate artifact meets specification and acceptance criteria",
        "tags": ["validation", "verification", "qa"],
        "examples": [
            "Verify this code implements the specification",
            "Check if the summary preserves all required invariants",
        ],
    },
    "decompose-task": {
        "id": "decompose_task",
        "name": "Decompose Task",
        "description": "Break complex task into executable subtasks",
        "tags": ["planning", "decomposition", "workflow"],
        "examples": [
            "Break down this feature request into implementation steps",
            "Create a task plan for refactoring this module",
        ],
    },
    "synthesize-outputs": {
        "id": "synthesize_outputs",
        "name": "Synthesize Outputs",
        "description": "Combine multiple artifacts into coherent result",
        "tags": ["synthesis", "merge", "reconciliation"],
        "examples": [
            "Merge these code snippets into a single module",
            "Reconcile conflicting analysis results",
        ],
    },
    "cross-reference": {
        "id": "cross_reference",
        "name": "Cross Reference",
        "description": "Find related content across knowledge bases via semantic search",
        "tags": ["search", "retrieval", "reference"],
        "examples": [
            "Find similar implementations in our codebase",
            "Search for related patterns in the reference library",
        ],
    },
}


# =============================================================================
# AC-PI2.3: Agent Card Generation
# =============================================================================


def generate_agent_card(
    flags: ProtocolFeatureFlags | None = None,
) -> AgentCard:
    """Generate A2A Agent Card from function registry.
    
    Builds an AgentCard manifest declaring service capabilities and skills
    for A2A protocol discovery (AC-PI2.3).
    
    Args:
        flags: Optional ProtocolFeatureFlags for capability configuration.
               Uses default flags if not provided.
    
    Returns:
        AgentCard with service metadata, capabilities, and skills
    
    Example:
        >>> card = generate_agent_card()
        >>> assert card.protocolVersion == "0.3.0"
        >>> assert len(card.skills) == 8
    """
    # Use default flags if not provided
    if flags is None:
        flags = get_feature_flags()
    
    # Build capabilities based on feature flags (AC-PI2.7)
    capabilities = Capability(
        streaming=flags.a2a_streaming_enabled,
        pushNotifications=flags.a2a_push_notifications,
        stateTransitionHistory=True,  # Always enabled
    )
    
    # Generate skills from function registry
    skills = []
    for func_name in FUNCTION_REGISTRY:
        if func_name in SKILL_METADATA:
            metadata = SKILL_METADATA[func_name]
            skill = Skill(
                id=metadata["id"],
                name=metadata["name"],
                description=metadata["description"],
                tags=metadata["tags"],
                examples=metadata["examples"],
                inputModes=["text/plain", "application/json"],
                outputModes=["application/json"],
            )
            skills.append(skill)
    
    # Build agent card (AC-PI2.6: protocolVersion defaults to "0.3.0")
    card = AgentCard(
        name="ai-agents-service",
        description="AI Platform Agent Functions Service - Pipeline orchestration and agent function execution",
        version="1.0.0",
        capabilities=capabilities,
        skills=skills,
    )
    
    return card


__all__ = ["generate_agent_card"]
