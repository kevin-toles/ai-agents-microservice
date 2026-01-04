"""Tests for A2A protocol models.

TDD tests for WBS-PI2: A2A Agent Card & Discovery - Models.

Acceptance Criteria Coverage:
- AC-PI2.1: AgentCard pydantic model validates against A2A spec v0.3.0
- AC-PI2.2: Skill model maps to agent functions with id, name, description, tags
- AC-PI2.6: Card includes protocolVersion: "0.3.0"
- AC-PI2.7: Card includes capabilities.streaming based on feature flag
- AC-PI2.8: Skills include inputModes and outputModes

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → Agent Card Schema
"""

import pytest
from pydantic import ValidationError


# =============================================================================
# AC-PI2.2, AC-PI2.8: Skill Model
# =============================================================================


class TestSkillModel:
    """Tests for Skill pydantic model."""

    def test_skill_model_exists(self) -> None:
        """Skill model can be imported."""
        from src.a2a.models import Skill

        assert isinstance(Skill, type)

    def test_skill_model_required_fields(self) -> None:
        """Skill model requires id, name, description."""
        from src.a2a.models import Skill

        with pytest.raises(ValidationError):
            Skill()  # Missing required fields

    def test_skill_model_creates_with_minimal_data(self) -> None:
        """Skill model creates with minimal required fields."""
        from src.a2a.models import Skill

        skill = Skill(
            id="test_function",
            name="Test Function",
            description="A test function",
        )

        assert skill.id == "test_function"
        assert skill.name == "Test Function"
        assert skill.description == "A test function"

    def test_skill_model_has_tags_field(self) -> None:
        """Skill model has optional tags field."""
        from src.a2a.models import Skill

        skill = Skill(
            id="test_function",
            name="Test Function",
            description="A test function",
            tags=["extraction", "nlp"],
        )

        assert skill.tags == ["extraction", "nlp"]

    def test_skill_model_tags_defaults_to_empty_list(self) -> None:
        """Skill model tags defaults to empty list."""
        from src.a2a.models import Skill

        skill = Skill(
            id="test_function",
            name="Test Function",
            description="A test function",
        )

        assert skill.tags == []

    def test_skill_model_has_examples_field(self) -> None:
        """Skill model has optional examples field."""
        from src.a2a.models import Skill

        skill = Skill(
            id="test_function",
            name="Test Function",
            description="A test function",
            examples=["Example 1", "Example 2"],
        )

        assert skill.examples == ["Example 1", "Example 2"]

    def test_skill_model_examples_defaults_to_empty_list(self) -> None:
        """Skill model examples defaults to empty list."""
        from src.a2a.models import Skill

        skill = Skill(
            id="test_function",
            name="Test Function",
            description="A test function",
        )

        assert skill.examples == []

    def test_skill_model_has_input_modes(self) -> None:
        """Skill model has inputModes field (AC-PI2.8)."""
        from src.a2a.models import Skill

        skill = Skill(
            id="test_function",
            name="Test Function",
            description="A test function",
        )

        assert hasattr(skill, "inputModes")
        assert isinstance(skill.inputModes, list)

    def test_skill_model_input_modes_defaults_to_json(self) -> None:
        """Skill model inputModes defaults to application/json (AC-PI2.8)."""
        from src.a2a.models import Skill

        skill = Skill(
            id="test_function",
            name="Test Function",
            description="A test function",
        )

        assert "application/json" in skill.inputModes

    def test_skill_model_has_output_modes(self) -> None:
        """Skill model has outputModes field (AC-PI2.8)."""
        from src.a2a.models import Skill

        skill = Skill(
            id="test_function",
            name="Test Function",
            description="A test function",
        )

        assert hasattr(skill, "outputModes")
        assert isinstance(skill.outputModes, list)

    def test_skill_model_output_modes_defaults_to_json(self) -> None:
        """Skill model outputModes defaults to application/json (AC-PI2.8)."""
        from src.a2a.models import Skill

        skill = Skill(
            id="test_function",
            name="Test Function",
            description="A test function",
        )

        assert "application/json" in skill.outputModes

    def test_skill_model_custom_input_output_modes(self) -> None:
        """Skill model allows custom inputModes and outputModes."""
        from src.a2a.models import Skill

        skill = Skill(
            id="test_function",
            name="Test Function",
            description="A test function",
            inputModes=["text/plain", "application/json"],
            outputModes=["application/json", "text/markdown"],
        )

        assert skill.inputModes == ["text/plain", "application/json"]
        assert skill.outputModes == ["application/json", "text/markdown"]


# =============================================================================
# AC-PI2.7: Capability Model
# =============================================================================


class TestCapabilityModel:
    """Tests for Capability pydantic model."""

    def test_capability_model_exists(self) -> None:
        """Capability model can be imported."""
        from src.a2a.models import Capability

        assert isinstance(Capability, type)

    def test_capability_model_has_streaming_field(self) -> None:
        """Capability model has streaming boolean field (AC-PI2.7)."""
        from src.a2a.models import Capability

        capability = Capability(streaming=True)

        assert capability.streaming is True

    def test_capability_model_has_push_notifications_field(self) -> None:
        """Capability model has pushNotifications boolean field."""
        from src.a2a.models import Capability

        capability = Capability(pushNotifications=True)

        assert capability.pushNotifications is True

    def test_capability_model_has_state_transition_history_field(self) -> None:
        """Capability model has stateTransitionHistory boolean field."""
        from src.a2a.models import Capability

        capability = Capability(stateTransitionHistory=True)

        assert capability.stateTransitionHistory is True

    def test_capability_model_defaults_to_false(self) -> None:
        """Capability model fields default to False."""
        from src.a2a.models import Capability

        capability = Capability()

        assert capability.streaming is False
        assert capability.pushNotifications is False
        assert capability.stateTransitionHistory is False

    def test_capability_model_accepts_partial_values(self) -> None:
        """Capability model allows partial field specification."""
        from src.a2a.models import Capability

        capability = Capability(streaming=True)

        assert capability.streaming is True
        assert capability.pushNotifications is False
        assert capability.stateTransitionHistory is False


# =============================================================================
# AC-PI2.1, AC-PI2.6: AgentCard Model
# =============================================================================


class TestAgentCardModel:
    """Tests for AgentCard pydantic model."""

    def test_agent_card_model_exists(self) -> None:
        """AgentCard model can be imported (AC-PI2.1)."""
        from src.a2a.models import AgentCard

        assert isinstance(AgentCard, type)

    def test_agent_card_requires_core_fields(self) -> None:
        """AgentCard requires name, description, version, capabilities, skills."""
        from src.a2a.models import AgentCard

        with pytest.raises(ValidationError):
            AgentCard()  # Missing required fields

    def test_agent_card_protocol_version_defaults_to_0_3_0(self) -> None:
        """AgentCard protocolVersion defaults to 0.3.0 (AC-PI2.6)."""
        from src.a2a.models import AgentCard, Capability

        card = AgentCard(
            name="test-agent",
            description="Test agent",
            version="1.0.0",
            capabilities=Capability(),
            skills=[],
        )

        assert card.protocolVersion == "0.3.0"

    def test_agent_card_creates_with_minimal_data(self) -> None:
        """AgentCard creates with minimal required fields."""
        from src.a2a.models import AgentCard, Capability

        card = AgentCard(
            name="test-agent",
            description="Test agent",
            version="1.0.0",
            capabilities=Capability(),
            skills=[],
        )

        assert card.name == "test-agent"
        assert card.description == "Test agent"
        assert card.version == "1.0.0"
        assert isinstance(card.capabilities, Capability)
        assert card.skills == []

    def test_agent_card_includes_capabilities(self) -> None:
        """AgentCard includes capabilities object (AC-PI2.7)."""
        from src.a2a.models import AgentCard, Capability

        capability = Capability(streaming=True, stateTransitionHistory=True)
        card = AgentCard(
            name="test-agent",
            description="Test agent",
            version="1.0.0",
            capabilities=capability,
            skills=[],
        )

        assert card.capabilities.streaming is True
        assert card.capabilities.stateTransitionHistory is True
        assert card.capabilities.pushNotifications is False

    def test_agent_card_includes_skills(self) -> None:
        """AgentCard includes skills list."""
        from src.a2a.models import AgentCard, Capability, Skill

        skill = Skill(
            id="test_function",
            name="Test Function",
            description="A test function",
        )
        card = AgentCard(
            name="test-agent",
            description="Test agent",
            version="1.0.0",
            capabilities=Capability(),
            skills=[skill],
        )

        assert len(card.skills) == 1
        assert card.skills[0].id == "test_function"

    def test_agent_card_validates_against_a2a_schema(self) -> None:
        """AgentCard structure matches A2A spec v0.3.0 (AC-PI2.1)."""
        from src.a2a.models import AgentCard, Capability, Skill

        skill = Skill(
            id="extract_structure",
            name="Extract Structure",
            description="Extract structured data from content",
            tags=["extraction", "nlp"],
            examples=["Extract keywords from this text"],
            inputModes=["text/plain", "application/json"],
            outputModes=["application/json"],
        )

        card = AgentCard(
            name="ai-agents-service",
            description="AI Platform Agent Functions Service",
            version="1.0.0",
            capabilities=Capability(
                streaming=True,
                pushNotifications=False,
                stateTransitionHistory=True,
            ),
            skills=[skill],
        )

        # Validate structure matches A2A spec
        assert card.protocolVersion == "0.3.0"
        assert card.name == "ai-agents-service"
        assert card.description == "AI Platform Agent Functions Service"
        assert card.version == "1.0.0"
        assert card.capabilities.streaming is True
        assert len(card.skills) == 1
        assert card.skills[0].inputModes == ["text/plain", "application/json"]
        assert card.skills[0].outputModes == ["application/json"]

    def test_agent_card_serializes_to_json(self) -> None:
        """AgentCard serializes to JSON correctly."""
        from src.a2a.models import AgentCard, Capability, Skill

        skill = Skill(
            id="test_function",
            name="Test Function",
            description="A test function",
        )
        card = AgentCard(
            name="test-agent",
            description="Test agent",
            version="1.0.0",
            capabilities=Capability(streaming=True),
            skills=[skill],
        )

        json_data = card.model_dump()

        assert json_data["protocolVersion"] == "0.3.0"
        assert json_data["name"] == "test-agent"
        assert json_data["capabilities"]["streaming"] is True
        assert len(json_data["skills"]) == 1
        assert json_data["skills"][0]["id"] == "test_function"

    def test_agent_card_json_uses_camel_case(self) -> None:
        """AgentCard JSON uses camelCase for field names."""
        from src.a2a.models import AgentCard, Capability

        card = AgentCard(
            name="test-agent",
            description="Test agent",
            version="1.0.0",
            capabilities=Capability(
                streaming=True,
                pushNotifications=True,
                stateTransitionHistory=True,
            ),
            skills=[],
        )

        json_data = card.model_dump(by_alias=True)

        # Check that field names are camelCase
        assert "protocolVersion" in json_data
        assert "capabilities" in json_data
        assert "pushNotifications" in json_data["capabilities"]
        assert "stateTransitionHistory" in json_data["capabilities"]
