"""Tests for A2A agent card generation.

TDD tests for WBS-PI2: A2A Agent Card & Discovery - Agent Card Generation.

Acceptance Criteria Coverage:
- AC-PI2.3: generate_agent_card() builds card from AgentFunctionRegistry
- AC-PI2.6: Card includes protocolVersion: "0.3.0"
- AC-PI2.7: Card includes capabilities.streaming based on feature flag

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → Agent Card Schema
"""

import pytest


# =============================================================================
# AC-PI2.3: Agent Card Generation
# =============================================================================


class TestGenerateAgentCard:
    """Tests for generate_agent_card() function."""

    def test_generate_agent_card_function_exists(self) -> None:
        """generate_agent_card() function can be imported."""
        from src.a2a.agent_card import generate_agent_card

        assert callable(generate_agent_card)

    def test_generate_agent_card_returns_agent_card(self) -> None:
        """generate_agent_card() returns AgentCard instance."""
        from src.a2a.agent_card import generate_agent_card
        from src.a2a.models import AgentCard

        card = generate_agent_card()

        assert isinstance(card, AgentCard)

    def test_generate_agent_card_includes_protocol_version(self) -> None:
        """Generated card includes protocolVersion 0.3.0 (AC-PI2.6)."""
        from src.a2a.agent_card import generate_agent_card

        card = generate_agent_card()

        assert card.protocolVersion == "0.3.0"

    def test_generate_agent_card_includes_service_metadata(self) -> None:
        """Generated card includes service name, description, version."""
        from src.a2a.agent_card import generate_agent_card

        card = generate_agent_card()

        assert card.name == "ai-agents-service"
        assert "AI Platform" in card.description
        assert card.version is not None

    def test_generate_agent_card_generates_skills_from_registry(self) -> None:
        """Generated card includes skills from FUNCTION_REGISTRY (AC-PI2.3)."""
        from src.a2a.agent_card import generate_agent_card

        card = generate_agent_card()

        # Should have 8 skills matching the 8 registered functions
        assert len(card.skills) == 8

    def test_generate_agent_card_skill_ids_match_function_names(self) -> None:
        """Generated skill IDs match function names (AC-PI2.3)."""
        from src.a2a.agent_card import generate_agent_card

        card = generate_agent_card()

        skill_ids = {skill.id for skill in card.skills}

        # Check for expected function names
        expected_ids = {
            "extract_structure",
            "summarize_content",
            "generate_code",
            "analyze_artifact",
            "validate_against_spec",
            "decompose_task",
            "synthesize_outputs",
            "cross_reference",
        }

        assert skill_ids == expected_ids

    def test_generate_agent_card_skills_have_descriptions(self) -> None:
        """Generated skills have non-empty descriptions."""
        from src.a2a.agent_card import generate_agent_card

        card = generate_agent_card()

        for skill in card.skills:
            assert skill.description != ""
            assert len(skill.description) > 10  # Meaningful description

    def test_generate_agent_card_skills_have_tags(self) -> None:
        """Generated skills include relevant tags."""
        from src.a2a.agent_card import generate_agent_card

        card = generate_agent_card()

        # Find extract_structure skill
        extract_skill = next(s for s in card.skills if s.id == "extract_structure")

        assert len(extract_skill.tags) > 0
        assert "extraction" in extract_skill.tags or "nlp" in extract_skill.tags

    def test_generate_agent_card_capabilities_based_on_flags(self) -> None:
        """Generated card capabilities reflect feature flags (AC-PI2.7)."""
        from src.a2a.agent_card import generate_agent_card
        from src.config.feature_flags import ProtocolFeatureFlags

        # Test with streaming enabled
        flags = ProtocolFeatureFlags(
            a2a_enabled=True,
            a2a_streaming_enabled=True,
        )

        card = generate_agent_card(flags=flags)

        assert card.capabilities.streaming is True

    def test_generate_agent_card_streaming_disabled_by_default(self) -> None:
        """Generated card has streaming disabled when flag is False."""
        from src.a2a.agent_card import generate_agent_card
        from src.config.feature_flags import ProtocolFeatureFlags

        flags = ProtocolFeatureFlags(
            a2a_enabled=True,
            a2a_streaming_enabled=False,
        )

        card = generate_agent_card(flags=flags)

        assert card.capabilities.streaming is False

    def test_generate_agent_card_push_notifications_based_on_flag(self) -> None:
        """Generated card pushNotifications reflects flag."""
        from src.a2a.agent_card import generate_agent_card
        from src.config.feature_flags import ProtocolFeatureFlags

        flags = ProtocolFeatureFlags(
            a2a_enabled=True,
            a2a_push_notifications=True,
        )

        card = generate_agent_card(flags=flags)

        assert card.capabilities.pushNotifications is True

    def test_generate_agent_card_state_history_always_enabled(self) -> None:
        """Generated card always has stateTransitionHistory enabled."""
        from src.a2a.agent_card import generate_agent_card

        card = generate_agent_card()

        assert card.capabilities.stateTransitionHistory is True

    def test_generate_agent_card_uses_default_flags_if_none(self) -> None:
        """generate_agent_card() uses default flags when none provided."""
        from src.a2a.agent_card import generate_agent_card
        from src.a2a.models import AgentCard

        # Should not raise error when flags=None
        card = generate_agent_card()

        assert isinstance(card, AgentCard)
        # Default flags should have streaming disabled
        assert card.capabilities.streaming is False

    def test_generate_agent_card_deterministic_output(self) -> None:
        """generate_agent_card() produces consistent output."""
        from src.a2a.agent_card import generate_agent_card

        card1 = generate_agent_card()
        card2 = generate_agent_card()

        assert card1.protocolVersion == card2.protocolVersion
        assert card1.name == card2.name
        assert len(card1.skills) == len(card2.skills)

    def test_generate_agent_card_serializes_to_json(self) -> None:
        """Generated card can be serialized to JSON."""
        from src.a2a.agent_card import generate_agent_card
        import json

        card = generate_agent_card()

        # Should serialize without errors
        json_str = card.model_dump_json()
        json_data = json.loads(json_str)

        assert json_data["protocolVersion"] == "0.3.0"
        assert "skills" in json_data
        assert len(json_data["skills"]) == 8
