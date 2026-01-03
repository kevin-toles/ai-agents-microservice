"""
Conversation Orchestrator - Central Inter-AI Conversation Coordinator

This module provides the ConversationOrchestrator class that manages
conversations between multiple AI participants (LLMs and BERT tools).

Key Responsibilities:
- Manage turn-taking between participants
- Route messages to appropriate services
- Maintain conversation context
- Detect consensus/completion
- Log full conversation transcript

Reference: docs/INTER_AI_ORCHESTRATION.md
"""

from __future__ import annotations

import aiofiles
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.conversation.models import (
    Conversation,
    ConversationMessage,
    ConversationStatus,
    Participant,
    ParticipantType,
)


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """Central orchestrator for inter-AI conversations.

    Manages turn-taking, message routing, and consensus detection
    for conversations between LLMs and BERT tools.

    CRITICAL: All participant communication goes through this orchestrator.
    Participants NEVER communicate directly with each other.

    Flow:
        LLM A → Orchestrator → LLM B → Orchestrator → Tool → Orchestrator → LLM A

    Architecture:
        - Orchestrator = CONTROL PLANE (who, when, consensus, persist)
        - Adapters = EXECUTION PLANE (how, where, normalize)
        - Single contract: adapter.respond(conversation, participant)

    Attributes:
        llm_gateway_url: URL of the llm-gateway service.
        tool_service_url: URL of the Code-Orchestrator service.
        conversations: Active conversations by ID.
        transcript_dir: Directory to save conversation transcripts.
    """

    def __init__(
        self,
        llm_gateway_url: str = "http://localhost:8080",
        tool_service_url: str = "http://localhost:8083",
        transcript_dir: str | Path = "data/conversations",
    ) -> None:
        """Initialize the orchestrator.

        Args:
            llm_gateway_url: URL of llm-gateway service.
            tool_service_url: URL of Code-Orchestrator service.
            transcript_dir: Directory to save transcripts.
        """
        self.llm_gateway_url = llm_gateway_url
        self.tool_service_url = tool_service_url
        self.conversations: dict[str, Conversation] = {}
        self.transcript_dir = Path(transcript_dir)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

        # Adapter instances - created once, reused
        self._llm_adapter: Any = None
        self._tool_adapter: Any = None

        # Callbacks for consensus detection
        self._consensus_callbacks: dict[str, Callable] = {}

    async def start_conversation(
        self,
        task: str,
        participants: list[Participant],
        context: dict[str, Any],
        turn_order: list[str] | None = None,
        max_rounds: int = 10,
        min_rounds: int = 2,
        consensus_threshold: float = 0.8,
    ) -> Conversation:
        """Start a new inter-AI conversation.

        Args:
            task: The problem/task to solve.
            participants: List of participant definitions.
            context: Shared context (book list, clusters, etc.).
            turn_order: Order of turns. Defaults to participant order.
            max_rounds: Maximum rounds before timeout.
            min_rounds: Minimum rounds before consensus can be declared (prevents premature exit).
            consensus_threshold: Agreement level needed.

        Returns:
            The created Conversation object.
        """
        conversation = Conversation(
            task=task,
            participants=participants,
            context=context,
            turn_order=turn_order or [p.id for p in participants],
            max_rounds=max_rounds,
            min_rounds=min_rounds,
            consensus_threshold=consensus_threshold,
        )

        self.conversations[conversation.conversation_id] = conversation
        conversation.status = ConversationStatus.IN_PROGRESS

        logger.info(
            f"Started conversation {conversation.conversation_id} "
            f"with {len(participants)} participants"
        )

        # Send initial orchestrator message with task
        await self._send_orchestrator_message(
            conversation,
            f"Task: {task}\n\nContext has been provided. Begin discussion.",
        )

        return conversation

    def _should_check_consensus(
        self,
        llms_responded: set[str],
        all_llms: set[str],
        last_response: dict[str, Any] | None,
    ) -> bool:
        """Check if all LLMs have responded and consensus check is needed."""
        return llms_responded >= all_llms and last_response is not None

    async def _handle_participant_response(
        self,
        conversation: Conversation,
        participant: Participant,
        on_message: Callable[[ConversationMessage], None] | None,
    ) -> tuple[dict[str, Any], ConversationMessage]:
        """Get and process a participant's response."""
        response = await self._get_participant_response(conversation, participant)

        message = ConversationMessage(
            conversation_id=conversation.conversation_id,
            participant_id=participant.id,
            participant_type=participant.participant_type,
            role="assistant",
            content=response["content"],
            tokens_used=response.get("tokens_used"),
            latency_ms=response.get("latency_ms", 0),
            metadata=response.get("metadata", {}),
        )

        conversation.add_message(message)

        if on_message:
            on_message(message)

        logger.info(f"[{participant.id}]: {response['content'][:100]}...")
        return response, message

    async def _process_turn(
        self,
        conversation: Conversation,
        participant: Participant,
        on_message: Callable[[ConversationMessage], None] | None,
        llms_responded: set[str],
        llm_participants: set[str],
    ) -> tuple[dict[str, Any] | None, bool]:
        """Process a single conversation turn.

        Returns:
            Tuple of (last_response, reached_consensus)
        """
        try:
            response, _ = await self._handle_participant_response(conversation, participant, on_message)

            if participant.participant_type == ParticipantType.LLM:
                llms_responded.add(participant.id)

                if self._should_check_consensus(llms_responded, llm_participants, response):
                    logger.info(f"All LLMs responded ({len(llms_responded)}/{len(llm_participants)}), checking for consensus...")
                    if await self._check_consensus(conversation, response):
                        return response, True
                    llms_responded.clear()

            return response, False

        except Exception as e:
            logger.error(f"Error from {participant.id}: {e}")
            if participant.participant_type == ParticipantType.LLM:
                llms_responded.add(participant.id)
                logger.warning(f"LLM {participant.id} failed, marking as responded. ({len(llms_responded)}/{len(llm_participants)} LLMs)")
            await self._send_orchestrator_message(conversation, f"Error from {participant.id}: {e!s}. Continuing...")
            return None, False

    async def run_conversation(
        self,
        conversation: Conversation,
        on_message: Callable[[ConversationMessage], None] | None = None,
    ) -> Conversation:
        """Run the conversation until completion.

        Requires ALL LLM participants to respond before checking for consensus.
        If an LLM fails, it's skipped but others must still respond.

        Args:
            conversation: The conversation to run.
            on_message: Optional callback for each message.

        Returns:
            The completed conversation.
        """
        logger.info(f"Running conversation {conversation.conversation_id}")

        llm_participants = {p.id for p in conversation.participants if p.participant_type == ParticipantType.LLM}
        llms_responded: set[str] = set()

        while conversation.status == ConversationStatus.IN_PROGRESS:
            if conversation.current_round >= conversation.max_rounds:
                conversation.status = ConversationStatus.TIMEOUT
                await self._send_orchestrator_message(conversation, "Maximum rounds reached. Ending conversation.")
                break

            participant = conversation.get_participant(conversation.current_turn or "")
            if not participant:
                logger.error(f"Unknown participant: {conversation.current_turn}")
                conversation.advance_turn()
                continue

            _, reached_consensus = await self._process_turn(
                conversation, participant, on_message, llms_responded, llm_participants
            )

            if reached_consensus:
                conversation.status = ConversationStatus.CONSENSUS
                break

            conversation.advance_turn()

        await self._save_transcript(conversation)
        logger.info(f"Conversation {conversation.conversation_id} ended with status: {conversation.status.value}")

        return conversation

    async def _get_participant_response(
        self,
        conversation: Conversation,
        participant: Participant,
    ) -> dict[str, Any]:
        """Get response from a participant via its adapter.

        Routes to the appropriate adapter based on participant type.
        Adapters implement the unified respond() contract.

        Args:
            conversation: Current conversation state.
            participant: The participant to query.

        Returns:
            Dict with 'content' and optional 'tokens_used', 'latency_ms', 'metadata'.
        """
        # Lazy import adapters to avoid circular imports
        from src.participants.llm_participant import LLMParticipantAdapter
        from src.participants.tool_participant import ToolParticipantAdapter

        # Get or create adapter based on participant type
        if participant.participant_type == ParticipantType.LLM:
            if self._llm_adapter is None:
                self._llm_adapter = LLMParticipantAdapter(
                    gateway_url=self.llm_gateway_url
                )
            adapter = self._llm_adapter
        else:
            if self._tool_adapter is None:
                self._tool_adapter = ToolParticipantAdapter(
                    orchestrator_url=self.tool_service_url
                )
            adapter = self._tool_adapter

        # Single contract: adapter.respond(conversation, participant)
        return await adapter.respond(conversation, participant)

    def _check_consensus(
        self,
        conversation: Conversation,
        latest_response: dict[str, Any],
    ) -> bool:
        """Check if participants have reached consensus.

        Requires:
        1. Minimum rounds completed (for deeper discussion)
        2. Explicit consensus signal from the latest response
        3. No recent disagreement signals

        Args:
            conversation: Current conversation.
            latest_response: The most recent response.

        Returns:
            True if consensus reached, False otherwise.
        """
        # Require minimum rounds for deeper discussion
        if conversation.current_round < conversation.min_rounds:
            logger.debug(
                f"Round {conversation.current_round} < min_rounds {conversation.min_rounds}, "
                f"continuing discussion..."
            )
            return False

        content = latest_response.get("content", "").lower()

        # Check for disagreement signals (prevents premature consensus)
        disagreement_signals = [
            "i disagree",
            "i don't agree",
            "however,",
            "but i think",
            "on the contrary",
            "let me offer an alternative",
            "i have concerns",
            "need to reconsider",
        ]

        for signal in disagreement_signals:
            if signal in content:
                logger.debug(f"Disagreement signal detected: '{signal}'")
                return False

        # Check for EXPLICIT consensus signals (must be clear declarations, not embedded in analysis)
        # These must appear as standalone declarations, not within quoted analysis
        consensus_signals = [
            "consensus reached:",  # Require colon to be declaration, not embedded
            "we all agree that",
            "agreement reached:",
            "final consensus:",
            "unanimous agreement:",
            "all participants agree that",
            "we have reached consensus:",
            "## consensus",  # Markdown header format
            "**consensus**",  # Bold format
        ]

        for signal in consensus_signals:
            if signal in content:
                logger.info(f"Consensus signal detected: '{signal}'")
                return True

        # Check custom consensus callback if registered
        callback = self._consensus_callbacks.get(conversation.conversation_id)
        if callback:
            return callback(conversation, latest_response)

        return False

    def _send_orchestrator_message(
        self,
        conversation: Conversation,
        content: str,
    ) -> None:
        """Send a message from the orchestrator.

        Args:
            conversation: Current conversation.
            content: Message content.
        """
        message = ConversationMessage(
            conversation_id=conversation.conversation_id,
            participant_id="orchestrator",
            participant_type=ParticipantType.ORCHESTRATOR,
            role="system",
            content=content,
        )
        conversation.add_message(message)

    async def _save_transcript(self, conversation: Conversation) -> Path:
        """Save conversation transcript to file.

        Args:
            conversation: The conversation to save.

        Returns:
            Path to the saved transcript.
        """
        # Save as JSON
        json_path = self.transcript_dir / f"{conversation.conversation_id}.json"
        async with aiofiles.open(json_path, "w") as f:
            await f.write(json.dumps(conversation.to_dict(), indent=2, default=str))

        # Save as text transcript
        txt_path = self.transcript_dir / f"{conversation.conversation_id}.txt"
        async with aiofiles.open(txt_path, "w") as f:
            await f.write(conversation.get_transcript())

        logger.info(f"Saved transcript to {json_path}")

        return json_path

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The Conversation if found, None otherwise.
        """
        return self.conversations.get(conversation_id)

    def register_consensus_callback(
        self,
        conversation_id: str,
        callback: Callable[[Conversation, dict], bool],
    ) -> None:
        """Register a custom consensus detection callback.

        Args:
            conversation_id: The conversation ID.
            callback: Function that returns True if consensus reached.
        """
        self._consensus_callbacks[conversation_id] = callback

    def inject_message(
        self,
        conversation_id: str,
        content: str,
        from_participant: str = "human",
    ) -> ConversationMessage | None:
        """Inject a message into an active conversation (human-in-the-loop).

        Args:
            conversation_id: The conversation ID.
            content: Message content to inject.
            from_participant: Who the message is from.

        Returns:
            The created message, or None if conversation not found.
        """
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return None

        message = ConversationMessage(
            conversation_id=conversation_id,
            participant_id=from_participant,
            participant_type=ParticipantType.ORCHESTRATOR,
            role="user",
            content=content,
        )

        conversation.add_message(message)
        logger.info(f"Injected message from {from_participant} into {conversation_id}")

        return message

    async def stop_conversation(
        self,
        conversation_id: str,
        reason: str = "Manually stopped",
    ) -> Conversation | None:
        """Stop an active conversation.

        Args:
            conversation_id: The conversation ID.
            reason: Reason for stopping.

        Returns:
            The stopped conversation, or None if not found.
        """
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return None

        await self._send_orchestrator_message(conversation, f"Stopping: {reason}")
        conversation.status = ConversationStatus.COMPLETE
        await self._save_transcript(conversation)

        return conversation

    async def close(self) -> None:
        """Close all adapters and cleanup resources."""
        if self._llm_adapter:
            await self._llm_adapter.close()
        if self._tool_adapter:
            await self._tool_adapter.close()
