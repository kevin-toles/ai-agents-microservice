"""
LLM Participant Adapter - Calls LLMs via llm-gateway.

This adapter routes LLM requests through the llm-gateway service,
which handles provider routing (OpenAI, OpenRouter, Anthropic, etc.).

Each LLM participant is independent - if one provider fails, that participant
fails but others continue. There is no fallback between participants.

Reference: docs/INTER_AI_ORCHESTRATION.md
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

from src.participants.base import BaseParticipant


if TYPE_CHECKING:
    from src.conversation.models import Conversation, Participant


logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Error from LLM provider with context for error handling."""

    def __init__(
        self,
        message: str,
        provider: str,
        model: str,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.error_code = error_code


class LLMParticipantAdapter(BaseParticipant):
    """Adapter for LLM participants via llm-gateway.

    Routes all LLM calls through llm-gateway, which handles:
    - Provider selection (OpenAI, OpenRouter, Anthropic, etc.)
    - Rate limiting
    - Error handling
    - Session management

    Each LLM participant is independent. If a provider fails (e.g., credits
    exhausted, rate limited), that participant fails but others continue.

    Attributes:
        gateway_url: URL of the llm-gateway service.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        gateway_url: str = "http://localhost:8080",
        timeout: float = 120.0,
    ) -> None:
        """Initialize the LLM participant adapter.

        Args:
            gateway_url: URL of llm-gateway service.
            timeout: Request timeout in seconds.
        """
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def respond(
        self,
        conversation: Conversation,
        participant: Participant,
    ) -> dict[str, Any]:
        """Generate a response from an LLM participant.

        Builds the message history and calls llm-gateway for completion.

        Args:
            conversation: Current conversation state.
            participant: The LLM participant definition.

        Returns:
            Dict with 'content', 'tokens_used', 'latency_ms', 'metadata'.

        Raises:
            LLMProviderError: If the provider fails (caller should handle gracefully).
        """
        start_time = datetime.utcnow()

        # Build messages for LLM
        messages = self._build_messages(conversation, participant)

        # Build request body
        request_body = {
            "model": participant.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000,
        }

        logger.info(
            "Calling LLM: provider=%s, model=%s, participant=%s",
            participant.provider, participant.model, participant.id
        )

        try:
            response = await self._client.post(
                f"{self.gateway_url}/v1/chat/completions",
                json=request_body,
            )

            # Check for errors in response
            if response.status_code >= 400:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("error", {}).get("message", response.text)
                error_code = error_data.get("error", {}).get("code")

                logger.error(
                    "LLM provider error: provider=%s, model=%s, status=%s, code=%s, message=%s",
                    participant.provider, participant.model,
                    response.status_code, error_code, error_msg
                )

                raise LLMProviderError(
                    message=error_msg,
                    provider=participant.provider or "unknown",
                    model=participant.model or "unknown",
                    status_code=response.status_code,
                    error_code=error_code,
                )

            data = response.json()

            # Extract content from OpenAI-format response
            content = ""
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content", "")

            # Extract usage
            tokens_used = None
            if data.get("usage"):
                tokens_used = data["usage"].get("total_tokens")

            elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            logger.info(
                "LLM response received: provider=%s, model=%s, tokens=%s, latency=%dms",
                participant.provider, participant.model, tokens_used, elapsed_ms
            )

            return {
                "content": content,
                "tokens_used": tokens_used,
                "latency_ms": elapsed_ms,
                "metadata": {
                    "model": participant.model,
                    "provider": participant.provider,
                    "response_id": data.get("id"),
                    "actual_model": data.get("model"),
                },
            }

        except httpx.HTTPStatusError as e:
            logger.error(
                "HTTP error calling LLM: provider=%s, model=%s, status=%s",
                participant.provider, participant.model, e.response.status_code
            )
            raise LLMProviderError(
                message=str(e),
                provider=participant.provider or "unknown",
                model=participant.model or "unknown",
                status_code=e.response.status_code,
            ) from e
        except LLMProviderError:
            raise
        except Exception as e:
            logger.error(
                "Unexpected error calling LLM: provider=%s, model=%s, error=%s",
                participant.provider, participant.model, e
            )
            raise LLMProviderError(
                message=str(e),
                provider=participant.provider or "unknown",
                model=participant.model or "unknown",
            ) from e

    def _build_messages(
        self,
        conversation: Conversation,
        participant: Participant,
    ) -> list[dict[str, str]]:
        """Build message list for LLM API.

        Args:
            conversation: Current conversation.
            participant: The LLM participant.

        Returns:
            List of {role, content} messages.
        """
        messages = []

        # System prompt with participant's role and context
        system_content = self._build_system_prompt(conversation, participant)
        messages.append({
            "role": "system",
            "content": system_content,
        })

        # Add conversation history
        for msg in conversation.messages[-30:]:  # Last 30 messages
            speaker = msg.participant_id.upper()
            formatted_content = f"[{speaker}]: {msg.content}"

            # Determine role based on who sent it
            role = "assistant" if msg.participant_id == participant.id else "user"

            messages.append({
                "role": role,
                "content": formatted_content,
            })

        return messages

    def _build_system_prompt(
        self,
        conversation: Conversation,
        participant: Participant,
    ) -> str:
        """Build system prompt for the LLM.

        Args:
            conversation: Current conversation.
            participant: The LLM participant.

        Returns:
            System prompt string.
        """
        # Base prompt
        base_prompt = participant.system_prompt or ""

        # Add task and context
        context_str = json.dumps(conversation.context, indent=2, default=str)

        prompt_parts = [
            f"You are {participant.name}, participating in a collaborative AI discussion.",
            "",
            f"TASK: {conversation.task}",
            "",
            "CONTEXT:",
            context_str[:8000],  # Limit context size
            "",
            "INSTRUCTIONS:",
            "- Analyze the problem collaboratively with other participants",
            "- Build on insights from BERT tools when provided",
            "- Work toward consensus on the solution",
            "- Be concise and constructive",
            "- When you believe consensus is reached, state 'CONSENSUS REACHED' followed by the final answer",
        ]

        if base_prompt:
            prompt_parts.insert(0, base_prompt)
            prompt_parts.insert(1, "")

        return "\n".join(prompt_parts)

    async def health_check(self) -> bool:
        """Check if llm-gateway is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            response = await self._client.get(f"{self.gateway_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
