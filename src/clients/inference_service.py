"""Inference Service Client.

HTTP client for inference-service:8085 chat completions API.
Implements WBS-AGT7.7: Implement run() with inference-service call.

Architecture:
    - inference-service OWNS model lifecycle (loading, presets, configuration)
    - ai-agents expresses model PREFERENCES, not requirements
    - Model resolution: preference → role fallback → any loaded model

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Service Integration, Preset Selection
Reference: inference-service/docs/ARCHITECTURE.md → Model Configuration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# =============================================================================
# Model Preferences per Agent Function
# =============================================================================

# Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Quality/Latency Tradeoff Matrix
# These are PREFERENCES - if not available, fallback to role or any loaded model
MODEL_PREFERENCES: dict[str, dict[str, Any]] = {
    # extract_structure: Light preset (S4) - fast parsing, no heavy reasoning
    "extract_structure": {
        "preferred": "llama-3.2-3b",
        "fallback_roles": ["fast", "primary"],
        "description": "Fast model for structure extraction",
    },
    # summarize_content: Standard preset (D4) - needs reasoning/thinker
    "summarize_content": {
        "preferred": "deepseek-r1-7b",
        "fallback_roles": ["thinker", "primary"],
        "description": "Reasoning model for summarization",
    },
    # generate_code: Standard preset (D4) - needs coder capability
    "generate_code": {
        "preferred": "qwen2.5-7b",
        "fallback_roles": ["coder", "primary"],
        "description": "Code-focused model for generation",
    },
    # analyze_artifact: Standard preset (D4) - needs analysis/critique
    "analyze_artifact": {
        "preferred": "deepseek-r1-7b",
        "fallback_roles": ["thinker", "coder", "primary"],
        "description": "Analytical model for artifact analysis",
    },
    # validate_against_spec: Standard preset (D4) - needs reasoning
    "validate_against_spec": {
        "preferred": "deepseek-r1-7b",
        "fallback_roles": ["thinker", "primary"],
        "description": "Reasoning model for validation",
    },
    # synthesize_outputs: Light preset (S1) - combining, not heavy reasoning
    "synthesize_outputs": {
        "preferred": "phi-4",
        "fallback_roles": ["primary", "thinker"],
        "description": "General model for synthesis",
    },
    # decompose_task: Uses chain-of-thought reasoning
    "decompose_task": {
        "preferred": "deepseek-r1-7b",
        "fallback_roles": ["thinker", "primary"],
        "description": "Reasoning model for task decomposition",
    },
    # cross_reference: Light preset (S4) - semantic ranking only
    "cross_reference": {
        "preferred": "llama-3.2-3b",
        "fallback_roles": ["fast", "primary"],
        "description": "Fast model for cross-referencing",
    },
}


# =============================================================================
# Model Info Dataclass
# =============================================================================


@dataclass
class ModelInfo:
    """Information about a model from inference-service.
    
    Reference: inference-service/docs/ARCHITECTURE.md → Models List Response
    """
    
    model_id: str
    status: str
    memory_mb: int = 0
    context_length: int = 0
    roles: list[str] = field(default_factory=list)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self.status == "loaded"
    
    def has_role(self, role: str) -> bool:
        """Check if model supports a specific role."""
        return role in self.roles


# =============================================================================
# Context Validation Constants
# =============================================================================

# Chars per token estimate for pre-validation
CHARS_PER_TOKEN_ESTIMATE = 4
# Default output reserve (tokens reserved for model response)
DEFAULT_OUTPUT_RESERVE = 2048


# =============================================================================
# Model Resolver
# =============================================================================


class ModelResolver:
    """Resolves model selection from preferences and availability.
    
    Architecture:
        - inference-service owns model lifecycle
        - ai-agents expresses preferences, ModelResolver finds best available
        - Resolution order: preference → role fallback → any loaded
    
    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Preset Selection
    """
    
    def __init__(self, models_response: dict[str, Any]) -> None:
        """Initialize resolver from GET /v1/models response.
        
        Args:
            models_response: Response from inference-service /v1/models endpoint
        """
        self._models: list[ModelInfo] = []
        self._config: str = models_response.get("config", "unknown")
        self._orchestration_mode: str = models_response.get(
            "orchestration_mode", "single"
        )
        
        # Parse model data
        for model_data in models_response.get("data", []):
            self._models.append(ModelInfo(
                model_id=model_data["id"],
                status=model_data.get("status", "unknown"),
                memory_mb=model_data.get("memory_mb", 0),
                context_length=model_data.get("context_length", 0),
                roles=model_data.get("roles", []),
            ))

    def _find_preferred_model(self, loaded: list[ModelInfo], preferred: str) -> str | None:
        """Find preferred model if it's loaded."""
        for model in loaded:
            if model.model_id == preferred:
                logger.debug("Using preferred model: %s", preferred)
                return preferred
        return None

    def _find_model_by_role(
        self, loaded: list[ModelInfo], roles: list[str], preferred: str | None
    ) -> str | None:
        """Find first model matching any of the specified roles."""
        for role in roles:
            for model in loaded:
                if model.has_role(role):
                    logger.debug(
                        "Preferred model '%s' not loaded, using '%s' with role '%s'",
                        preferred, model.model_id, role,
                    )
                    return model.model_id
        return None

    def resolve(
        self,
        preferred: str | None = None,
        fallback_roles: list[str] | None = None,
    ) -> str:
        """Resolve model from preference with fallbacks.

        Resolution order:
        1. If preferred model is loaded → use it
        2. Try each fallback role in order → use first match
        3. Return any loaded model

        Args:
            preferred: Preferred model ID (not required to be loaded)
            fallback_roles: Roles to try if preferred not loaded

        Returns:
            Model ID to use for inference

        Raises:
            ValueError: If no models are loaded
        """
        loaded = self.get_loaded_models()

        if not loaded:
            raise ValueError("No models are currently loaded in inference-service")

        if preferred:
            if model_id := self._find_preferred_model(loaded, preferred):
                return model_id

        if fallback_roles:
            if model_id := self._find_model_by_role(loaded, fallback_roles, preferred):
                return model_id

        first_loaded = loaded[0]
        logger.debug("No preference/role match, using first loaded model: %s", first_loaded.model_id)
        return first_loaded.model_id
    
    def get_loaded_models(self) -> list[ModelInfo]:
        """Get list of currently loaded models."""
        return [m for m in self._models if m.is_loaded]
    
    def get_all_models(self) -> list[ModelInfo]:
        """Get list of all models (loaded and available)."""
        return list(self._models)
    
    def get_current_config(self) -> str:
        """Get the current preset configuration (e.g., 'D4', 'S1')."""
        return self._config
    
    def get_orchestration_mode(self) -> str:
        """Get current orchestration mode (e.g., 'single', 'critique')."""
        return self._orchestration_mode


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatMessage(BaseModel):
    """Chat message for inference-service."""

    role: str = Field(..., description="Message role: user, system, assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(..., description="Model ID")
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    stream: bool = Field(default=False)


class ChatCompletionChoice(BaseModel):
    """Choice in chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage | None = None


# =============================================================================
# Inference Service Client
# =============================================================================

class InferenceServiceClient:
    """HTTP client for inference-service chat completions.

    Provides async interface to inference-service:8085/v1/chat/completions.
    
    Architecture:
        - inference-service OWNS model lifecycle
        - This client queries available models and uses preferences with fallback
        - Model selection: preference → role fallback → any loaded model

    Usage:
        client = InferenceServiceClient("http://localhost:8085")
        
        # With model preference (preferred model or fallback)
        response = await client.complete(
            messages=[{"role": "user", "content": "Hello"}],
            model_preference="summarize_content",  # Uses MODEL_PREFERENCES
        )
        
        # With explicit model (must be loaded)
        response = await client.complete(
            messages=[{"role": "user", "content": "Hello"}],
            model="deepseek-r1-7b",
        )

    Attributes:
        base_url: Base URL for inference-service
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8085",
        timeout: float = 120.0,
    ) -> None:
        """Initialize inference service client.

        Args:
            base_url: Base URL for inference-service (default: localhost:8085)
            timeout: Request timeout in seconds (default: 120s for long generations)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._resolver: ModelResolver | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client (lazy initialization)."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Release HTTP client resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._resolver = None

    async def get_models(self, refresh: bool = False) -> ModelResolver:
        """Get available models from inference-service.
        
        Queries GET /v1/models to discover what models are loaded.
        
        Args:
            refresh: Force refresh of cached model list
        
        Returns:
            ModelResolver for model selection
        
        Raises:
            httpx.HTTPStatusError: On HTTP errors
        """
        if self._resolver is not None and not refresh:
            return self._resolver
        
        client = self._get_client()
        response = await client.get("/v1/models")
        response.raise_for_status()
        
        data = response.json()
        self._resolver = ModelResolver(data)
        
        logger.info(
            "Got models from inference-service: config=%s, loaded=%d",
            self._resolver.get_current_config(),
            len(self._resolver.get_loaded_models()),
        )
        
        return self._resolver

    async def _resolve_model(
        self,
        model: str | None = None,
        model_preference: str | None = None,
    ) -> str:
        """Resolve which model to use for inference.
        
        Resolution order:
        1. If explicit model provided → use it (caller's responsibility)
        2. If model_preference provided → use MODEL_PREFERENCES with fallback
        3. Else → use first loaded model
        
        Args:
            model: Explicit model ID (takes precedence)
            model_preference: Agent function name to get preference for
        
        Returns:
            Model ID to use
        
        Raises:
            ValueError: If no models loaded
        """
        # Explicit model takes precedence
        if model:
            return model
        
        # Get resolver (queries inference-service if not cached)
        resolver = await self.get_models()
        
        # Use preference if provided
        if model_preference and model_preference in MODEL_PREFERENCES:
            pref = MODEL_PREFERENCES[model_preference]
            return resolver.resolve(
                preferred=pref.get("preferred"),
                fallback_roles=pref.get("fallback_roles"),
            )
        
        # Default: resolve with no preference (returns any loaded model)
        return resolver.resolve()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Args:
            text: Text to estimate tokens for.
            
        Returns:
            Estimated token count.
        """
        return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)

    def _get_model_context_length(self, model_id: str) -> int:
        """Get context length for a model.
        
        Args:
            model_id: Model identifier.
            
        Returns:
            Context length in tokens (default 4096 if unknown).
        """
        if self._resolver is None:
            return 4096  # Conservative default
        
        for model in self._resolver.get_loaded_models():
            if model.model_id == model_id:
                return model.context_length or 4096
        
        return 4096

    def _truncate_messages_for_context(
        self,
        messages: list[ChatMessage],
        model_id: str,
        max_tokens: int,
    ) -> tuple[list[ChatMessage], bool]:
        """Truncate messages to fit within model context window.
        
        Strategy: Preserve system message and last user message, truncate middle.
        
        Args:
            messages: Original messages.
            model_id: Target model ID.
            max_tokens: Max tokens requested for output.
            
        Returns:
            Tuple of (truncated messages, was_truncated flag).
        """
        context_length = self._get_model_context_length(model_id)
        available_input = context_length - max_tokens - DEFAULT_OUTPUT_RESERVE
        
        # Estimate current usage
        total_text = "\n".join(f"{m.role}: {m.content}" for m in messages)
        current_tokens = self._estimate_tokens(total_text)
        
        if current_tokens <= available_input:
            return messages, False
        
        logger.warning(
            "Context exceeds model limit, truncating client-side: %d > %d (model: %s)",
            current_tokens,
            available_input,
            model_id,
        )
        
        # Can't truncate less than 2 messages
        if len(messages) <= 2:
            # Truncate content of last message
            truncated = list(messages)
            last_msg = truncated[-1]
            overage_tokens = current_tokens - available_input
            chars_to_cut = overage_tokens * CHARS_PER_TOKEN_ESTIMATE
            if last_msg.content and len(last_msg.content) > chars_to_cut:
                new_content = last_msg.content[:-chars_to_cut] + "\n\n[...truncated for model context limits...]"
                truncated[-1] = ChatMessage(role=last_msg.role, content=new_content)
                return truncated, True
            return messages, False
        
        # Preserve: system (first if present) + last message
        result: list[ChatMessage] = []
        
        # Keep system message if present
        if messages[0].role == "system":
            result.append(messages[0])
            remaining = messages[1:]
        else:
            remaining = list(messages)
        
        # Always keep last message
        last_message = remaining[-1]
        middle_messages = remaining[:-1]
        
        # Build from end until we hit budget
        kept_middle: list[ChatMessage] = []
        for msg in reversed(middle_messages):
            test_messages = result + [msg] + kept_middle + [last_message]
            test_text = "\n".join(f"{m.role}: {m.content}" for m in test_messages)
            if self._estimate_tokens(test_text) <= available_input:
                kept_middle.insert(0, msg)
            else:
                break
        
        # Add truncation notice if we dropped messages
        if len(kept_middle) < len(middle_messages):
            dropped_count = len(middle_messages) - len(kept_middle)
            notice = ChatMessage(
                role="system",
                content=f"[Note: {dropped_count} earlier messages truncated to fit context window]"
            )
            result.append(notice)
        
        result.extend(kept_middle)
        result.append(last_message)
        
        return result, True

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        model_preference: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        auto_truncate: bool = True,
    ) -> str:
        """Generate chat completion.

        Model Selection (in order of precedence):
        1. model: Explicit model ID (must be loaded)
        2. model_preference: Agent function name → uses MODEL_PREFERENCES
        3. Default: First loaded model
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Explicit model ID (optional, takes precedence)
            model_preference: Agent function name for preference lookup
            system_prompt: Optional system prompt to prepend
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Sampling temperature (default: 0.7)
            auto_truncate: Auto-truncate if context exceeds model limit (default: True)

        Returns:
            Generated completion text

        Raises:
            httpx.HTTPStatusError: On HTTP errors
            ValueError: On invalid response or no models loaded
        """
        # Resolve model using preference system
        resolved_model = await self._resolve_model(
            model=model,
            model_preference=model_preference,
        )

        # Build messages list
        chat_messages: list[ChatMessage] = []

        # Add system prompt if provided
        if system_prompt:
            chat_messages.append(ChatMessage(role="system", content=system_prompt))

        # Add user messages
        for msg in messages:
            chat_messages.append(ChatMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
            ))

        # Client-side context validation (defense in depth with server-side)
        if auto_truncate:
            chat_messages, was_truncated = self._truncate_messages_for_context(
                chat_messages, resolved_model, max_tokens
            )
            if was_truncated:
                logger.info(
                    "Messages truncated client-side for model %s",
                    resolved_model,
                )

        # Build request
        request = ChatCompletionRequest(
            model=resolved_model,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        logger.info(
            "Calling inference-service: model=%s (preference=%s), messages=%d",
            resolved_model,
            model_preference,
            len(chat_messages),
        )

        # Make request
        client = self._get_client()
        response = await client.post(
            "/v1/chat/completions",
            json=request.model_dump(),
        )
        response.raise_for_status()

        # Parse response
        data = response.json()
        completion = ChatCompletionResponse.model_validate(data)

        if not completion.choices:
            raise ValueError("No completion choices returned")

        result = completion.choices[0].message.content

        logger.info(
            "Inference complete: tokens=%s, model=%s",
            completion.usage.total_tokens if completion.usage else "unknown",
            completion.model,
        )

        return result

    async def summarize(
        self,
        content: str,
        model_preference: str = "summarize_content",
        detail_level: str = "standard",
        style: str = "technical",
        preserve: list[str] | None = None,
    ) -> str:
        """Generate summary using LLM.

        Specialized method for summarization tasks.

        Args:
            content: Text content to summarize
            model_preference: Agent function for preference lookup (default: summarize_content)
            detail_level: brief, standard, or comprehensive
            style: technical, executive, or bullets
            preserve: Concepts that must be preserved in summary

        Returns:
            Generated summary text
        """
        # Build system prompt
        system_prompt = self._build_summarization_prompt(
            detail_level=detail_level,
            style=style,
            preserve=preserve,
        )

        # Build user message
        messages = [{"role": "user", "content": content}]

        # Determine max tokens based on detail level
        max_tokens_map = {
            "brief": 500,
            "standard": 1500,
            "comprehensive": 3000,
        }
        max_tokens = max_tokens_map.get(detail_level, 1500)

        return await self.complete(
            messages=messages,
            model_preference=model_preference,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )

    def _build_summarization_prompt(
        self,
        detail_level: str,
        style: str,
        preserve: list[str] | None,
    ) -> str:
        """Build system prompt for summarization.

        Args:
            detail_level: brief, standard, or comprehensive
            style: technical, executive, or bullets
            preserve: Concepts to preserve

        Returns:
            System prompt string
        """
        # Base instruction
        prompt_parts = [
            "You are an expert summarizer. Generate a concise, accurate summary.",
        ]

        # Detail level instruction
        detail_instructions = {
            "brief": "Keep the summary very brief (under 200 words). Focus only on the key takeaways.",
            "standard": "Provide a balanced summary (300-500 words) covering main points.",
            "comprehensive": "Provide a detailed summary (500-800 words) with examples and nuance.",
        }
        prompt_parts.append(detail_instructions.get(detail_level, detail_instructions["standard"]))

        # Style instruction
        style_instructions = {
            "technical": "Use precise technical language. Include specific terms and concepts.",
            "executive": "Use clear business language. Focus on outcomes and implications.",
            "bullets": "Format as bullet points. Each point should be a complete thought.",
        }
        prompt_parts.append(style_instructions.get(style, style_instructions["technical"]))

        # Preserve concepts
        if preserve:
            preserve_str = ", ".join(preserve)
            prompt_parts.append(f"IMPORTANT: You MUST mention these concepts: {preserve_str}")

        return "\n\n".join(prompt_parts)


# =============================================================================
# Factory Function
# =============================================================================

def create_inference_client(
    base_url: str | None = None,
    timeout: float = 120.0,
) -> InferenceServiceClient:
    """Create inference service client.

    Args:
        base_url: Base URL (default: from env or localhost:8085)
        timeout: Request timeout in seconds

    Returns:
        InferenceServiceClient instance
    """
    import os

    if base_url is None:
        base_url = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8085")

    return InferenceServiceClient(base_url=base_url, timeout=timeout)
