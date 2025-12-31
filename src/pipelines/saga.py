"""Pipeline Saga Pattern for Compensation.

WBS-AGT14: Pipeline Orchestrator - Saga Pattern

Implements:
- PipelineSaga for tracking stage completions
- Compensation execution in reverse order
- Error handling during compensation

Acceptance Criteria:
- AC-14.5: Saga compensation on stage failure

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Error Handling / Saga
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


# =============================================================================
# Compensation Result
# =============================================================================

@dataclass
class CompensationResult:
    """Result of saga compensation execution."""

    stages_compensated: int = 0
    stages_succeeded: int = 0
    stages_failed: int = 0
    errors: list[tuple[str, Exception]] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        """Check if all compensations succeeded."""
        return self.stages_failed == 0


# =============================================================================
# Completed Stage Record
# =============================================================================

@dataclass
class CompletedStage:
    """Record of a completed stage for compensation."""

    stage: str
    context: dict[str, Any]


# =============================================================================
# Pipeline Saga
# =============================================================================

@dataclass
class PipelineSaga:
    """Saga pattern implementation for pipeline compensation.

    Tracks completed stages and their compensation handlers.
    On failure, executes compensation in reverse order.

    Example:
        saga = PipelineSaga(name="resource_pipeline")

        # Register compensation for each stage
        saga.register_compensation("create_user", rollback_user)
        saga.register_compensation("create_order", rollback_order)

        # Record completions as stages succeed
        saga.record_completion("create_user", {"user_id": "123"})
        saga.record_completion("create_order", {"order_id": "456"})

        # On failure, compensate
        if error:
            await saga.compensate()  # Rolls back order, then user
    """

    name: str
    completed_stages: list[CompletedStage] = field(default_factory=list)
    compensation_handlers: dict[str, Callable[[dict[str, Any]], Coroutine[Any, Any, None]]] = field(
        default_factory=dict
    )

    def register_compensation(
        self,
        stage_name: str,
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a compensation handler for a stage.

        Args:
            stage_name: Name of the stage this handler compensates
            handler: Async function that performs compensation,
                    receives the stage's output context
        """
        self.compensation_handlers[stage_name] = handler

    def record_completion(
        self,
        stage_name: str,
        context: dict[str, Any],
    ) -> None:
        """Record that a stage has completed successfully.

        Args:
            stage_name: Name of the completed stage
            context: Output/context from the stage (passed to compensation)
        """
        self.completed_stages.append(
            CompletedStage(stage=stage_name, context=context)
        )

    async def compensate(self) -> CompensationResult:
        """Execute compensation for all completed stages.

        Compensation runs in reverse order of completion.
        Continues even if individual handlers fail.

        Returns:
            CompensationResult with success/failure counts and errors
        """
        result = CompensationResult()

        # Process in reverse order
        for completed in reversed(self.completed_stages):
            stage_name = completed.stage
            result.stages_compensated += 1

            # Find handler
            handler = self.compensation_handlers.get(stage_name)
            if handler is None:
                # No handler registered, skip
                result.stages_succeeded += 1
                continue

            # Execute compensation
            try:
                await handler(completed.context)
                result.stages_succeeded += 1
            except Exception as e:
                result.stages_failed += 1
                result.errors.append((stage_name, e))

        return result

    def clear(self) -> None:
        """Clear all recorded completions (but keep handlers)."""
        self.completed_stages.clear()


__all__ = [
    "CompensationResult",
    "CompletedStage",
    "PipelineSaga",
]
