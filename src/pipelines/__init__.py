"""Pipeline Package for Agent Functions.

WBS-AGT14: Pipeline Orchestrator
WBS-AGT15: Chapter Summarization Pipeline
WBS-AGT16: Code Generation Pipeline

This package provides:
- PipelineOrchestrator: Executes function DAGs (AC-14.1)
- Agent Patterns: Sequential, Parallel, Loop (AC-14.2)
- HandoffState: Stage-to-stage data flow (AC-14.3)
- Conditional Stages: Skip or fail based on conditions (AC-14.4)
- PipelineSaga: Compensation on failure (AC-14.5)
- ChapterSummarizationPipeline: 4-stage citation pipeline (AC-15.x)
- CodeGenerationPipeline: 6-stage code generation pipeline (AC-16.x)
"""

from src.pipelines.agents import (
    BaseAgent,
    LoopAgent,
    MaxIterationsExceededError,
    ParallelAgent,
    SequentialAgent,
)
from src.pipelines.chapter_summarization import (
    PRESET_MAPPING as CHAPTER_PRESET_MAPPING,
)
from src.pipelines.chapter_summarization import (
    ChapterSummarizationInput,
    ChapterSummarizationOutput,
    ChapterSummarizationPipeline,
    CitationAggregator,
)
from src.pipelines.chapter_summarization import (
    PresetType as ChapterPresetType,
)
from src.pipelines.code_generation import (
    PRESET_MAPPING as CODE_PRESET_MAPPING,
)
from src.pipelines.code_generation import (
    CodeGenerationInput,
    CodeGenerationOutput,
    CodeGenerationPipeline,
    Subtask,
)
from src.pipelines.code_generation import (
    PresetType as CodePresetType,
)
from src.pipelines.orchestrator import (
    ConditionEvaluator,
    DAGBuilder,
    HandoffState,
    PipelineDefinition,
    PipelineOrchestrator,
    PipelineResult,
    RetryConfig,
    StageCondition,
    StageDefinition,
    StageResult,
    StageStatus,
)
from src.pipelines.saga import (
    CompensationResult,
    CompletedStage,
    PipelineSaga,
)


__all__ = [
    "CHAPTER_PRESET_MAPPING",
    "CODE_PRESET_MAPPING",
    # Agents
    "BaseAgent",
    "ChapterPresetType",
    "ChapterSummarizationInput",
    "ChapterSummarizationOutput",
    # Chapter Summarization
    "ChapterSummarizationPipeline",
    "CitationAggregator",
    "CodeGenerationInput",
    "CodeGenerationOutput",
    # Code Generation
    "CodeGenerationPipeline",
    "CodePresetType",
    "CompensationResult",
    "CompletedStage",
    "ConditionEvaluator",
    "DAGBuilder",
    "HandoffState",
    "LoopAgent",
    "MaxIterationsExceededError",
    "ParallelAgent",
    "PipelineDefinition",
    # Orchestrator
    "PipelineOrchestrator",
    "PipelineResult",
    # Saga
    "PipelineSaga",
    "RetryConfig",
    "SequentialAgent",
    "StageCondition",
    "StageDefinition",
    "StageResult",
    "StageStatus",
    "Subtask",
]
