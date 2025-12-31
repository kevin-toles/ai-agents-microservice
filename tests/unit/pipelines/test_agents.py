"""Tests for Agent Patterns (Sequential, Parallel, Loop).

TDD tests for WBS-AGT14: Pipeline Orchestrator - Agent Patterns.

Acceptance Criteria Coverage:
- AC-14.2: Supports SequentialAgent, ParallelAgent, LoopAgent patterns

Exit Criteria:
- SequentialAgent executes stages in order
- ParallelAgent uses asyncio.gather
- LoopAgent repeats until condition

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pipeline Composition
"""

import pytest
import asyncio
from typing import Any
from pydantic import BaseModel


# =============================================================================
# AC-14.2: SequentialAgent Tests
# =============================================================================

class TestSequentialAgent:
    """Tests for SequentialAgent pattern."""

    def test_sequential_agent_creation(self) -> None:
        """SequentialAgent can be created with stages."""
        from src.pipelines.agents import SequentialAgent
        from src.functions.base import AgentFunction
        
        class DummyOutput(BaseModel):
            data: str
        
        class DummyFunction(AgentFunction):
            name = "dummy"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> DummyOutput:
                return DummyOutput(data="test")
        
        agent = SequentialAgent(
            name="test_sequence",
            functions=[DummyFunction()],
        )
        
        assert agent.name == "test_sequence"
        assert len(agent.functions) == 1

    @pytest.mark.asyncio
    async def test_sequential_agent_executes_in_order(self) -> None:
        """SequentialAgent executes functions in order (AC-14.2)."""
        from src.pipelines.agents import SequentialAgent
        from src.functions.base import AgentFunction
        
        execution_log: list[str] = []
        
        class Output(BaseModel):
            value: int
        
        class FirstFunc(AgentFunction):
            name = "first"
            default_preset = "S1"
            
            async def run(self, *, value: int = 0, **kwargs: Any) -> Output:
                execution_log.append("first")
                return Output(value=value + 1)
        
        class SecondFunc(AgentFunction):
            name = "second"
            default_preset = "S1"
            
            async def run(self, *, value: int = 0, **kwargs: Any) -> Output:
                execution_log.append("second")
                return Output(value=value + 2)
        
        class ThirdFunc(AgentFunction):
            name = "third"
            default_preset = "S1"
            
            async def run(self, *, value: int = 0, **kwargs: Any) -> Output:
                execution_log.append("third")
                return Output(value=value + 3)
        
        agent = SequentialAgent(
            name="ordered_sequence",
            functions=[FirstFunc(), SecondFunc(), ThirdFunc()],
        )
        
        result = await agent.execute(value=0)
        
        # Verify execution order
        assert execution_log == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_sequential_agent_passes_output_to_next(self) -> None:
        """SequentialAgent passes output from one function to the next."""
        from src.pipelines.agents import SequentialAgent
        from src.functions.base import AgentFunction
        
        class Output(BaseModel):
            value: int
        
        class IncrementFunc(AgentFunction):
            name = "increment"
            default_preset = "S1"
            
            async def run(self, *, value: int, **kwargs: Any) -> Output:
                return Output(value=value + 10)
        
        agent = SequentialAgent(
            name="chain",
            functions=[IncrementFunc(), IncrementFunc(), IncrementFunc()],
        )
        
        result = await agent.execute(value=0)
        
        # 0 + 10 + 10 + 10 = 30
        assert result.value == 30

    @pytest.mark.asyncio
    async def test_sequential_agent_stops_on_error(self) -> None:
        """SequentialAgent stops execution on error."""
        from src.pipelines.agents import SequentialAgent
        from src.functions.base import AgentFunction
        
        execution_log: list[str] = []
        
        class Output(BaseModel):
            value: int
        
        class GoodFunc(AgentFunction):
            name = "good"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> Output:
                execution_log.append("good")
                return Output(value=1)
        
        class BadFunc(AgentFunction):
            name = "bad"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> Output:
                execution_log.append("bad")
                raise RuntimeError("Failed!")
        
        class NeverFunc(AgentFunction):
            name = "never"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> Output:
                execution_log.append("never")  # Should not run
                return Output(value=3)
        
        agent = SequentialAgent(
            name="error_chain",
            functions=[GoodFunc(), BadFunc(), NeverFunc()],
        )
        
        with pytest.raises(RuntimeError):
            await agent.execute()
        
        # Third function should not have executed
        assert "good" in execution_log
        assert "bad" in execution_log
        assert "never" not in execution_log


# =============================================================================
# AC-14.2: ParallelAgent Tests
# =============================================================================

class TestParallelAgent:
    """Tests for ParallelAgent pattern."""

    def test_parallel_agent_creation(self) -> None:
        """ParallelAgent can be created with functions."""
        from src.pipelines.agents import ParallelAgent
        from src.functions.base import AgentFunction
        
        class DummyOutput(BaseModel):
            data: str
        
        class DummyFunction(AgentFunction):
            name = "dummy"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> DummyOutput:
                return DummyOutput(data="test")
        
        agent = ParallelAgent(
            name="test_parallel",
            functions=[DummyFunction(), DummyFunction()],
        )
        
        assert agent.name == "test_parallel"
        assert len(agent.functions) == 2

    @pytest.mark.asyncio
    async def test_parallel_agent_uses_asyncio_gather(self) -> None:
        """ParallelAgent uses asyncio.gather for concurrent execution (AC-14.2)."""
        from src.pipelines.agents import ParallelAgent
        from src.functions.base import AgentFunction
        import time
        
        class Output(BaseModel):
            value: str
        
        class SlowFunc(AgentFunction):
            name = "slow"
            default_preset = "S1"
            
            async def run(self, *, id: str = "default", **kwargs: Any) -> Output:
                await asyncio.sleep(0.1)  # 100ms
                return Output(value=f"done_{id}")
        
        # Create 3 slow functions - should complete in ~100ms if parallel
        # Would take ~300ms if sequential
        agent = ParallelAgent(
            name="concurrent",
            functions=[SlowFunc(), SlowFunc(), SlowFunc()],
        )
        
        start = time.time()
        results = await agent.execute(id="test")
        elapsed = time.time() - start
        
        # All three results should be present
        assert len(results) == 3
        # Should complete in ~100ms, not ~300ms (allow some margin)
        assert elapsed < 0.25  # Less than 250ms means parallel

    @pytest.mark.asyncio
    async def test_parallel_agent_returns_all_results(self) -> None:
        """ParallelAgent returns results from all functions."""
        from src.pipelines.agents import ParallelAgent
        from src.functions.base import AgentFunction
        
        class Output(BaseModel):
            name: str
        
        class FuncA(AgentFunction):
            name = "func_a"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> Output:
                return Output(name="A")
        
        class FuncB(AgentFunction):
            name = "func_b"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> Output:
                return Output(name="B")
        
        class FuncC(AgentFunction):
            name = "func_c"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> Output:
                return Output(name="C")
        
        agent = ParallelAgent(
            name="multi",
            functions=[FuncA(), FuncB(), FuncC()],
        )
        
        results = await agent.execute()
        
        assert len(results) == 3
        names = {r.name for r in results}
        assert names == {"A", "B", "C"}

    @pytest.mark.asyncio
    async def test_parallel_agent_collects_partial_on_error(self) -> None:
        """ParallelAgent can collect partial results on error."""
        from src.pipelines.agents import ParallelAgent
        from src.functions.base import AgentFunction
        
        class Output(BaseModel):
            value: str
        
        class GoodFunc(AgentFunction):
            name = "good"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> Output:
                return Output(value="success")
        
        class FailFunc(AgentFunction):
            name = "fail"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> Output:
                raise RuntimeError("Failed!")
        
        agent = ParallelAgent(
            name="partial",
            functions=[GoodFunc(), FailFunc()],
            return_exceptions=True,  # Collect errors instead of raising
        )
        
        results = await agent.execute()
        
        # Should have 2 results - one success, one exception
        assert len(results) == 2
        # One should be Output, one should be an exception
        successes = [r for r in results if isinstance(r, Output)]
        errors = [r for r in results if isinstance(r, Exception)]
        
        assert len(successes) == 1
        assert len(errors) == 1


# =============================================================================
# AC-14.2: LoopAgent Tests
# =============================================================================

class TestLoopAgent:
    """Tests for LoopAgent pattern."""

    def test_loop_agent_creation(self) -> None:
        """LoopAgent can be created with function and condition."""
        from src.pipelines.agents import LoopAgent
        from src.functions.base import AgentFunction
        
        class Output(BaseModel):
            value: int
        
        class IterFunc(AgentFunction):
            name = "iterate"
            default_preset = "S1"
            
            async def run(self, *, value: int = 0, **kwargs: Any) -> Output:
                return Output(value=value + 1)
        
        agent = LoopAgent(
            name="test_loop",
            function=IterFunc(),
            condition=lambda result: result.value < 5,
            max_iterations=10,
        )
        
        assert agent.name == "test_loop"
        assert agent.max_iterations == 10

    @pytest.mark.asyncio
    async def test_loop_agent_repeats_until_condition_false(self) -> None:
        """LoopAgent repeats until condition returns False."""
        from src.pipelines.agents import LoopAgent
        from src.functions.base import AgentFunction
        
        iteration_count = 0
        
        class Output(BaseModel):
            count: int
        
        class CountFunc(AgentFunction):
            name = "count"
            default_preset = "S1"
            
            async def run(self, *, count: int = 0, **kwargs: Any) -> Output:
                nonlocal iteration_count
                iteration_count += 1
                return Output(count=count + 1)
        
        agent = LoopAgent(
            name="counter",
            function=CountFunc(),
            condition=lambda result: result.count < 5,
            max_iterations=100,
        )
        
        result = await agent.execute(count=0)
        
        # Should loop 5 times: 0->1->2->3->4->5 (stops when count >= 5)
        assert result.count == 5
        assert iteration_count == 5

    @pytest.mark.asyncio
    async def test_loop_agent_respects_max_iterations(self) -> None:
        """LoopAgent respects max_iterations limit."""
        from src.pipelines.agents import LoopAgent, MaxIterationsExceededError
        from src.functions.base import AgentFunction
        
        class Output(BaseModel):
            value: int
        
        class InfiniteFunc(AgentFunction):
            name = "infinite"
            default_preset = "S1"
            
            async def run(self, *, value: int = 0, **kwargs: Any) -> Output:
                return Output(value=value + 1)
        
        agent = LoopAgent(
            name="bounded",
            function=InfiniteFunc(),
            condition=lambda result: True,  # Always continue (infinite loop)
            max_iterations=5,
        )
        
        with pytest.raises(MaxIterationsExceededError) as exc_info:
            await agent.execute(value=0)
        
        assert exc_info.value.iterations == 5

    @pytest.mark.asyncio
    async def test_loop_agent_passes_output_to_next_iteration(self) -> None:
        """LoopAgent passes output from one iteration to the next."""
        from src.pipelines.agents import LoopAgent
        from src.functions.base import AgentFunction
        
        class Output(BaseModel):
            accumulated: str
        
        class AppendFunc(AgentFunction):
            name = "append"
            default_preset = "S1"
            
            async def run(self, *, accumulated: str = "", **kwargs: Any) -> Output:
                return Output(accumulated=accumulated + "x")
        
        agent = LoopAgent(
            name="accumulator",
            function=AppendFunc(),
            condition=lambda result: len(result.accumulated) < 5,
            max_iterations=10,
        )
        
        result = await agent.execute(accumulated="")
        
        # Should accumulate "xxxxx" (5 x's)
        assert result.accumulated == "xxxxx"

    @pytest.mark.asyncio
    async def test_loop_agent_returns_iteration_history(self) -> None:
        """LoopAgent can return history of all iterations."""
        from src.pipelines.agents import LoopAgent
        from src.functions.base import AgentFunction
        
        class Output(BaseModel):
            value: int
        
        class IncrementFunc(AgentFunction):
            name = "increment"
            default_preset = "S1"
            
            async def run(self, *, value: int = 0, **kwargs: Any) -> Output:
                return Output(value=value + 1)
        
        agent = LoopAgent(
            name="history_loop",
            function=IncrementFunc(),
            condition=lambda result: result.value < 3,
            max_iterations=10,
            keep_history=True,
        )
        
        result = await agent.execute(value=0)
        
        # Should have history of all iterations stored in agent
        assert agent.history is not None
        assert len(agent.history) == 3  # 0->1, 1->2, 2->3


# =============================================================================
# Agent Composition Tests
# =============================================================================

class TestAgentComposition:
    """Tests for composing agents together."""

    @pytest.mark.asyncio
    async def test_sequential_contains_parallel(self) -> None:
        """SequentialAgent can contain ParallelAgent."""
        from src.pipelines.agents import SequentialAgent, ParallelAgent
        from src.functions.base import AgentFunction
        
        class Output(BaseModel):
            value: str
        
        class SetupFunc(AgentFunction):
            name = "setup"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> Output:
                return Output(value="setup_done")
        
        class WorkerFunc(AgentFunction):
            name = "worker"
            default_preset = "S1"
            
            async def run(self, *, id: str = "0", **kwargs: Any) -> Output:
                return Output(value=f"work_{id}")
        
        class FinalizeFunc(AgentFunction):
            name = "finalize"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> Output:
                return Output(value="finalized")
        
        # Create parallel workers
        parallel_workers = ParallelAgent(
            name="workers",
            functions=[WorkerFunc(), WorkerFunc()],
        )
        
        # Wrap in sequence: setup -> parallel work -> finalize
        pipeline = SequentialAgent(
            name="full_pipeline",
            functions=[SetupFunc(), parallel_workers, FinalizeFunc()],
        )
        
        result = await pipeline.execute()
        
        # Final result should be from FinalizeFunc
        assert result.value == "finalized"


# =============================================================================
# Agent Protocol Tests
# =============================================================================

class TestAgentProtocol:
    """Tests for agent protocol compliance."""

    def test_sequential_agent_implements_protocol(self) -> None:
        """SequentialAgent implements BaseAgent protocol."""
        from src.pipelines.agents import SequentialAgent, BaseAgent
        
        agent = SequentialAgent(name="test", functions=[])
        
        assert isinstance(agent, BaseAgent)
        assert hasattr(agent, "execute")
        assert hasattr(agent, "name")

    def test_parallel_agent_implements_protocol(self) -> None:
        """ParallelAgent implements BaseAgent protocol."""
        from src.pipelines.agents import ParallelAgent, BaseAgent
        
        agent = ParallelAgent(name="test", functions=[])
        
        assert isinstance(agent, BaseAgent)
        assert hasattr(agent, "execute")
        assert hasattr(agent, "name")

    def test_loop_agent_implements_protocol(self) -> None:
        """LoopAgent implements BaseAgent protocol."""
        from src.pipelines.agents import LoopAgent, BaseAgent
        from src.functions.base import AgentFunction
        from pydantic import BaseModel
        
        class DummyOutput(BaseModel):
            value: int
        
        class DummyFunc(AgentFunction):
            name = "dummy"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> DummyOutput:
                return DummyOutput(value=0)
        
        agent = LoopAgent(
            name="test",
            function=DummyFunc(),
            condition=lambda r: False,
            max_iterations=1,
        )
        
        assert isinstance(agent, BaseAgent)
        assert hasattr(agent, "execute")
        assert hasattr(agent, "name")


__all__ = [
    "TestSequentialAgent",
    "TestParallelAgent",
    "TestLoopAgent",
    "TestAgentComposition",
    "TestAgentProtocol",
]
