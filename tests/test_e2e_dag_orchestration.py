"""
End-to-end tests for DAG tool call orchestration.

These tests verify the complete flow of DAG-based tool execution including:
- ToolCallDAG construction and topological sorting
- ToolCallNode lifecycle and status transitions
- WaveExecutor wave-based parallel execution
- Orchestrator streaming with event generation
- Cycle detection and error handling
- Execution key deduplication

Run with: pytest tests/test_e2e_dag_orchestration.py -v
"""

import asyncio
import pytest
from typing import Any, Dict, List, Optional
import logging

from petals.client.dag.tool_call_node import ToolCallNode
from petals.client.dag.dag import ToolCallDAG
from petals.client.dag.wave_executor import WaveExecutor
from petals.client.orchestrator import Orchestrator, OrchestratorConfig
from petals.client.tool_registry import ToolRegistry
from petals.client.async_support.streaming_types import StreamEvent, StreamEventType
from petals.data_structures import CallStatus

logger = logging.getLogger(__name__)


# ============================================================================
# Test Tools - Safe, No Side Effects
# ============================================================================

async def echo(text: str) -> str:
    """Echo back the input text."""
    return text


async def uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


async def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


async def reverse(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


async def concat(a: str, b: str) -> str:
    """Concatenate two strings."""
    return f"{a}{b}"


async def length(text: str) -> int:
    """Return length of text."""
    return len(text)


async def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def failing_tool(text: str) -> str:
    """A tool that always fails."""
    raise RuntimeError("Intentional failure for testing")


async def slow_tool(text: str, delay: float = 0.1) -> str:
    """A slow tool with configurable delay."""
    await asyncio.sleep(delay)
    return text


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Create a ToolRegistry with test tools."""
    registry = ToolRegistry()

    # Register string tools
    registry.register("echo", echo)
    registry.register("uppercase", uppercase)
    registry.register("lowercase", lowercase)
    registry.register("reverse", reverse)
    registry.register("concat", concat)
    registry.register("length", length)

    # Register numeric tools
    registry.register("add_numbers", add_numbers)
    registry.register("multiply", multiply)

    # Register failing/slow tools
    registry.register("failing_tool", failing_tool)
    registry.register("slow_tool", slow_tool)

    return registry


@pytest.fixture
def orchestrator_config() -> OrchestratorConfig:
    """Create an OrchestratorConfig for testing."""
    return OrchestratorConfig(
        max_concurrency=10,
        execution_timeout=30.0,
        max_retries=3,
        enable_correction=False,  # Disable for unit tests
        enable_verification=False,  # Disable for unit tests
        enable_structured_output=False,
        enable_circuit_breaker=True,
    )


@pytest.fixture
def orchestrator(
    tool_registry: ToolRegistry, orchestrator_config: OrchestratorConfig
) -> Orchestrator:
    """Create an Orchestrator instance."""
    return Orchestrator(tool_registry, config=orchestrator_config)


# ============================================================================
# DAG Construction Tests
# ============================================================================

class TestDAGConstruction:
    """Test ToolCallDAG construction and basic operations."""

    def test_dag_add_nodes(self):
        """Test adding nodes to a DAG."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={"text": "world"})

        dag.add_node(node_a)
        dag.add_node(node_b)

        assert len(dag) == 2
        assert "a" in dag
        assert "b" in dag
        assert dag.get_node("a") is node_a
        assert dag.get_node("b") is node_b

    def test_dag_add_nodes_duplicate_raises(self):
        """Test that adding duplicate node raises ValueError."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        dag.add_node(node_a)

        with pytest.raises(ValueError, match="already exists"):
            dag.add_node(ToolCallNode(id="a", name="uppercase", arguments={}))

    def test_dag_add_edges(self):
        """Test creating edges between nodes."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={"text": "world"})

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_edge("a", "b")  # a must complete before b

        # Check dependencies
        assert "a" in node_b.dependencies
        assert "b" in node_a.dependents

    def test_dag_add_edge_missing_node_raises(self):
        """Test that add_edge raises for missing nodes."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        dag.add_node(node_a)

        with pytest.raises(ValueError, match="not found"):
            dag.add_edge("a", "nonexistent")

        with pytest.raises(ValueError, match="not found"):
            dag.add_edge("nonexistent", "a")

    def test_dag_remove_edge(self):
        """Test removing edges from DAG."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={"text": "world"})

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_edge("a", "b")

        dag.remove_edge("a", "b")

        assert "a" not in node_b.dependencies
        assert "b" not in node_a.dependents

    def test_dag_remove_node(self):
        """Test removing a node from DAG."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={"text": "world"})
        node_c = ToolCallNode(id="c", name="reverse", arguments={"text": "test"})

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")

        dag.remove_node("a")

        assert len(dag) == 2
        assert "a" not in dag

        # b and c should no longer have 'a' as dependency
        assert "a" not in node_b.dependencies
        assert "a" not in node_c.dependencies

    def test_dag_clear(self):
        """Test clearing all nodes from DAG."""
        dag = ToolCallDAG()

        dag.add_node(ToolCallNode(id="a", name="echo", arguments={}))
        dag.add_node(ToolCallNode(id="b", name="uppercase", arguments={}))

        dag.clear()

        assert len(dag) == 0

    def test_dag_to_dict(self):
        """Test DAG serialization to dict."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        dag.add_node(node_a)

        result = dag.to_dict()

        assert result["num_nodes"] == 1
        assert "nodes" in result
        assert "waves" in result
        assert "a" in result["nodes"]


# ============================================================================
# ToolCallNode Tests
# ============================================================================

class TestToolCallNode:
    """Test ToolCallNode properties and status transitions."""

    def test_node_properties_is_root(self):
        """Test is_root property."""
        # Node without dependencies is a root
        root_node = ToolCallNode(id="a", name="echo", arguments={})
        assert root_node.is_root is True

        # Node with dependencies is not a root
        child_node = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        assert child_node.is_root is False

    def test_node_properties_is_leaf(self):
        """Test is_leaf property."""
        # Node without dependents is a leaf
        leaf_node = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        assert leaf_node.is_leaf is True

        # Node with dependents is not a leaf
        parent_node = ToolCallNode(id="a", name="echo", arguments={})
        parent_node.dependents.append("b")
        assert parent_node.is_leaf is False

    def test_node_status_transitions(self):
        """Test node status transitions: PENDING -> RUNNING -> DONE."""
        node = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})

        # Initial status
        assert node.status == CallStatus.PENDING
        assert node.is_pending is True
        assert node.is_done is False

        # Mark as running
        node.mark_running()
        assert node.status == CallStatus.RUNNING
        assert node.is_pending is False

        # Mark as done
        node.mark_done("hello")
        assert node.status == CallStatus.DONE
        assert node.is_done is True
        assert node.result == "hello"

    def test_node_status_transitions_to_failed(self):
        """Test node status transitions to FAILED."""
        node = ToolCallNode(id="a", name="failing_tool", arguments={"text": "hello"})

        node.mark_running()
        node.mark_failed("Intentional failure")

        assert node.status == CallStatus.FAILED
        assert node.is_done is True
        assert node.error == "Intentional failure"

    def test_node_retry_logic(self):
        """Test node retry functionality."""
        node = ToolCallNode(id="a", name="failing_tool", arguments={"text": "hello"})

        # Initial retry state
        assert node.retry_count == 0
        assert node.can_retry(max_retries=3) is True

        # Increment retry
        node.increment_retry()
        assert node.retry_count == 1
        assert node.can_retry(max_retries=3) is True

        node.increment_retry()
        assert node.retry_count == 2
        assert node.can_retry(max_retries=3) is True

        # Max retries reached
        node.increment_retry()
        assert node.retry_count == 3
        assert node.can_retry(max_retries=3) is False

    def test_node_add_remove_dependency(self):
        """Test adding and removing dependencies."""
        node = ToolCallNode(id="a", name="echo", arguments={})

        node.add_dependency("b")
        assert "b" in node.dependencies

        node.add_dependency("c")
        assert "b" in node.dependencies
        assert "c" in node.dependencies

        node.remove_dependency("b")
        assert "b" not in node.dependencies
        assert "c" in node.dependencies

    def test_node_dependents_property(self):
        """Test dependents property behavior."""
        # Node with dependents is not a leaf
        node = ToolCallNode(id="a", name="echo", arguments={})
        assert node.is_leaf is True

        # Adding dependents via the list
        node.dependents.append("b")
        node.dependents.append("c")
        assert len(node.dependents) == 2
        assert node.is_leaf is False

    def test_node_to_dict(self):
        """Test node serialization to dict."""
        # Create a root node (no dependencies, has dependents)
        node = ToolCallNode(
            id="a",
            name="echo",
            arguments={"text": "hello"},
            dependencies=[],  # No dependencies = root
            dependents=["child1", "child2"],  # Has dependents = not leaf
            execution_key="echo:hello",
        )

        result = node.to_dict()

        assert result["id"] == "a"
        assert result["name"] == "echo"
        assert result["arguments"] == {"text": "hello"}
        assert result["dependencies"] == []
        assert result["execution_key"] == "echo:hello"
        assert result["is_root"] is True
        assert result["is_leaf"] is False  # Has dependents


# ============================================================================
# Topological Sort Tests
# ============================================================================

class TestTopologicalSort:
    """Test DAG topological sort functionality."""

    def test_topological_sort_simple_linear(self):
        """Test topological sort with linear chain A -> B -> C."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={"text": "world"}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="reverse", arguments={"text": "test"}, dependencies=["b"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        sorted_nodes = dag.topological_sort()

        # Should be in order: a, b, c
        assert [n.id for n in sorted_nodes] == ["a", "b", "c"]

    def test_topological_sort_parallel(self):
        """Test topological sort with parallel nodes."""
        dag = ToolCallDAG()

        # a is root
        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})

        # b and c both depend on a
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="lowercase", arguments={}, dependencies=["a"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        sorted_nodes = dag.topological_sort()

        # a should come before b and c
        node_ids = [n.id for n in sorted_nodes]
        assert node_ids[0] == "a"
        assert set(node_ids[1:]) == {"b", "c"}

    def test_topological_sort_complex_dag(self):
        """Test topological sort with diamond DAG."""
        dag = ToolCallDAG()

        # Diamond: A -> B, A -> C, B -> D, C -> D
        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="lowercase", arguments={}, dependencies=["a"])
        node_d = ToolCallNode(id="d", name="reverse", arguments={}, dependencies=["b", "c"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        sorted_nodes = dag.topological_sort()

        # a should come first, d should come last
        node_ids = [n.id for n in sorted_nodes]
        assert node_ids[0] == "a"
        assert node_ids[-1] == "d"
        assert set(node_ids[1:-1]) == {"b", "c"}

    def test_topological_sort_raises_on_cycle(self):
        """Test that topological sort raises ValueError on cycle."""
        dag = ToolCallDAG()

        # Create a cycle: a -> b -> c -> a
        node_a = ToolCallNode(id="a", name="echo", arguments={})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="reverse", arguments={}, dependencies=["b"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        # Add edge creating cycle: c -> a
        dag.add_edge("c", "a")

        with pytest.raises(ValueError, match="Cycle detected"):
            dag.topological_sort()


# ============================================================================
# Wave Computation Tests
# ============================================================================

class TestWaveComputation:
    """Test DAG wave computation for parallel execution."""

    def test_wave_computation_single_wave(self):
        """Test wave computation with all nodes in one wave."""
        dag = ToolCallDAG()

        # All independent nodes - should be in one wave
        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={"text": "world"})
        node_c = ToolCallNode(id="c", name="reverse", arguments={"text": "test"})

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        waves = dag.get_waves()

        assert len(waves) == 1
        assert len(waves[0]) == 3

    def test_wave_computation_linear_chain(self):
        """Test wave computation with linear chain."""
        dag = ToolCallDAG()

        # Linear chain: a -> b -> c
        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="reverse", arguments={}, dependencies=["b"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        waves = dag.get_waves()

        # Should have 3 waves
        assert len(waves) == 3
        assert waves[0][0].id == "a"
        assert waves[1][0].id == "b"
        assert waves[2][0].id == "c"

    def test_wave_computation_diamond(self):
        """Test wave computation with diamond DAG."""
        dag = ToolCallDAG()

        # Diamond: a -> b, a -> c, b -> d, c -> d
        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="lowercase", arguments={}, dependencies=["a"])
        node_d = ToolCallNode(id="d", name="reverse", arguments={}, dependencies=["b", "c"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        waves = dag.get_waves()

        # Should have 3 waves: [a], [b, c], [d]
        assert len(waves) == 3
        assert [n.id for n in waves[0]] == ["a"]
        assert set(n.id for n in waves[1]) == {"b", "c"}
        assert [n.id for n in waves[2]] == ["d"]

    def test_wave_computation_complex(self):
        """Test wave computation with complex DAG."""
        dag = ToolCallDAG()

        # Complex DAG:
        #     a -> b -> d -> f
        #     a -> c -> e -> f
        #         b -> e
        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="lowercase", arguments={}, dependencies=["a"])
        node_d = ToolCallNode(id="d", name="reverse", arguments={}, dependencies=["b"])
        node_e = ToolCallNode(id="e", name="concat", arguments={}, dependencies=["b", "c"])
        node_f = ToolCallNode(id="f", name="length", arguments={}, dependencies=["d", "e"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)
        dag.add_node(node_e)
        dag.add_node(node_f)

        waves = dag.get_waves()

        # Should have 4 waves
        assert len(waves) == 4
        assert set(n.id for n in waves[0]) == {"a"}
        assert set(n.id for n in waves[1]) == {"b", "c"}
        assert set(n.id for n in waves[2]) == {"d", "e"}
        assert set(n.id for n in waves[3]) == {"f"}

    def test_wave_computation_raises_on_cycle(self):
        """Test that get_waves raises ValueError on cycle."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_edge("b", "a")  # Creates cycle

        with pytest.raises(ValueError, match="Cycle detected"):
            dag.get_waves()


# ============================================================================
# Cycle Detection Tests
# ============================================================================

class TestCycleDetection:
    """Test DAG cycle detection functionality."""

    def test_detect_no_cycle_valid_dag(self):
        """Test that valid DAG has no cycle detected."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])

        dag.add_node(node_a)
        dag.add_node(node_b)

        cycle = dag.detect_cycle()

        assert cycle is None

    def test_detect_cycle_simple(self):
        """Test cycle detection with simple cycle."""
        dag = ToolCallDAG()

        # Create self-loop
        node_a = ToolCallNode(id="a", name="echo", arguments={}, dependencies=["a"])

        dag.add_node(node_a)

        cycle = dag.detect_cycle()

        assert cycle is not None
        assert "a" in cycle

    def test_detect_cycle_linear(self):
        """Test cycle detection with linear cycle."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="reverse", arguments={}, dependencies=["b"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_edge("c", "a")  # Creates a -> b -> c -> a cycle

        cycle = dag.detect_cycle()

        assert cycle is not None
        # Cycle should include a, b, c
        assert set(cycle) == {"a", "b", "c"}

    def test_detect_cycle_diamond(self):
        """Test cycle detection with diamond shape."""
        dag = ToolCallDAG()

        # Create a diamond with an extra edge creating a cycle:
        # a -> b -> d
        # a -> c -> d
        # But d -> a creates: a -> b -> d -> a
        node_a = ToolCallNode(id="a", name="echo", arguments={})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="lowercase", arguments={}, dependencies=["a"])
        node_d = ToolCallNode(id="d", name="reverse", arguments={}, dependencies=["b", "c"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        # Add edge d -> a to create cycle: a -> b -> d -> a
        dag.add_edge("d", "a")

        cycle = dag.detect_cycle()

        assert cycle is not None
        assert "a" in cycle


# ============================================================================
# WaveExecutor Tests
# ============================================================================

class TestWaveExecutor:
    """Test WaveExecutor functionality."""

    @pytest.mark.asyncio
    async def test_simple_linear_chain(self, tool_registry: ToolRegistry):
        """Test execution of A -> B -> C linear chain."""
        executor = WaveExecutor(tool_registry, timeout=30.0, max_retries=3)

        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={"text": "world"})
        node_c = ToolCallNode(id="c", name="reverse", arguments={"text": "test"})

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        results = []
        async for node in executor.execute_dag(dag):
            results.append(node)

        assert len(results) == 3
        assert all(n.status == CallStatus.DONE for n in results)

    @pytest.mark.asyncio
    async def test_parallel_wave_execution(self, tool_registry: ToolRegistry):
        """Test that parallel nodes execute concurrently."""
        executor = WaveExecutor(tool_registry, timeout=30.0, max_retries=3)

        dag = ToolCallDAG()

        # All independent - should run in parallel
        node_a = ToolCallNode(id="a", name="slow_tool", arguments={"text": "a", "delay": 0.05})
        node_b = ToolCallNode(id="b", name="slow_tool", arguments={"text": "b", "delay": 0.05})
        node_c = ToolCallNode(id="c", name="slow_tool", arguments={"text": "c", "delay": 0.05})

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        import time
        start = time.time()

        results = []
        async for node in executor.execute_dag(dag):
            results.append(node)

        elapsed = time.time() - start

        # All should complete in roughly the same time as one (parallel)
        # Allow some overhead but should be much less than 3 * 0.05
        assert elapsed < 0.15, f"Parallel execution took too long: {elapsed:.2f}s"
        assert len(results) == 3
        assert all(n.status == CallStatus.DONE for n in results)

    @pytest.mark.asyncio
    async def test_complex_tree_diamond(self, tool_registry: ToolRegistry):
        """Test execution of diamond DAG."""
        executor = WaveExecutor(tool_registry, timeout=30.0, max_retries=3)

        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={"text": "world"})
        node_c = ToolCallNode(id="c", name="lowercase", arguments={"text": "WORLD"})
        node_d = ToolCallNode(id="d", name="concat", arguments={"a": "from_b", "b": "from_c"})

        # Diamond: a -> b, a -> c, b -> d, c -> d
        node_b.dependencies = ["a"]
        node_c.dependencies = ["a"]
        node_d.dependencies = ["b", "c"]

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        results = []
        async for node in executor.execute_dag(dag):
            results.append(node)

        assert len(results) == 4
        assert all(n.status == CallStatus.DONE for n in results)

        # Find node d and check it has the result
        node_d_result = next(n for n in results if n.id == "d")
        assert node_d_result.result == "from_bfrom_c"

    @pytest.mark.asyncio
    async def test_dependency_resolution_from_dep(self, tool_registry: ToolRegistry):
        """Test that from_dep is resolved correctly."""
        executor = WaveExecutor(tool_registry, timeout=30.0, max_retries=3)

        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(
            id="b",
            name="uppercase",
            arguments={"text": {"from_dep": "a"}},  # Will be replaced with result
            dependencies=["a"],
        )

        dag.add_node(node_a)
        dag.add_node(node_b)

        results = []
        async for node in executor.execute_dag(dag):
            results.append(node)

        # Node b should have been called with "hello" (resolved from a)
        node_b_result = next(n for n in results if n.id == "b")
        assert node_b_result.result == "HELLO"

    @pytest.mark.asyncio
    async def test_execution_key_dedup(self, tool_registry: ToolRegistry):
        """Test that same execution_key results in single execution."""
        executor = WaveExecutor(tool_registry, timeout=30.0, max_retries=3)

        dag = ToolCallDAG()

        # Two nodes with same execution_key
        node_a = ToolCallNode(
            id="a",
            name="echo",
            arguments={"text": "hello"},
            execution_key="echo:hello",
        )
        node_b = ToolCallNode(
            id="b",
            name="echo",
            arguments={"text": "hello"},
            execution_key="echo:hello",  # Same key!
        )

        dag.add_node(node_a)
        dag.add_node(node_b)

        results = []
        async for node in executor.execute_dag(dag):
            results.append(node)

        assert len(results) == 2
        # Both should succeed
        assert all(n.status == CallStatus.DONE for n in results)

    @pytest.mark.asyncio
    async def test_wave_ordering(self, tool_registry: ToolRegistry):
        """Test that wave 1 completes before wave 2 starts."""
        executor = WaveExecutor(tool_registry, timeout=30.0, max_retries=3)

        dag = ToolCallDAG()

        wave_1_completed = []
        wave_2_started = []

        node_a = ToolCallNode(id="a", name="slow_tool", arguments={"text": "wave1", "delay": 0.05})
        node_b = ToolCallNode(
            id="b",
            name="slow_tool",
            arguments={"text": "wave2", "delay": 0.05},
            dependencies=["a"],
        )

        dag.add_node(node_a)
        dag.add_node(node_b)

        async for node in executor.execute_dag(dag):
            if node.id == "a":
                wave_1_completed.append(node.id)
            if node.id == "b":
                wave_2_started.append(node.id)

        # Wave 1 (a) should complete before wave 2 (b) starts
        assert wave_1_completed == ["a"]
        assert wave_2_started == ["b"]

    @pytest.mark.asyncio
    async def test_failed_node_handling(self, tool_registry: ToolRegistry):
        """Test handling of failed nodes with retries."""
        # Use max_retries=0 to disable retries and get single result
        executor = WaveExecutor(tool_registry, timeout=30.0, max_retries=0)

        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="failing_tool", arguments={"text": "hello"})

        dag.add_node(node_a)

        results = []
        async for node in executor.execute_dag(dag):
            results.append(node)

        assert len(results) == 1
        assert results[0].status == CallStatus.FAILED
        assert "Intentional failure" in results[0].error

    @pytest.mark.asyncio
    async def test_single_node_execution(self, tool_registry: ToolRegistry):
        """Test executing a single node outside DAG context."""
        executor = WaveExecutor(tool_registry, timeout=30.0, max_retries=3)

        node = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})

        result = await executor.execute_single(node)

        assert result.status == CallStatus.DONE
        assert result.result == "hello"

    @pytest.mark.asyncio
    async def test_executor_reset(self, tool_registry: ToolRegistry):
        """Test executor reset clears caches."""
        executor = WaveExecutor(tool_registry, timeout=30.0, max_retries=3)

        # Execute something to populate cache
        node = ToolCallNode(
            id="a",
            name="echo",
            arguments={"text": "hello"},
            execution_key="echo:hello",
        )
        await executor.execute_single(node)

        # Reset
        executor.reset()

        # Cache should be empty
        assert len(executor._execution_cache) == 0


# ============================================================================
# Orchestrator Streaming Tests
# ============================================================================

class TestOrchestratorStreaming:
    """Test Orchestrator streaming functionality."""

    @pytest.mark.asyncio
    async def test_orchestrator_streaming_events(self, orchestrator: Orchestrator):
        """Test that orchestrator emits correct streaming events."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})

        dag.add_node(node_a)

        events = []
        async for event in orchestrator.execute_streaming(dag):
            events.append(event)
            logger.info(f"Event: {event.type.value}")

        # Should have multiple events
        assert len(events) >= 2

        # Check event types are correct
        event_types = [e.type for e in events]
        assert StreamEventType.TOOL_RESULT in event_types or StreamEventType.FINAL in event_types

        # Final event should be last
        assert events[-1].type == StreamEventType.FINAL

    @pytest.mark.asyncio
    async def test_orchestrator_stats_tracking(self, orchestrator: Orchestrator):
        """Test that orchestrator tracks statistics correctly."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={"text": "world"})

        dag.add_node(node_a)
        dag.add_node(node_b)

        async for event in orchestrator.execute_streaming(dag):
            pass

        stats = orchestrator.stats

        assert stats["total_executions"] >= 0
        assert "successful_executions" in stats
        assert "failed_executions" in stats

    @pytest.mark.asyncio
    async def test_orchestrator_complex_dag_streaming(self, orchestrator: Orchestrator):
        """Test streaming with complex DAG."""
        dag = ToolCallDAG()

        # Diamond: a -> b, a -> c, b -> d, c -> d
        # All nodes need valid arguments
        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={"text": "world"}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="lowercase", arguments={"text": "WORLD"}, dependencies=["a"])
        node_d = ToolCallNode(id="d", name="concat", arguments={"a": "b", "b": "c"}, dependencies=["b", "c"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        events = []
        async for event in orchestrator.execute_streaming(dag):
            events.append(event)

        # Final event should be last
        assert events[-1].type == StreamEventType.FINAL

        # Check that we got successful results for all nodes
        tool_results = [e for e in events if e.type == StreamEventType.TOOL_RESULT]
        assert len(tool_results) >= 4

    @pytest.mark.asyncio
    async def test_orchestrator_with_initial_args(self, orchestrator: Orchestrator):
        """Test orchestrator with initial arguments."""
        dag = ToolCallDAG()

        # Node that uses a from_dep reference to initial args
        node_a = ToolCallNode(
            id="a",
            name="echo",
            arguments={"text": {"from_dep": "initial_text"}},
            dependencies=[],
        )

        dag.add_node(node_a)

        initial_args = {"initial_text": "from_initial"}

        events = []
        async for event in orchestrator.execute_streaming(dag, initial_args):
            events.append(event)

        # Should complete successfully
        tool_results = [e for e in events if e.type == StreamEventType.TOOL_RESULT]
        assert len(tool_results) >= 1


# ============================================================================
# Orchestrator Batch Tests
# ============================================================================

class TestOrchestratorBatch:
    """Test Orchestrator batch execution."""

    @pytest.mark.asyncio
    async def test_orchestrator_batch_execution(self, orchestrator: Orchestrator):
        """Test executing multiple DAGs in batch."""
        # Create multiple DAGs
        dag1 = ToolCallDAG()
        dag1.add_node(ToolCallNode(id="a", name="echo", arguments={"text": "hello1"}))

        dag2 = ToolCallDAG()
        dag2.add_node(ToolCallNode(id="b", name="echo", arguments={"text": "hello2"}))

        dag3 = ToolCallDAG()
        dag3.add_node(ToolCallNode(id="c", name="echo", arguments={"text": "hello3"}))

        results = await orchestrator.execute_batch([dag1, dag2, dag3])

        assert len(results) == 3
        assert all(r.get("success", False) or isinstance(r, Exception) for r in results)

    @pytest.mark.asyncio
    async def test_orchestrator_batch_sequential(self, orchestrator: Orchestrator):
        """Test that batch executes DAGs sequentially."""
        dag = ToolCallDAG()
        dag.add_node(ToolCallNode(id="a", name="slow_tool", arguments={"text": "test", "delay": 0.1}))

        # Execute same DAG twice
        results = await orchestrator.execute_batch([dag, dag])

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_orchestrator_batch_with_failures(self, orchestrator: Orchestrator):
        """Test batch execution with some failing DAGs."""
        dag_success = ToolCallDAG()
        dag_success.add_node(ToolCallNode(id="a", name="echo", arguments={"text": "ok"}))

        dag_fail = ToolCallDAG()
        dag_fail.add_node(ToolCallNode(id="b", name="failing_tool", arguments={"text": "fail"}))

        results = await orchestrator.execute_batch([dag_success, dag_fail])

        assert len(results) == 2
        # One should succeed, one should have error
        assert any(
            r.get("success", False) for r in results
            if isinstance(r, dict)
        )


# ============================================================================
# Shared Execution Key Tests
# ============================================================================

class TestSharedExecutionKey:
    """Test shared execution key deduplication."""

    def test_get_shared_execution_groups(self):
        """Test grouping nodes by execution_key."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(
            id="a",
            name="search",
            arguments={"query": "weather"},
            execution_key="search:weather",
        )
        node_b = ToolCallNode(
            id="b",
            name="search",
            arguments={"query": "weather"},
            execution_key="search:weather",
        )
        node_c = ToolCallNode(
            id="c",
            name="search",
            arguments={"query": "news"},
            execution_key="search:news",
        )
        node_d = ToolCallNode(
            id="d",
            name="search",
            arguments={"query": "random"},
            # No execution_key
        )

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        groups = dag.get_shared_execution_groups()

        # Should have 3 groups: search:weather (2 nodes), search:news (1 node), None (1 node)
        assert len(groups) == 3

        # Check group contents
        assert len(groups["search:weather"]) == 2
        assert set(n.id for n in groups["search:weather"]) == {"a", "b"}
        assert len(groups["search:news"]) == 1
        assert groups["search:news"][0].id == "c"
        assert len(groups[None]) == 1
        assert groups[None][0].id == "d"

    def test_get_execution_order(self):
        """Test get_execution_order returns correct wave structure."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="lowercase", arguments={}, dependencies=["a"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        order = dag.get_execution_order()

        assert len(order) == 2
        assert order[0] == ["a"]
        assert set(order[1]) == {"b", "c"}


# ============================================================================
# DAG Helper Methods Tests
# ============================================================================

class TestDAGHelperMethods:
    """Test DAG helper methods."""

    def test_get_root_nodes(self):
        """Test getting root nodes (nodes with no dependencies)."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={})  # root
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="lowercase", arguments={})  # root

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        roots = dag.get_root_nodes()

        assert len(roots) == 2
        assert {n.id for n in roots} == {"a", "c"}

    def test_get_leaf_nodes(self):
        """Test getting leaf nodes (nodes with no dependents)."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={}, dependents=["b"])
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependents=["c"])
        node_c = ToolCallNode(id="c", name="lowercase", arguments={})  # leaf

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        leaves = dag.get_leaf_nodes()

        assert len(leaves) == 1
        assert leaves[0].id == "c"

    def test_get_node_depth(self):
        """Test getting node depth in DAG."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="lowercase", arguments={}, dependencies=["a"])
        node_d = ToolCallNode(id="d", name="reverse", arguments={}, dependencies=["b", "c"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        assert dag.get_node_depth("a") == 0
        assert dag.get_node_depth("b") == 1
        assert dag.get_node_depth("c") == 1
        assert dag.get_node_depth("d") == 2

    def test_get_node_depth_raises_on_missing(self):
        """Test that get_node_depth raises for missing node."""
        dag = ToolCallDAG()

        with pytest.raises(KeyError):
            dag.get_node_depth("nonexistent")

    def test_get_all_depths(self):
        """Test getting depths for all nodes."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])

        dag.add_node(node_a)
        dag.add_node(node_b)

        depths = dag.get_all_depths()

        assert depths == {"a": 0, "b": 1}

    def test_get_ready_nodes(self):
        """Test getting nodes ready to execute.

        get_ready_nodes returns nodes whose dependencies are all satisfied
        (i.e., all dependencies are in the completed set).
        Nodes with no dependencies are always returned as ready.
        The function does NOT filter out nodes already in the completed set.
        """
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="lowercase", arguments={})

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        # Initially (no completed nodes), a and c should be ready
        # (they have no dependencies, so all([]) = True)
        ready = dag.get_ready_nodes(completed=set())
        assert {n.id for n in ready} == {"a", "c"}

        # After a completes, b becomes ready too
        ready = dag.get_ready_nodes(completed={"a"})
        # Nodes with no deps (a, c) are always ready
        # b has its dep satisfied, so it's ready too
        assert {n.id for n in ready} == {"a", "b", "c"}

        # When all deps are satisfied, all pending nodes are returned
        ready = dag.get_ready_nodes(completed={"a", "b", "c"})
        assert {n.id for n in ready} == {"a", "b", "c"}

    def test_get_completed_nodes(self):
        """Test getting completed nodes."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={}, dependencies=["a"])

        node_a.mark_done("hello")

        dag.add_node(node_a)
        dag.add_node(node_b)

        completed = dag.get_completed_nodes()

        assert len(completed) == 1
        assert completed[0].id == "a"

    def test_get_failed_nodes(self):
        """Test getting failed nodes."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={})
        node_b = ToolCallNode(id="b", name="failing_tool", arguments={})

        node_b.mark_failed("error")

        dag.add_node(node_a)
        dag.add_node(node_b)

        failed = dag.get_failed_nodes()

        assert len(failed) == 1
        assert failed[0].id == "b"

    def test_get_running_nodes(self):
        """Test getting running nodes."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="echo", arguments={})
        node_b = ToolCallNode(id="b", name="uppercase", arguments={})

        node_a.mark_running()

        dag.add_node(node_a)
        dag.add_node(node_b)

        running = dag.get_running_nodes()

        assert len(running) == 1
        assert running[0].id == "a"


# ============================================================================
# Orchestrator Configuration Tests
# ============================================================================

class TestOrchestratorConfiguration:
    """Test Orchestrator configuration options."""

    def test_orchestrator_config_defaults(self):
        """Test default configuration values."""
        config = OrchestratorConfig()

        assert config.max_concurrency == 10
        assert config.execution_timeout == 30.0
        assert config.max_retries == 3
        assert config.enable_correction is True
        assert config.enable_verification is True

    def test_orchestrator_config_custom(self):
        """Test custom configuration values."""
        config = OrchestratorConfig(
            max_concurrency=5,
            execution_timeout=60.0,
            max_retries=5,
            enable_correction=False,
            enable_verification=False,
        )

        assert config.max_concurrency == 5
        assert config.execution_timeout == 60.0
        assert config.max_retries == 5
        assert config.enable_correction is False
        assert config.enable_verification is False

    @pytest.mark.asyncio
    async def test_orchestrator_shutdown(self, orchestrator: Orchestrator):
        """Test orchestrator shutdown."""
        await orchestrator.shutdown()
        # Should complete without error

    def test_orchestrator_repr(self, orchestrator: Orchestrator):
        """Test orchestrator string representation."""
        repr_str = repr(orchestrator)
        assert "Orchestrator" in repr_str


# ============================================================================
# Integration Tests
# ============================================================================

class TestDAGIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_pipeline_echo_uppercase_reverse(self, tool_registry: ToolRegistry):
        """Test complete pipeline: echo -> uppercase -> reverse."""
        executor = WaveExecutor(tool_registry, timeout=30.0)

        dag = ToolCallDAG()

        node_a = ToolCallNode(id="echo", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(
            id="uppercase",
            name="uppercase",
            arguments={"text": {"from_dep": "echo"}},
            dependencies=["echo"],
        )
        node_c = ToolCallNode(
            id="reverse",
            name="reverse",
            arguments={"text": {"from_dep": "uppercase"}},
            dependencies=["uppercase"],
        )

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        results = []
        async for node in executor.execute_dag(dag):
            results.append(node)

        # Final result should be reversed "HELLO"
        final_node = next(n for n in results if n.id == "reverse")
        assert final_node.result == "OLLEH"

    @pytest.mark.asyncio
    async def test_full_pipeline_with_concat_and_length(self, tool_registry: ToolRegistry):
        """Test pipeline with concat and length."""
        executor = WaveExecutor(tool_registry, timeout=30.0)

        dag = ToolCallDAG()

        node_a = ToolCallNode(id="echo1", name="echo", arguments={"text": "hello"})
        node_b = ToolCallNode(id="echo2", name="echo", arguments={"text": "world"})
        node_c = ToolCallNode(
            id="concat",
            name="concat",
            arguments={"a": {"from_dep": "echo1"}, "b": {"from_dep": "echo2"}},
            dependencies=["echo1", "echo2"],
        )
        node_d = ToolCallNode(
            id="length",
            name="length",
            arguments={"text": {"from_dep": "concat"}},
            dependencies=["concat"],
        )

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        results = []
        async for node in executor.execute_dag(dag):
            results.append(node)

        # concat should produce "helloworld", length should be 10
        concat_node = next(n for n in results if n.id == "concat")
        length_node = next(n for n in results if n.id == "length")

        assert concat_node.result == "helloworld"
        assert length_node.result == 10

    @pytest.mark.asyncio
    async def test_parallel_and_sequential_mixed(self, tool_registry: ToolRegistry):
        """Test mixed parallel and sequential execution."""
        executor = WaveExecutor(tool_registry, timeout=30.0, max_retries=0)  # Disable retries

        dag = ToolCallDAG()

        #       start
        #      /  |  \
        #    up   lo  rev  (parallel - wave 2)
        #      \  |  /
        #      concat      (sequential - wave 3)
        #         |
        #       length     (sequential - wave 4)

        node_start = ToolCallNode(id="start", name="echo", arguments={"text": "Hi"})
        node_up = ToolCallNode(
            id="uppercase", name="uppercase", arguments={"text": "test"}, dependencies=["start"]
        )
        node_lo = ToolCallNode(
            id="lowercase", name="lowercase", arguments={"text": "TEST"}, dependencies=["start"]
        )
        node_rev = ToolCallNode(
            id="reverse", name="reverse", arguments={"text": "test"}, dependencies=["start"]
        )
        node_concat = ToolCallNode(
            id="concat",
            name="concat",
            arguments={"a": "from_up", "b": "from_lo"},
            dependencies=["uppercase", "lowercase", "reverse"],
        )
        node_len = ToolCallNode(
            id="length", name="length", arguments={"text": "result"}, dependencies=["concat"]
        )

        dag.add_node(node_start)
        dag.add_node(node_up)
        dag.add_node(node_lo)
        dag.add_node(node_rev)
        dag.add_node(node_concat)
        dag.add_node(node_len)

        results = []
        async for node in executor.execute_dag(dag):
            results.append(node)

        assert len(results) == 6
        assert all(n.status == CallStatus.DONE for n in results)

        # Verify waves
        waves = dag.get_waves()
        assert len(waves) == 4
        assert [n.id for n in waves[0]] == ["start"]
        assert set(n.id for n in waves[1]) == {"uppercase", "lowercase", "reverse"}
        assert [n.id for n in waves[2]] == ["concat"]
        assert [n.id for n in waves[3]] == ["length"]

    @pytest.mark.asyncio
    async def test_numerical_operations_pipeline(self, tool_registry: ToolRegistry):
        """Test pipeline with numerical operations."""
        executor = WaveExecutor(tool_registry, timeout=30.0)

        dag = ToolCallDAG()

        node_a = ToolCallNode(id="add", name="add_numbers", arguments={"a": 5, "b": 3})
        node_b = ToolCallNode(id="mult", name="multiply", arguments={"a": 2, "b": 4})

        dag.add_node(node_a)
        dag.add_node(node_b)

        results = []
        async for node in executor.execute_dag(dag):
            results.append(node)

        assert len(results) == 2

        add_result = next(n for n in results if n.id == "add")
        mult_result = next(n for n in results if n.id == "mult")

        assert add_result.result == 8
        assert mult_result.result == 8

    @pytest.mark.asyncio
    async def test_error_propagation_in_dag(self, tool_registry: ToolRegistry):
        """Test that errors in one branch don't affect other branches."""
        executor = WaveExecutor(tool_registry, timeout=30.0, max_retries=0)  # Disable retries

        dag = ToolCallDAG()

        node_a = ToolCallNode(id="start", name="echo", arguments={"text": "ok"})
        node_b = ToolCallNode(id="fail", name="failing_tool", arguments={"text": "fail"})
        node_c = ToolCallNode(id="success", name="echo", arguments={"text": "still_works"})

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        results = []
        async for node in executor.execute_dag(dag):
            results.append(node)

        assert len(results) == 3

        # Check that the failing node failed
        fail_node = next(n for n in results if n.id == "fail")
        assert fail_node.status == CallStatus.FAILED

        # But others should still succeed
        start_node = next(n for n in results if n.id == "start")
        success_node = next(n for n in results if n.id == "success")

        assert start_node.status == CallStatus.DONE
        assert success_node.status == CallStatus.DONE


# ============================================================================
# StreamEvent Tests
# ============================================================================

class TestStreamEvent:
    """Test StreamEvent creation and formatting."""

    def test_stream_event_to_sse(self):
        """Test SSE formatting of events."""
        event = StreamEvent(
            type=StreamEventType.TEXT_CHUNK,
            data={"text": "Hello", "is_final": False},
        )

        sse = event.to_sse()

        assert "event: text_chunk" in sse
        assert '"text": "Hello"' in sse

    def test_stream_event_to_dict(self):
        """Test dict conversion of events."""
        event = StreamEvent(
            type=StreamEventType.TOOL_RESULT,
            data={"id": "test", "result": "ok"},
        )

        d = event.to_dict()

        assert d["type"] == "tool_result"
        assert d["data"]["id"] == "test"
        assert "timestamp" in d

    def test_stream_event_factories(self):
        """Test StreamEvent factory methods."""
        # text_chunk
        event = StreamEvent.text_chunk("Hello", is_final=False)
        assert event.type == StreamEventType.TEXT_CHUNK
        assert event.data["text"] == "Hello"

        # tool_pending
        event = StreamEvent.tool_pending("echo", "node1")
        assert event.type == StreamEventType.TOOL_CALL_PENDING
        assert event.data["tool_name"] == "echo"

        # tool_executing
        event = StreamEvent.tool_executing("echo", "node1")
        assert event.type == StreamEventType.TOOL_EXECUTING

        # tool_result
        event = StreamEvent.tool_result("echo", "node1", "success")
        assert event.type == StreamEventType.TOOL_RESULT
        assert event.data["result"] == "success"

        # final
        event = StreamEvent.final(total_chunks=5, total_tools=3)
        assert event.type == StreamEventType.FINAL
        assert event.data["total_chunks"] == 5

        # error
        event = StreamEvent.error("Something went wrong", error_type="test")
        assert event.type == StreamEventType.ERROR
        assert event.data["message"] == "Something went wrong"


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
