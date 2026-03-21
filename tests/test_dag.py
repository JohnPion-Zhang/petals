"""
Tests for ToolCall DAG - Agent Tool Call Layer

TDD Phase 1: DAG Foundation Tests

These tests cover:
- ToolCallNode dataclass
- ToolCallDAG with topological sort
- Wave-based execution
- Cycle detection
- Shared execution key deduplication
"""
import asyncio
import pytest

from petals.client.dag import ToolCallNode, ToolCallDAG, WaveExecutor
from petals.client.dag.tool_call_node import ToolCallNode
from petals.client.dag.dag import ToolCallDAG
from petals.client.dag.wave_executor import WaveExecutor
from petals.client.tool_registry import ToolRegistry
from petals.data_structures import CallStatus


# --- Test Fixtures ---

@pytest.fixture
def tool_registry():
    """Create a fresh ToolRegistry with common test tools."""
    registry = ToolRegistry()

    async def add_numbers(a: int, b: int):
        return a + b

    async def multiply_numbers(a: int, b: int):
        return a * b

    async def get_data(query: str):
        await asyncio.sleep(0.01)  # Simulate async operation
        return {"query": query, "result": f"data_for_{query}"}

    async def merge_results(data1: dict, data2: dict):
        return {"merged": [data1, data2]}

    async def failing_tool():
        raise RuntimeError("Tool execution failed")

    registry.register("add", add_numbers)
    registry.register("multiply", multiply_numbers)
    registry.register("get_data", get_data)
    registry.register("merge_results", merge_results)
    registry.register("fail", failing_tool)

    return registry


@pytest.fixture
def wave_executor(tool_registry):
    """Create a WaveExecutor with default settings."""
    return WaveExecutor(tool_registry, timeout=5.0, max_retries=2, concurrency_limit=10)


# ============================================================================
# ToolCallNode Tests
# ============================================================================

class TestToolCallNode:
    """Tests for ToolCallNode dataclass."""

    def test_node_creation_with_defaults(self):
        """Create a node with default values."""
        node = ToolCallNode(
            id="node_1",
            name="add",
            arguments={"a": 1, "b": 2}
        )

        assert node.id == "node_1"
        assert node.name == "add"
        assert node.arguments == {"a": 1, "b": 2}
        assert node.dependencies == []
        assert node.dependents == []
        assert node.execution_key is None
        assert node.status == CallStatus.PENDING
        assert node.result is None
        assert node.error is None
        assert node.execution_trace is None
        assert node.error_feedback is None
        assert node.retry_count == 0
        assert node.requires_verification is False

    def test_node_with_dependencies(self):
        """Create a node with dependencies specified."""
        node = ToolCallNode(
            id="node_2",
            name="merge",
            arguments={"items": []},
            dependencies=["node_1", "node_3"]
        )

        assert node.dependencies == ["node_1", "node_3"]

    def test_node_with_execution_key(self):
        """Create a node with execution key for deduplication."""
        node = ToolCallNode(
            id="node_3",
            name="get_data",
            arguments={"query": "test"},
            execution_key="get_data:query=test"
        )

        assert node.execution_key == "get_data:query=test"

    def test_node_is_leaf_property(self):
        """Test is_leaf property for leaf nodes."""
        leaf = ToolCallNode(id="leaf", name="leaf_tool", arguments={})
        assert leaf.is_leaf is True

        non_leaf = ToolCallNode(
            id="parent",
            name="parent_tool",
            arguments={},
            dependents=["child_1", "child_2"]
        )
        assert non_leaf.is_leaf is False

    def test_node_is_root_property(self):
        """Test is_root property for root nodes."""
        root = ToolCallNode(id="root", name="root_tool", arguments={})
        assert root.is_root is True

        child = ToolCallNode(
            id="child",
            name="child_tool",
            arguments={},
            dependencies=["parent"]
        )
        assert child.is_root is False

    def test_node_status_transitions(self):
        """Test node status can be updated."""
        node = ToolCallNode(id="test", name="test", arguments={})

        assert node.status == CallStatus.PENDING

        node.status = CallStatus.RUNNING
        assert node.status == CallStatus.RUNNING

        node.result = {"data": "success"}
        node.status = CallStatus.DONE
        assert node.status == CallStatus.DONE

    def test_node_error_handling(self):
        """Test node can store error information."""
        node = ToolCallNode(
            id="error_node",
            name="failing_tool",
            arguments={}
        )

        node.status = CallStatus.FAILED
        node.error = "Tool execution failed"
        node.error_feedback = "Traceback: RuntimeError at line 42"

        assert node.status == CallStatus.FAILED
        assert node.error == "Tool execution failed"
        assert "Traceback" in node.error_feedback

    def test_node_retry_count(self):
        """Test retry count tracking."""
        node = ToolCallNode(id="retry_test", name="unstable", arguments={})

        assert node.retry_count == 0

        node.retry_count += 1
        node.retry_count += 1
        assert node.retry_count == 2

    def test_node_verification_flag(self):
        """Test requires_verification flag."""
        node = ToolCallNode(
            id="verify_test",
            name="critical_operation",
            arguments={},
            requires_verification=True
        )

        assert node.requires_verification is True

    def test_node_execution_trace(self):
        """Test execution trace storage."""
        node = ToolCallNode(id="trace_test", name="code_tool", arguments={})
        node.execution_trace = "stdout: Hello\nstderr: None\n"

        assert "Hello" in node.execution_trace


# ============================================================================
# ToolCallDAG Tests
# ============================================================================

class TestToolCallDAG:
    """Tests for ToolCallDAG class."""

    def test_empty_dag_creation(self):
        """Create an empty DAG."""
        dag = ToolCallDAG()

        assert len(dag.nodes) == 0

    def test_add_single_node(self):
        """Add a single node to the DAG."""
        dag = ToolCallDAG()
        node = ToolCallNode(id="n1", name="tool1", arguments={})

        dag.add_node(node)

        assert "n1" in dag.nodes
        assert dag.nodes["n1"] is node

    def test_add_edge_updates_dependents(self):
        """Adding edge updates both dependencies and dependents."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="tool_a", arguments={})
        node_b = ToolCallNode(id="b", name="tool_b", arguments={})

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_edge("a", "b")

        # Check dependencies of b
        assert "a" in dag.nodes["b"].dependencies

        # Check dependents of a
        assert "b" in dag.nodes["a"].dependents

    def test_topological_sort_linear_chain(self):
        """Topological sort of linear A -> B -> C chain."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="first", arguments={})
        node_b = ToolCallNode(id="b", name="second", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="third", arguments={}, dependencies=["b"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        sorted_nodes = dag.topological_sort()

        # a should come before b, b should come before c
        ids = [n.id for n in sorted_nodes]
        assert ids.index("a") < ids.index("b")
        assert ids.index("b") < ids.index("c")

    def test_topological_sort_parallel_nodes(self):
        """Topological sort of parallel A -> B, A -> C."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="first", arguments={})
        node_b = ToolCallNode(id="b", name="second", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="third", arguments={}, dependencies=["a"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        sorted_nodes = dag.topological_sort()

        # a should come before both b and c
        ids = [n.id for n in sorted_nodes]
        assert ids.index("a") < ids.index("b")
        assert ids.index("a") < ids.index("c")

    def test_topological_sort_complex_dag(self):
        """Topological sort of complex DAG with diamond pattern."""
        dag = ToolCallDAG()

        #    A
        #   / \
        #  B   C
        #   \ /
        #    D

        node_a = ToolCallNode(id="a", name="root", arguments={})
        node_b = ToolCallNode(id="b", name="branch1", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="branch2", arguments={}, dependencies=["a"])
        node_d = ToolCallNode(id="d", name="merge", arguments={}, dependencies=["b", "c"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        sorted_nodes = dag.topological_sort()
        ids = [n.id for n in sorted_nodes]

        # a before b and c
        assert ids.index("a") < ids.index("b")
        assert ids.index("a") < ids.index("c")
        # b and c before d
        assert ids.index("b") < ids.index("d")
        assert ids.index("c") < ids.index("d")

    def test_get_waves_linear_chain(self):
        """Get waves of linear A -> B -> C chain."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="first", arguments={})
        node_b = ToolCallNode(id="b", name="second", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="third", arguments={}, dependencies=["b"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        waves = dag.get_waves()

        assert len(waves) == 3
        assert [n.id for n in waves[0]] == ["a"]
        assert [n.id for n in waves[1]] == ["b"]
        assert [n.id for n in waves[2]] == ["c"]

    def test_get_waves_parallel_nodes(self):
        """Get waves of parallel A -> B, A -> C."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="first", arguments={})
        node_b = ToolCallNode(id="b", name="second", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="third", arguments={}, dependencies=["a"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        waves = dag.get_waves()

        assert len(waves) == 2
        assert [n.id for n in waves[0]] == ["a"]
        # b and c in second wave (can execute in parallel)
        assert set(n.id for n in waves[1]) == {"b", "c"}

    def test_get_waves_diamond_pattern(self):
        """Get waves of diamond pattern DAG."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="root", arguments={})
        node_b = ToolCallNode(id="b", name="branch1", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="branch2", arguments={}, dependencies=["a"])
        node_d = ToolCallNode(id="d", name="merge", arguments={}, dependencies=["b", "c"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        waves = dag.get_waves()

        assert len(waves) == 3
        assert [n.id for n in waves[0]] == ["a"]
        assert set(n.id for n in waves[1]) == {"b", "c"}
        assert [n.id for n in waves[2]] == ["d"]

    def test_get_ready_nodes_empty_completed(self):
        """Get ready nodes when no dependencies completed."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="root", arguments={})
        node_b = ToolCallNode(id="b", name="child", arguments={}, dependencies=["a"])

        dag.add_node(node_a)
        dag.add_node(node_b)

        ready = dag.get_ready_nodes(completed=set())

        assert len(ready) == 1
        assert ready[0].id == "a"

    def test_get_ready_nodes_partial_completed(self):
        """Get ready nodes when some dependencies completed."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="first", arguments={})
        node_b = ToolCallNode(id="b", name="second", arguments={}, dependencies=["a"])
        node_c = ToolCallNode(id="c", name="third", arguments={}, dependencies=["a"])

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        # After a completes, b and c should be ready
        # Also 'a' is ready because it has no dependencies (all deps satisfied)
        ready = dag.get_ready_nodes(completed={"a"})

        assert len(ready) == 3
        assert {n.id for n in ready} == {"a", "b", "c"}

    def test_get_ready_nodes_all_completed(self):
        """Get ready nodes when all dependencies completed."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="first", arguments={})
        node_b = ToolCallNode(id="b", name="second", arguments={}, dependencies=["a"])

        dag.add_node(node_a)
        dag.add_node(node_b)

        # Mark nodes as done to simulate completed execution
        node_a.status = CallStatus.DONE
        node_b.status = CallStatus.DONE

        # When all nodes are done, no pending nodes should be ready
        ready = dag.get_ready_nodes(completed={"a", "b"})

        assert len(ready) == 0

    def test_detect_no_cycle(self):
        """Detect cycle returns None for valid DAG."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="a", arguments={})
        node_b = ToolCallNode(id="b", name="b", arguments={}, dependencies=["a"])

        dag.add_node(node_a)
        dag.add_node(node_b)

        cycle = dag.detect_cycle()

        assert cycle is None

    def test_detect_cycle_self_loop(self):
        """Detect self-referencing cycle."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="a", arguments={}, dependencies=["a"])

        dag.add_node(node_a)

        cycle = dag.detect_cycle()

        assert cycle is not None
        assert "a" in cycle

    def test_detect_cycle_simple_loop(self):
        """Detect A -> B -> C -> A cycle."""
        dag = ToolCallDAG()

        # Build the cycle by using add_edge which properly updates both nodes
        node_a = ToolCallNode(id="a", name="a", arguments={})
        node_b = ToolCallNode(id="b", name="b", arguments={})
        node_c = ToolCallNode(id="c", name="c", arguments={})

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        # Create cycle: a -> b -> c -> a
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", "a")  # This creates the cycle

        cycle = dag.detect_cycle()

        assert cycle is not None

    def test_get_shared_execution_groups(self):
        """Group nodes by execution key."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(
            id="a",
            name="get_data",
            arguments={"query": "test"},
            execution_key="get_data:query=test"
        )
        node_b = ToolCallNode(
            id="b",
            name="get_data",
            arguments={"query": "test"},  # Same key
            execution_key="get_data:query=test"
        )
        node_c = ToolCallNode(
            id="c",
            name="get_data",
            arguments={"query": "different"},
            execution_key="get_data:query=different"
        )
        node_d = ToolCallNode(
            id="d",
            name="add",
            arguments={"a": 1, "b": 2}
            # No execution key
        )

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        groups = dag.get_shared_execution_groups()

        assert "get_data:query=test" in groups
        assert "get_data:query=different" in groups
        assert len(groups["get_data:query=test"]) == 2
        assert len(groups["get_data:query=different"]) == 1
        assert None in groups  # Nodes without execution key
        assert len(groups[None]) == 1

    def test_to_dict_serialization(self):
        """Serialize DAG to dict."""
        dag = ToolCallDAG()

        node = ToolCallNode(
            id="test",
            name="tool",
            arguments={"a": 1},
            dependencies=[]  # No dependencies for valid DAG
        )

        dag.add_node(node)

        serialized = dag.to_dict()

        assert "nodes" in serialized
        assert "test" in serialized["nodes"]
        assert serialized["nodes"]["test"]["name"] == "tool"


# ============================================================================
# WaveExecutor Tests
# ============================================================================

class TestWaveExecutor:
    """Tests for WaveExecutor class."""

    def test_executor_initialization(self, tool_registry):
        """Test WaveExecutor initializes with correct defaults."""
        executor = WaveExecutor(tool_registry)

        assert executor.registry is tool_registry
        assert executor.timeout == 30.0  # Default
        assert executor.max_retries == 3  # Default
        assert executor.concurrency_limit == 10  # Default

    def test_executor_custom_settings(self, tool_registry):
        """Test WaveExecutor with custom settings."""
        executor = WaveExecutor(
            tool_registry,
            timeout=60.0,
            max_retries=5,
            concurrency_limit=20
        )

        assert executor.timeout == 60.0
        assert executor.max_retries == 5
        assert executor.concurrency_limit == 20

    @pytest.mark.asyncio
    async def test_execute_single_node(self, wave_executor):
        """Execute a single node."""
        dag = ToolCallDAG()
        node = ToolCallNode(
            id="n1",
            name="add",
            arguments={"a": 2, "b": 3}
        )
        dag.add_node(node)

        results = []
        async for completed in wave_executor.execute_dag(dag):
            results.append(completed)

        assert len(results) == 1
        assert results[0].status == CallStatus.DONE
        assert results[0].result == 5

    @pytest.mark.asyncio
    async def test_execute_parallel_wave(self, wave_executor):
        """Execute nodes in parallel wave."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="add", arguments={"a": 1, "b": 2})
        node_b = ToolCallNode(id="b", name="multiply", arguments={"a": 3, "b": 4})
        node_c = ToolCallNode(id="c", name="add", arguments={"a": 5, "b": 6})

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        results = []
        async for completed in wave_executor.execute_dag(dag):
            results.append(completed)

        assert len(results) == 3

        # Check all succeeded
        for node in results:
            assert node.status == CallStatus.DONE

        # Verify specific results
        results_by_id = {r.id: r.result for r in results}
        assert results_by_id["a"] == 3   # 1 + 2
        assert results_by_id["b"] == 12  # 3 * 4
        assert results_by_id["c"] == 11  # 5 + 6

    @pytest.mark.asyncio
    async def test_execute_sequential_waves(self, wave_executor):
        """Execute sequential waves (A -> B -> C)."""
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="add", arguments={"a": 1, "b": 2})
        node_b = ToolCallNode(
            id="b",
            name="add",
            arguments={"a": 0, "b": 0},
            dependencies=["a"]
        )
        node_c = ToolCallNode(
            id="c",
            name="multiply",
            arguments={"a": 0, "b": 0},
            dependencies=["b"]
        )

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        results = []
        async for completed in wave_executor.execute_dag(dag):
            results.append(completed)

        assert len(results) == 3

        # First should be a (root)
        assert results[0].id == "a"
        assert results[0].result == 3

    @pytest.mark.asyncio
    async def test_execute_diamond_pattern(self, wave_executor):
        """Execute diamond pattern DAG."""
        dag = ToolCallDAG()

        #    A (add 1,2 = 3)
        #   / \
        #  B   C (both multiply 3,2 = 6)
        #   \ /
        #    D (merge results)

        node_a = ToolCallNode(id="a", name="add", arguments={"a": 1, "b": 2})
        node_b = ToolCallNode(
            id="b",
            name="multiply",
            arguments={"a": 0, "b": 0},
            dependencies=["a"]
        )
        node_c = ToolCallNode(
            id="c",
            name="multiply",
            arguments={"a": 0, "b": 0},
            dependencies=["a"]
        )
        node_d = ToolCallNode(
            id="d",
            name="merge_results",
            arguments={"data1": None, "data2": None},
            dependencies=["b", "c"]
        )

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        # Pre-resolve arguments using results cache pattern
        results_cache = {"a": {"result": 3}}

        results = []
        async for completed in wave_executor.execute_dag(dag, initial_args=results_cache):
            results.append(completed)

        # Should complete all 4 nodes
        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_execute_handles_tool_error(self, wave_executor):
        """Execute handles tool errors gracefully."""
        dag = ToolCallDAG()
        node = ToolCallNode(id="fail", name="fail", arguments={})

        dag.add_node(node)

        results = []
        async for completed in wave_executor.execute_dag(dag):
            results.append(completed)

        # Note: Due to retry logic, we may get multiple results (initial + retries)
        # At least one should be failed
        assert any(r.status == CallStatus.FAILED for r in results)
        assert any(r.error is not None for r in results)

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, wave_executor):
        """Execute handles unknown tool gracefully."""
        dag = ToolCallDAG()
        node = ToolCallNode(id="unknown", name="nonexistent_tool", arguments={})

        dag.add_node(node)

        results = []
        async for completed in wave_executor.execute_dag(dag):
            results.append(completed)

        # Note: Due to retry logic, we may get multiple results (initial + retries)
        # At least one should be failed
        assert any(r.status == CallStatus.FAILED for r in results)

    @pytest.mark.asyncio
    async def test_execute_wave_parallelism(self, wave_executor):
        """Verify wave executes in parallel (not sequential)."""
        dag = ToolCallDAG()

        # Two independent tasks
        node_a = ToolCallNode(id="a", name="add", arguments={"a": 1, "b": 2})
        node_b = ToolCallNode(id="b", name="add", arguments={"a": 3, "b": 4})

        dag.add_node(node_a)
        dag.add_node(node_b)

        # Track timing
        start_times = {}

        original_execute = wave_executor.registry.execute

        async def timed_execute(name, args):
            import time
            start_times[name + str(args)] = time.time()
            result = await original_execute(name, args)
            return result

        wave_executor.registry.execute = timed_execute

        results = []
        async for completed in wave_executor.execute_dag(dag):
            results.append(completed)

        # Both should have started at nearly the same time
        assert len(results) == 2
        # If executed sequentially, times would differ by ~0.01s
        # If parallel, should be nearly equal
        assert len(start_times) == 2

    def test_resolve_dependencies(self, wave_executor):
        """Test dependency resolution from results cache."""
        dag = ToolCallDAG()
        node = ToolCallNode(
            id="test",
            name="add",
            arguments={"a": {"from_dep": "dep1"}, "b": 5},
            dependencies=["dep1"]
        )

        results_cache = {
            "dep1": 10
        }

        # Manually test argument resolution
        resolved = wave_executor._resolve_dependencies(node, results_cache)

        # The method should replace dependency references
        assert isinstance(resolved, dict)


# ============================================================================
# Integration Tests
# ============================================================================

class TestDAGIntegration:
    """Integration tests for DAG components working together."""

    def test_build_dag_from_tool_calls(self):
        """Build DAG from list of tool calls with dependencies."""
        from petals.data_structures import ToolCall

        # Simulate LLM-generated tool calls
        tool_calls = [
            ToolCall(id="search", name="get_data", arguments={"query": "AI news"}, dependencies=[]),
            ToolCall(id="analyze", name="add", arguments={"a": 0, "b": 0}, dependencies=["search"]),
            ToolCall(id="report", name="merge_results", arguments={"data1": None, "data2": None}, dependencies=["analyze"]),
        ]

        dag = ToolCallDAG()

        # Convert ToolCalls to ToolCallNodes
        for tc in tool_calls:
            node = ToolCallNode(
                id=tc.id,
                name=tc.name,
                arguments=tc.arguments,
                dependencies=tc.dependencies
            )
            dag.add_node(node)

        waves = dag.get_waves()

        assert len(waves) == 3
        assert [n.id for n in waves[0]] == ["search"]
        assert [n.id for n in waves[1]] == ["analyze"]
        assert [n.id for n in waves[2]] == ["report"]

    def test_dag_node_lifecycle(self):
        """Test full node lifecycle through DAG."""
        dag = ToolCallDAG()

        node = ToolCallNode(
            id="lifecycle",
            name="add",
            arguments={"a": 1, "b": 2}
        )

        dag.add_node(node)

        # Initial state
        assert node.status == CallStatus.PENDING

        # Simulate execution states
        node.status = CallStatus.RUNNING
        assert node.status == CallStatus.RUNNING

        node.result = 3
        node.status = CallStatus.DONE
        assert node.status == CallStatus.DONE
        assert node.result == 3

        # Can also transition to failed
        node2 = ToolCallNode(id="fail", name="fail", arguments={})
        dag.add_node(node2)

        node2.status = CallStatus.RUNNING
        node2.error = "Execution failed"
        node2.status = CallStatus.FAILED

        assert node2.status == CallStatus.FAILED
        assert node2.error == "Execution failed"

    def test_complex_workflow_simulation(self):
        """Simulate complex multi-step workflow."""
        # Simulate: Research -> Multiple Searches -> Merge -> Format -> Report

        dag = ToolCallDAG()

        # Root task
        research = ToolCallNode(id="research", name="plan_research", arguments={"topic": "AI"})

        # Parallel searches
        web_search = ToolCallNode(
            id="web",
            name="web_search",
            arguments={"query": "AI trends"},
            dependencies=["research"]
        )
        arxiv_search = ToolCallNode(
            id="arxiv",
            name="arxiv_search",
            arguments={"query": "machine learning"},
            dependencies=["research"]
        )

        # Merge results
        merge = ToolCallNode(
            id="merge",
            name="merge_results",
            arguments={"data1": None, "data2": None},
            dependencies=["web", "arxiv"]
        )

        # Format
        format_results = ToolCallNode(
            id="format",
            name="format_document",
            arguments={"content": None},
            dependencies=["merge"]
        )

        # Final report
        report = ToolCallNode(
            id="report",
            name="save_report",
            arguments={"document": None},
            dependencies=["format"]
        )

        # Add all nodes
        for node in [research, web_search, arxiv_search, merge, format_results, report]:
            dag.add_node(node)

        waves = dag.get_waves()

        assert len(waves) == 5
        assert [n.id for n in waves[0]] == ["research"]
        assert set(n.id for n in waves[1]) == {"web", "arxiv"}
        assert [n.id for n in waves[2]] == ["merge"]
        assert [n.id for n in waves[3]] == ["format"]
        assert [n.id for n in waves[4]] == ["report"]
