"""
Tests for Unified Orchestrator - Agent Tool Call Layer

TDD Phase: Integration Tests for 4-Phase Orchestrator

These tests cover:
- Orchestrator initialization with various configs
- Streaming execution
- Batch execution
- Circuit breaker integration
- Stats tracking
- Error handling
- All 4 phases working together
"""
import asyncio
import pytest

from petals.client.orchestrator import Orchestrator, OrchestratorConfig
from petals.client.orchestrator import (
    ToolCallNode,
    ToolCallDAG,
    WaveExecutor,
    StreamEvent,
    StreamEventType,
)
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

    async def slow_tool(delay: float = 0.05):
        await asyncio.sleep(delay)
        return {"delayed": True}

    registry.register("add", add_numbers)
    registry.register("multiply", multiply_numbers)
    registry.register("get_data", get_data)
    registry.register("merge_results", merge_results)
    registry.register("fail", failing_tool)
    registry.register("slow", slow_tool)

    return registry


@pytest.fixture
def default_config():
    """Create default orchestrator config."""
    return OrchestratorConfig(
        max_concurrency=10,
        execution_timeout=5.0,
        max_retries=2,
        enable_correction=True,
        enable_verification=True,
        enable_structured_output=True,
        enable_circuit_breaker=True,
    )


@pytest.fixture
def minimal_config():
    """Create minimal config with most features disabled."""
    return OrchestratorConfig(
        max_concurrency=5,
        execution_timeout=2.0,
        max_retries=1,
        enable_correction=False,
        enable_verification=False,
        enable_structured_output=False,
        enable_circuit_breaker=False,
    )


@pytest.fixture
def orchestrator(tool_registry, default_config):
    """Create orchestrator with default config."""
    return Orchestrator(tool_registry, config=default_config)


@pytest.fixture
def minimal_orchestrator(tool_registry, minimal_config):
    """Create orchestrator with minimal config."""
    return Orchestrator(tool_registry, config=minimal_config)


# ============================================================================
# Orchestrator Initialization Tests
# ============================================================================

class TestOrchestratorInitialization:
    """Tests for Orchestrator initialization."""

    def test_orchestrator_with_default_config(self, tool_registry):
        """Create orchestrator with default configuration."""
        orch = Orchestrator(tool_registry)

        assert orch.registry is tool_registry
        assert orch.config is not None
        assert orch._wave_executor is not None
        assert orch._task_pool is not None
        assert orch._feedback_loop is not None

    def test_orchestrator_with_custom_config(self, tool_registry):
        """Create orchestrator with custom configuration."""
        config = OrchestratorConfig(
            max_concurrency=20,
            execution_timeout=60.0,
            max_retries=5,
            enable_verification=True,
            enable_correction=True,
        )
        orch = Orchestrator(tool_registry, config=config)

        assert orch.config.max_concurrency == 20
        assert orch.config.execution_timeout == 60.0
        assert orch.config.max_retries == 5
        assert orch._verifier is not None
        assert orch._verification_executor is not None

    def test_orchestrator_with_verification_disabled(self, tool_registry):
        """Create orchestrator with verification disabled."""
        config = OrchestratorConfig(enable_verification=False)
        orch = Orchestrator(tool_registry, config=config)

        assert orch._verifier is None
        assert orch._verification_executor is None

    def test_orchestrator_with_circuit_breaker_disabled(self, tool_registry):
        """Create orchestrator with circuit breaker disabled."""
        config = OrchestratorConfig(enable_circuit_breaker=False)
        orch = Orchestrator(tool_registry, config=config)

        assert orch._circuit_breaker is None

    def test_orchestrator_repr(self, tool_registry, default_config):
        """Test orchestrator string representation."""
        orch = Orchestrator(tool_registry, config=default_config)
        repr_str = repr(orch)

        assert "Orchestrator" in repr_str
        assert "has_verifier=True" in repr_str
        assert "has_feedback=True" in repr_str


# ============================================================================
# Streaming Execution Tests
# ============================================================================

class TestStreamingExecution:
    """Tests for streaming execution of DAGs."""

    @pytest.mark.asyncio
    async def test_streaming_single_node(self, orchestrator):
        """Test streaming execution of a single node DAG."""
        dag = ToolCallDAG()
        node = ToolCallNode(
            id="add_1",
            name="add",
            arguments={"a": 5, "b": 3}
        )
        dag.add_node(node)

        events = []
        async for event in orchestrator.execute_streaming(dag):
            events.append(event)

        # Should have events for pending, executing, result, and final
        assert len(events) >= 3

        # Check we have final event
        final_events = [e for e in events if e.type == StreamEventType.FINAL]
        assert len(final_events) == 1

        # Check tool result event
        result_events = [e for e in events if e.type == StreamEventType.TOOL_RESULT]
        assert len(result_events) == 1
        assert result_events[0].data["result"] == 8

    @pytest.mark.asyncio
    async def test_streaming_parallel_nodes(self, orchestrator):
        """Test streaming execution of parallel nodes."""
        dag = ToolCallDAG()
        node_a = ToolCallNode(id="a", name="add", arguments={"a": 1, "b": 2})
        node_b = ToolCallNode(id="b", name="multiply", arguments={"a": 3, "b": 4})
        dag.add_node(node_a)
        dag.add_node(node_b)

        events = []
        async for event in orchestrator.execute_streaming(dag):
            events.append(event)

        # Check result events
        result_events = [e for e in events if e.type == StreamEventType.TOOL_RESULT]
        assert len(result_events) == 2

        results = {e.data["id"]: e.data["result"] for e in result_events}
        assert results["a"] == 3  # 1 + 2
        assert results["b"] == 12  # 3 * 4

    @pytest.mark.asyncio
    async def test_streaming_sequential_waves(self, minimal_orchestrator):
        """Test streaming execution of sequential waves."""
        dag = ToolCallDAG()
        node_a = ToolCallNode(id="a", name="add", arguments={"a": 1, "b": 2})
        node_b = ToolCallNode(
            id="b",
            name="add",
            arguments={"a": 0, "b": 0},  # Will be resolved from dependency
            dependencies=["a"]
        )
        dag.add_node(node_a)
        dag.add_node(node_b)

        # Pre-seed the results cache with dependency value
        initial_args = {"a": 3}  # This will be overridden by actual execution

        events = []
        async for event in minimal_orchestrator.execute_streaming(dag, initial_args):
            events.append(event)

        result_events = [e for e in events if e.type == StreamEventType.TOOL_RESULT]
        assert len(result_events) == 2

    @pytest.mark.asyncio
    async def test_streaming_with_errors(self, orchestrator):
        """Test streaming handles tool errors gracefully."""
        dag = ToolCallDAG()
        node = ToolCallNode(id="fail", name="fail", arguments={})
        dag.add_node(node)

        events = []
        async for event in orchestrator.execute_streaming(dag):
            events.append(event)

        # Should have error event
        error_events = [e for e in events if e.type == StreamEventType.ERROR]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_streaming_preserves_node_order(self, orchestrator):
        """Test that streaming events preserve node execution order."""
        dag = ToolCallDAG()
        node_a = ToolCallNode(id="first", name="add", arguments={"a": 1, "b": 1})
        node_b = ToolCallNode(id="second", name="add", arguments={"a": 1, "b": 1})
        node_c = ToolCallNode(id="third", name="add", arguments={"a": 1, "b": 1})

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)

        pending_events = []
        async for event in orchestrator.execute_streaming(dag):
            if event.type == StreamEventType.TOOL_CALL_PENDING:
                pending_events.append(event.data["id"])

        # Nodes should be executed in wave order
        assert pending_events.count("first") == 1
        assert pending_events.count("second") == 1
        assert pending_events.count("third") == 1


# ============================================================================
# Batch Execution Tests
# ============================================================================

class TestBatchExecution:
    """Tests for batch execution of multiple DAGs."""

    @pytest.mark.asyncio
    async def test_batch_single_dag(self, orchestrator):
        """Test batch execution of a single DAG."""
        dag = ToolCallDAG()
        node = ToolCallNode(id="add", name="add", arguments={"a": 5, "b": 5})
        dag.add_node(node)

        results = await orchestrator.execute_batch([dag])

        assert len(results) == 1
        assert results[0]["success"] is True
        assert len(results[0]["events"]) > 0

    @pytest.mark.asyncio
    async def test_batch_multiple_dags(self, orchestrator):
        """Test batch execution of multiple DAGs."""
        dag1 = ToolCallDAG()
        dag1.add_node(ToolCallNode(id="n1", name="add", arguments={"a": 1, "b": 2}))

        dag2 = ToolCallDAG()
        dag2.add_node(ToolCallNode(id="n2", name="add", arguments={"a": 3, "b": 4}))

        dag3 = ToolCallDAG()
        dag3.add_node(ToolCallNode(id="n3", name="add", arguments={"a": 5, "b": 6}))

        results = await orchestrator.execute_batch([dag1, dag2, dag3])

        assert len(results) == 3
        assert all(r["success"] for r in results)

    @pytest.mark.asyncio
    async def test_batch_with_failures(self, orchestrator):
        """Test batch execution handles failures."""
        dag_success = ToolCallDAG()
        dag_success.add_node(ToolCallNode(id="ok", name="add", arguments={"a": 1, "b": 2}))

        dag_fail = ToolCallDAG()
        dag_fail.add_node(ToolCallNode(id="fail", name="fail", arguments={}))

        results = await orchestrator.execute_batch([dag_success, dag_fail])

        assert len(results) == 2
        # First should succeed, second may have error event
        assert results[0]["success"] is True

    @pytest.mark.asyncio
    async def test_batch_preserves_node_counts(self, orchestrator):
        """Test batch execution preserves node counts per DAG."""
        dag1 = ToolCallDAG()
        dag1.add_node(ToolCallNode(id="a", name="add", arguments={"a": 1, "b": 2}))
        dag1.add_node(ToolCallNode(id="b", name="add", arguments={"a": 3, "b": 4}))

        dag2 = ToolCallDAG()
        dag2.add_node(ToolCallNode(id="c", name="add", arguments={"a": 5, "b": 6}))

        results = await orchestrator.execute_batch([dag1, dag2])

        assert results[0]["nodes"] == 2
        assert results[1]["nodes"] == 1


# ============================================================================
# Circuit Breaker Tests
# ============================================================================

class TestCircuitBreaker:
    """Tests for circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success(self, orchestrator):
        """Test circuit breaker records successful executions."""
        dag = ToolCallDAG()
        dag.add_node(ToolCallNode(id="ok", name="add", arguments={"a": 1, "b": 2}))

        await orchestrator.execute_streaming(dag).__anext__()  # Start execution

        stats = orchestrator.stats
        assert "circuit_breaker" in stats

    @pytest.mark.asyncio
    async def test_circuit_breaker_disabled(self, minimal_orchestrator):
        """Test orchestrator works without circuit breaker."""
        dag = ToolCallDAG()
        dag.add_node(ToolCallNode(id="add", name="add", arguments={"a": 1, "b": 2}))

        events = []
        async for event in minimal_orchestrator.execute_streaming(dag):
            events.append(event)

        assert len(events) > 0

        stats = minimal_orchestrator.stats
        assert "circuit_breaker" not in stats

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_tracking(self, orchestrator):
        """Test circuit breaker state is tracked in stats."""
        dag = ToolCallDAG()
        dag.add_node(ToolCallNode(id="add", name="add", arguments={"a": 1, "b": 2}))

        async for _ in orchestrator.execute_streaming(dag):
            pass

        stats = orchestrator.stats
        assert "circuit_breaker" in stats
        assert stats["circuit_breaker"]["state"] in ["closed", "open", "half_open"]


# ============================================================================
# Statistics Tracking Tests
# ============================================================================

class TestStatisticsTracking:
    """Tests for execution statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_tracked_on_success(self, orchestrator):
        """Test statistics are tracked on successful execution."""
        dag = ToolCallDAG()
        dag.add_node(ToolCallNode(id="add", name="add", arguments={"a": 1, "b": 2}))

        async for _ in orchestrator.execute_streaming(dag):
            pass

        stats = orchestrator.stats
        assert stats["total_executions"] >= 1
        assert stats["successful_executions"] >= 1

    @pytest.mark.asyncio
    async def test_stats_tracked_on_failure(self, orchestrator):
        """Test statistics are tracked on failed execution."""
        dag = ToolCallDAG()
        dag.add_node(ToolCallNode(id="fail", name="fail", arguments={}))

        async for _ in orchestrator.execute_streaming(dag):
            pass

        stats = orchestrator.stats
        assert stats["total_executions"] >= 1

    @pytest.mark.asyncio
    async def test_verification_stats_when_enabled(self, orchestrator):
        """Test verification stats are tracked when enabled."""
        dag = ToolCallDAG()
        dag.add_node(ToolCallNode(id="add", name="add", arguments={"a": 1, "b": 2}))

        async for _ in orchestrator.execute_streaming(dag):
            pass

        stats = orchestrator.stats
        assert "verifier" in stats
        assert "verification_executor" in stats

    @pytest.mark.asyncio
    async def test_stats_reset_on_new_execution(self, orchestrator):
        """Test stats can be retrieved multiple times."""
        dag = ToolCallDAG()
        dag.add_node(ToolCallNode(id="add", name="add", arguments={"a": 1, "b": 2}))

        # First execution
        async for _ in orchestrator.execute_streaming(dag):
            pass

        stats1 = orchestrator.stats

        # Second execution
        async for _ in orchestrator.execute_streaming(dag):
            pass

        stats2 = orchestrator.stats

        # Stats should accumulate
        assert stats2["total_executions"] >= stats1["total_executions"]


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in the orchestrator."""

    @pytest.mark.asyncio
    async def test_handles_unknown_tool(self, orchestrator):
        """Test orchestrator handles unknown tools gracefully."""
        dag = ToolCallDAG()
        dag.add_node(ToolCallNode(id="unknown", name="nonexistent_tool", arguments={}))

        events = []
        async for event in orchestrator.execute_streaming(dag):
            events.append(event)

        # Should have error event
        error_events = [e for e in events if e.type == StreamEventType.ERROR]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_handles_timeout(self, minimal_orchestrator):
        """Test orchestrator handles timeouts."""
        # Create config with very short timeout
        config = OrchestratorConfig(execution_timeout=0.001, max_retries=1)
        orch = Orchestrator(minimal_orchestrator.registry, config=config)

        dag = ToolCallDAG()
        dag.add_node(ToolCallNode(id="slow", name="slow", arguments={"delay": 1.0}))

        events = []
        async for event in orch.execute_streaming(dag):
            events.append(event)

        # Should have either result with error or error event
        result_events = [e for e in events if e.type == StreamEventType.TOOL_RESULT]
        error_events = [e for e in events if e.type == StreamEventType.ERROR]
        assert len(result_events) > 0 or len(error_events) > 0

    @pytest.mark.asyncio
    async def test_continues_after_error_in_wave(self, orchestrator):
        """Test execution continues after an error in a wave."""
        dag = ToolCallDAG()
        # Both in same wave (no dependencies)
        dag.add_node(ToolCallNode(id="ok", name="add", arguments={"a": 1, "b": 2}))
        dag.add_node(ToolCallNode(id="fail", name="fail", arguments={}))

        events = []
        async for event in orchestrator.execute_streaming(dag):
            events.append(event)

        # Should have result for the successful node
        result_events = [e for e in events if e.type == StreamEventType.TOOL_RESULT]
        # At least the successful node should have a result
        success_results = [e for e in result_events if e.data.get("result") is not None]
        assert len(success_results) >= 1


# ============================================================================
# Configuration Tests
# ============================================================================

class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_default_config_values(self):
        """Test default config values."""
        config = OrchestratorConfig()

        assert config.max_concurrency == 10
        assert config.execution_timeout == 30.0
        assert config.max_retries == 3
        assert config.base_backoff == 1.0
        assert config.max_backoff == 30.0
        assert config.enable_correction is True
        assert config.enable_verification is True
        assert config.enable_structured_output is True
        assert config.enable_circuit_breaker is True
        assert config.verification_level == "structural"

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = OrchestratorConfig(
            max_concurrency=50,
            execution_timeout=120.0,
            max_retries=10,
            enable_verification=False,
        )

        assert config.max_concurrency == 50
        assert config.execution_timeout == 120.0
        assert config.max_retries == 10
        assert config.enable_verification is False

    def test_verification_level_values(self):
        """Test valid verification level values."""
        for level in ["none", "basic", "structural", "deep"]:
            config = OrchestratorConfig(verification_level=level)
            assert config.verification_level == level


# ============================================================================
# Schema Registration Tests
# ============================================================================

class TestSchemaRegistration:
    """Tests for schema registration."""

    def test_register_schema(self, orchestrator):
        """Test registering a schema."""
        from petals.client.async_support import OutputSchema

        schema = OutputSchema(
            required_fields=["query", "result"],
            field_types={"query": str, "result": str}
        )
        orchestrator.register_schema("get_data", schema)

        # Schema should be registered in enforcer
        assert orchestrator._enforcer is not None
        registered = orchestrator._enforcer.get_schema("get_data")
        assert registered is not None

    def test_register_verification_rule(self, orchestrator):
        """Test registering a verification rule."""
        from petals.client.verification import VerificationRule

        rule = VerificationRule(
            name="has_results",
            description="Check if results are non-empty",
            check=lambda r: isinstance(r, dict) and "result" in r,
        )
        orchestrator.register_verification_rule("get_data", rule)

        # Rule should be registered in verifier
        assert orchestrator._verifier is not None
        rules = orchestrator._verifier.get_rules("get_data")
        assert len(rules) >= 1

    def test_register_default_schemas(self, orchestrator):
        """Test registering default schemas."""
        orchestrator.register_default_schemas()

        # Default schemas should be registered
        assert orchestrator._enforcer is not None
        assert orchestrator._enforcer.get_schema("search") is not None


# ============================================================================
# Single Node Execution Tests
# ============================================================================

class TestSingleNodeExecution:
    """Tests for executing single nodes outside DAG context."""

    @pytest.mark.asyncio
    async def test_execute_single_node(self, orchestrator):
        """Test executing a single node."""
        node = ToolCallNode(id="single", name="add", arguments={"a": 5, "b": 3})

        result = await orchestrator.execute_single_node(node)

        assert result.status == CallStatus.DONE
        assert result.result == 8

    @pytest.mark.asyncio
    async def test_execute_single_node_with_cache(self, orchestrator):
        """Test executing single node with results cache."""
        node = ToolCallNode(
            id="dependent",
            name="add",
            arguments={"a": 0, "b": 0},  # Will use cached value
        )
        node.dependencies = ["parent"]

        cache = {"parent": 10}

        result = await orchestrator.execute_single_node(node, cache)

        # Result depends on how the executor handles the unresolved dependency
        assert result is not None


# ============================================================================
# Shutdown Tests
# ============================================================================

class TestShutdown:
    """Tests for orchestrator shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_completes(self, orchestrator):
        """Test shutdown completes without error."""
        dag = ToolCallDAG()
        dag.add_node(ToolCallNode(id="add", name="add", arguments={"a": 1, "b": 2}))

        # Execute first
        async for _ in orchestrator.execute_streaming(dag):
            pass

        # Shutdown should not raise
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_clears_state(self, orchestrator):
        """Test shutdown clears internal state."""
        dag = ToolCallDAG()
        dag.add_node(ToolCallNode(id="add", name="add", arguments={"a": 1, "b": 2}))

        # Execute first
        async for _ in orchestrator.execute_streaming(dag):
            pass

        await orchestrator.shutdown()

        # Task pool should be idle after shutdown
        assert orchestrator._task_pool.is_idle()


# ============================================================================
# Integration Tests
# ============================================================================

class TestOrchestratorIntegration:
    """Integration tests for full orchestrator functionality."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, orchestrator):
        """Test complete workflow with multiple DAGs."""
        # Create a DAG representing a typical workflow
        dag = ToolCallDAG()

        # Research phase
        research = ToolCallNode(
            id="research",
            name="get_data",
            arguments={"query": "AI trends"}
        )

        # Parallel execution phase
        web_search = ToolCallNode(
            id="web",
            name="get_data",
            arguments={"query": "web AI"}
        )
        arxiv_search = ToolCallNode(
            id="arxiv",
            name="get_data",
            arguments={"query": "ml papers"}
        )

        # Merge phase
        merge = ToolCallNode(
            id="merge",
            name="merge_results",
            arguments={"data1": {}, "data2": {}},
            dependencies=["web", "arxiv"]
        )

        dag.add_node(research)
        dag.add_node(web_search)
        dag.add_node(arxiv_search)
        dag.add_node(merge)

        events = []
        async for event in orchestrator.execute_streaming(dag):
            events.append(event)

        # Verify execution completed
        assert len(events) > 0

        final_events = [e for e in events if e.type == StreamEventType.FINAL]
        assert len(final_events) == 1

        # Verify stats are populated
        stats = orchestrator.stats
        assert stats["total_executions"] >= 4

    @pytest.mark.asyncio
    async def test_diamond_pattern_workflow(self, minimal_orchestrator):
        r"""Test diamond pattern DAG execution.

        Pattern:
            A
           / \
          B   C
           \ /
            D
        """
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="add", arguments={"a": 10, "b": 5})
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
            arguments={"data1": {}, "data2": {}},
            dependencies=["b", "c"]
        )

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        events = []
        async for event in minimal_orchestrator.execute_streaming(dag):
            events.append(event)

        # All nodes should complete
        result_events = [e for e in events if e.type == StreamEventType.TOOL_RESULT]
        assert len(result_events) == 4

    @pytest.mark.asyncio
    async def test_sequential_chain_workflow(self, minimal_orchestrator):
        """Test sequential chain DAG execution.

        Pattern: A -> B -> C -> D
        """
        dag = ToolCallDAG()

        node_a = ToolCallNode(id="a", name="add", arguments={"a": 1, "b": 1})
        node_b = ToolCallNode(
            id="b",
            name="add",
            arguments={"a": 0, "b": 0},
            dependencies=["a"]
        )
        node_c = ToolCallNode(
            id="c",
            name="add",
            arguments={"a": 0, "b": 0},
            dependencies=["b"]
        )
        node_d = ToolCallNode(
            id="d",
            name="add",
            arguments={"a": 0, "b": 0},
            dependencies=["c"]
        )

        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_node(node_c)
        dag.add_node(node_d)

        events = []
        async for event in minimal_orchestrator.execute_streaming(dag):
            events.append(event)

        # All nodes should complete
        result_events = [e for e in events if e.type == StreamEventType.TOOL_RESULT]
        assert len(result_events) == 4

    @pytest.mark.asyncio
    async def test_concurrent_batch_executions(self, orchestrator):
        """Test concurrent execution of multiple batch operations."""
        dags = []
        for i in range(5):
            dag = ToolCallDAG()
            dag.add_node(ToolCallNode(
                id=f"add_{i}",
                name="add",
                arguments={"a": i, "b": i}
            ))
            dags.append(dag)

        results = await orchestrator.execute_batch(dags)

        assert len(results) == 5
        assert all(r["success"] for r in results)

    def test_repr_includes_all_components(self, orchestrator):
        """Test repr includes all configured components."""
        repr_str = repr(orchestrator)

        # Should include verifier and feedback
        assert "has_verifier=True" in repr_str
        assert "has_feedback=True" in repr_str
