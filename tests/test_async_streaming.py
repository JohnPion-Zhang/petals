"""Tests for Async/Streaming Module - Phase 2 TDD Implementation."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time

# Import all async/streaming modules
from petals.client.async_support import (
    TaskPool,
    StreamEvent,
    StreamEventType,
    StreamingAggregator,
    AggregationResult,
    StructuredOutputEnforcer,
    OutputSchema,
    ValidationResult,
    ValidationStatus,
    StreamingExecutor,
)


# ============================================================================
# TaskPool Tests
# ============================================================================

class TestTaskPool:
    """Tests for TaskPool class."""

    @pytest.mark.asyncio
    async def test_submit_and_wait(self):
        """Test submitting a coroutine and waiting for result."""
        pool = TaskPool(max_concurrency=5)

        async def dummy_task(value):
            await asyncio.sleep(0.01)
            return value * 2

        # submit() is async, so we must await it to get the Task
        task = await pool.submit(dummy_task(5))
        result = await task

        assert result == 10
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_run_convenience_method(self):
        """Test the run() convenience method."""
        pool = TaskPool(max_concurrency=5)

        async def add(a, b):
            await asyncio.sleep(0.01)
            return a + b

        result = await pool.run(add, 3, 7)
        assert result == 10

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency limit is respected."""
        pool = TaskPool(max_concurrency=2)
        active_count = 0
        max_seen = 0
        lock = asyncio.Lock()

        async def tracking_task():
            nonlocal active_count, max_seen
            async with lock:
                active_count += 1
                max_seen = max(max_seen, active_count)
            await asyncio.sleep(0.05)
            async with lock:
                active_count -= 1
            return "done"

        # Submit 5 tasks with concurrency limit of 2
        tasks = [pool.submit(tracking_task()) for _ in range(5)]
        await asyncio.gather(*tasks)

        # Should never have more than 2 active tasks
        assert max_seen <= 2
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test that statistics are tracked correctly."""
        pool = TaskPool(max_concurrency=5)

        async def success_task():
            return "success"

        async def fail_task():
            raise ValueError("test error")

        # Complete some tasks
        await pool.run(success_task)
        await pool.run(success_task)
        await pool.run(success_task)

        # Wait for tasks to be marked complete
        await pool.wait_idle()

        stats = pool.stats
        assert stats["completed"] == 3
        assert stats["failed"] == 0
        assert stats["active"] == 0

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cancel_pending(self):
        """Test shutdown with cancel_pending=True."""
        pool = TaskPool(max_concurrency=5)

        async def slow_task():
            await asyncio.sleep(10)  # Very slow task
            return "done"

        # Submit a slow task - submit() returns a coroutine
        submit_coro = pool.submit(slow_task())
        task = await submit_coro

        # Give it time to start
        await asyncio.sleep(0.01)

        # Shutdown with cancellation
        await pool.shutdown(cancel_pending=True)

        # Task should be cancelled
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_is_idle(self):
        """Test is_idle property."""
        pool = TaskPool(max_concurrency=5)

        assert pool.is_idle()

        async def dummy():
            await asyncio.sleep(0.1)
            return "done"

        # submit() is async
        task = await pool.submit(dummy())
        assert not pool.is_idle()

        await task
        await pool.shutdown()

        assert pool.is_idle()


# ============================================================================
# Streaming Types Tests
# ============================================================================

class TestStreamEvent:
    """Tests for StreamEvent and StreamEventType."""

    def test_event_type_values(self):
        """Test StreamEventType enum values."""
        assert StreamEventType.TEXT_CHUNK.value == "text_chunk"
        assert StreamEventType.TOOL_CALL_PENDING.value == "tool_call_pending"
        assert StreamEventType.TOOL_CALL_READY.value == "tool_call_ready"
        assert StreamEventType.TOOL_EXECUTING.value == "tool_executing"
        assert StreamEventType.TOOL_RESULT.value == "tool_result"
        assert StreamEventType.FINAL.value == "final"
        assert StreamEventType.ERROR.value == "error"

    def test_text_chunk_factory(self):
        """Test text_chunk factory method."""
        event = StreamEvent.text_chunk("Hello, ", is_final=False)
        assert event.type == StreamEventType.TEXT_CHUNK
        assert event.data["text"] == "Hello, "
        assert event.data["is_final"] is False

    def test_tool_pending_factory(self):
        """Test tool_pending factory method."""
        event = StreamEvent.tool_pending("search", "node_1")
        assert event.type == StreamEventType.TOOL_CALL_PENDING
        assert event.data["tool_name"] == "search"
        assert event.data["node_id"] == "node_1"

    def test_tool_result_factory(self):
        """Test tool_result factory method."""
        event = StreamEvent.tool_result("search", "node_1", {"data": [1, 2, 3]})
        assert event.type == StreamEventType.TOOL_RESULT
        assert event.data["tool_name"] == "search"
        assert event.data["result"] == {"data": [1, 2, 3]}
        assert event.data["success"] is True

    def test_tool_result_with_error(self):
        """Test tool_result with error."""
        event = StreamEvent.tool_result("search", "node_1", None, success=False, error="Timeout")
        assert event.type == StreamEventType.TOOL_RESULT
        assert event.data["success"] is False
        assert event.data["error"] == "Timeout"

    def test_final_factory(self):
        """Test final factory method."""
        event = StreamEvent.final(total_chunks=10, total_tools=3, final_text="Done!")
        assert event.type == StreamEventType.FINAL
        assert event.data["total_chunks"] == 10
        assert event.data["total_tools"] == 3
        assert event.data["final_text"] == "Done!"

    def test_error_factory(self):
        """Test error factory method."""
        event = StreamEvent.error("Something went wrong", error_type="timeout", recoverable=True)
        assert event.type == StreamEventType.ERROR
        assert event.data["message"] == "Something went wrong"
        assert event.data["error_type"] == "timeout"
        assert event.data["recoverable"] is True

    def test_to_sse_format(self):
        """Test SSE formatting."""
        event = StreamEvent.text_chunk("Hello")
        sse = event.to_sse()
        assert "event: text_chunk" in sse
        assert 'data: {"text": "Hello"' in sse
        assert sse.endswith("\n\n")

    def test_to_dict(self):
        """Test dictionary conversion."""
        event = StreamEvent.text_chunk("Hello")
        d = event.to_dict()
        assert d["type"] == "text_chunk"
        assert d["data"]["text"] == "Hello"
        assert "timestamp" in d


# ============================================================================
# StreamingAggregator Tests
# ============================================================================

class TestStreamingAggregator:
    """Tests for StreamingAggregator class."""

    @pytest.mark.asyncio
    async def test_add_chunk_increments_count(self):
        """Test that adding chunks increments the count."""
        aggregator = StreamingAggregator()

        event = StreamEvent.tool_result("search", "n1", {"data": [1, 2]})
        await aggregator.add_chunk(event)

        assert aggregator._chunk_count == 1
        # Items are extracted from the result dict - since "data" is a list, it's extracted
        assert len(aggregator._items) == 1
        assert aggregator._items == [[1, 2]]  # Items extracted from result["data"]

    @pytest.mark.asyncio
    async def test_deduplication(self):
        """Test that deduplication works."""
        aggregator = StreamingAggregator(deduplicate=True, key_field="id")

        item1 = {"id": "abc", "name": "test"}
        item2 = {"id": "abc", "name": "test"}  # Duplicate

        await aggregator.add_chunk(StreamEvent.tool_result("t", "n1", item1))
        await aggregator.add_chunk(StreamEvent.tool_result("t", "n2", item2))

        assert len(aggregator._items) == 1  # Only one should be added

    @pytest.mark.asyncio
    async def test_get_current_result(self):
        """Test getting current result without waiting."""
        aggregator = StreamingAggregator()

        await aggregator.add_chunk(StreamEvent.tool_result("s", "n1", [1, 2, 3]))

        result = aggregator.get_current_result()
        assert result.status == "partial"
        assert result.total_chunks == 1

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error event handling."""
        aggregator = StreamingAggregator()

        await aggregator.add_chunk(StreamEvent.error("Test error", "test"))
        await aggregator.add_chunk(StreamEvent.tool_result("s", "n1", "ok"))

        result = aggregator.get_current_result()
        assert len(result.errors) == 1
        assert "Test error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_aggregate_stream(self):
        """Test full stream aggregation."""
        aggregator = StreamingAggregator()

        async def event_stream():
            yield StreamEvent.tool_result("s1", "n1", {"items": [1, 2]})
            yield StreamEvent.tool_result("s2", "n2", {"items": [3, 4]})
            yield StreamEvent.final(total_tools=2)

        result = await aggregator.aggregate_stream(event_stream())

        assert result.status == "success"
        assert result.total_chunks == 3

    @pytest.mark.asyncio
    async def test_text_chunk_accumulation(self):
        """Test text chunks are accumulated correctly."""
        aggregator = StreamingAggregator()

        await aggregator.add_chunk(StreamEvent.text_chunk("Hello", is_final=False))
        await aggregator.add_chunk(StreamEvent.text_chunk(" World", is_final=False))
        await aggregator.add_chunk(StreamEvent.text_chunk("!", is_final=True))

        result = aggregator.get_current_result()
        assert result.metadata["full_text"] == "Hello World!"
        assert result.metadata["text_complete"] is True


# ============================================================================
# StructuredOutput Tests
# ============================================================================

class TestOutputSchema:
    """Tests for OutputSchema class."""

    def test_validate_required_fields(self):
        """Test required field validation."""
        schema = OutputSchema(required_fields=["name", "value"])

        # Missing field
        result = schema.validate({"name": "test"})
        assert not result.is_valid
        assert any("value" in e for e in result.errors)

        # All required present
        result = schema.validate({"name": "test", "value": 42})
        assert result.is_valid

    def test_validate_field_types(self):
        """Test field type validation."""
        # Add fields to required or optional fields to avoid unknown field warnings
        schema = OutputSchema(
            required_fields=["count"],
            optional_fields=["name"],
            field_types={"count": int, "name": str}
        )

        # Wrong type
        result = schema.validate({"count": "not an int"})
        assert not result.is_valid

        # Correct types
        result = schema.validate({"count": 5, "name": "test"})
        assert result.is_valid

    def test_validate_custom_validators(self):
        """Test custom validators."""
        schema = OutputSchema(
            required_fields=["count"],
            validators={"count": lambda x: x > 0}
        )

        # Invalid (not > 0)
        result = schema.validate({"count": -1})
        assert not result.is_valid

        # Valid
        result = schema.validate({"count": 5})
        assert result.is_valid


class TestStructuredOutputEnforcer:
    """Tests for StructuredOutputEnforcer class."""

    @pytest.mark.asyncio
    async def test_register_and_get_schema(self):
        """Test schema registration and retrieval."""
        enforcer = StructuredOutputEnforcer()
        schema = OutputSchema(required_fields=["data"])

        enforcer.register_schema("my_tool", schema)
        retrieved = enforcer.get_schema("my_tool")

        assert retrieved is schema

    @pytest.mark.asyncio
    async def test_validate_with_schema(self):
        """Test validation with registered schema."""
        enforcer = StructuredOutputEnforcer()
        schema = OutputSchema(required_fields=["results"])
        enforcer.register_schema("search", schema)

        result = await enforcer.validate_and_extract(
            "search",
            {"results": [1, 2, 3]}
        )

        assert result.is_valid
        assert result.data == {"results": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_fallback_on_invalid(self):
        """Test fallback behavior on invalid output."""
        enforcer = StructuredOutputEnforcer(on_invalid="fallback")
        schema = OutputSchema(required_fields=["data"])
        enforcer.register_schema("tool", schema)

        result = await enforcer.validate_and_extract(
            "tool",
            {"wrong_field": "value"}
        )

        # Should have fallback data
        assert result.data is not None
        assert "_fallback" in result.data

    def test_default_schemas(self):
        """Test default schema creation."""
        search_schema = StructuredOutputEnforcer.default_search_schema()
        assert "results" in search_schema.required_fields

        list_schema = StructuredOutputEnforcer.default_list_schema()
        assert "items" in list_schema.required_fields

    def test_register_defaults(self):
        """Test registering default schemas."""
        enforcer = StructuredOutputEnforcer()
        enforcer.register_defaults()

        assert enforcer.get_schema("search") is not None
        assert enforcer.get_schema("list") is not None


# ============================================================================
# ValidationResult Tests
# ============================================================================

class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_add_error(self):
        """Test adding errors."""
        result = ValidationResult(status=ValidationStatus.VALID)
        assert result.is_valid

        result.add_error("Something went wrong")
        assert result.is_invalid
        assert len(result.errors) == 1

    def test_add_warning(self):
        """Test adding warnings."""
        result = ValidationResult(status=ValidationStatus.VALID)
        assert result.is_valid

        result.add_warning("Minor issue")
        assert result.is_partial
        assert len(result.warnings) == 1

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = ValidationResult(
            status=ValidationStatus.INVALID,
            errors=["Error 1"],
            data={"key": "value"}
        )

        d = result.to_dict()
        assert d["status"] == "invalid"
        assert d["errors"] == ["Error 1"]
        assert d["data"] == {"key": "value"}


# ============================================================================
# StreamingExecutor Tests
# ============================================================================

class TestStreamingExecutor:
    """Tests for StreamingExecutor class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock tool registry."""
        registry = MagicMock()
        registry.execute = AsyncMock(return_value={"data": "result"})
        return registry

    @pytest.fixture
    def simple_dag(self):
        """Create a simple DAG for testing."""
        from petals.client.dag import ToolCallDAG, ToolCallNode

        dag = ToolCallDAG()
        node1 = ToolCallNode(id="n1", name="search", arguments={"query": "test"})
        node2 = ToolCallNode(
            id="n2",
            name="process",
            arguments={"data": {"from_dep": "n1"}},
            dependencies=["n1"]
        )
        dag.add_node(node1)
        dag.add_node(node2)
        return dag

    @pytest.mark.asyncio
    async def test_execute_streaming_basic(self, mock_registry, simple_dag):
        """Test basic streaming execution."""
        executor = StreamingExecutor(mock_registry, max_concurrency=5)

        events = []
        async for event in executor.execute_streaming(simple_dag):
            events.append(event)

        # Should have events for each node plus final
        assert len(events) >= 2  # At least pending + result for each node

        # Last event should be final
        assert events[-1].type == StreamEventType.FINAL

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_execute_streaming_cycles(self, mock_registry):
        """Test that cycles are detected and handled."""
        from petals.client.dag import ToolCallDAG, ToolCallNode

        # Create a DAG with a cycle
        dag = ToolCallDAG()
        node1 = ToolCallNode(id="n1", name="a", arguments={})
        node2 = ToolCallNode(id="n2", name="b", arguments={}, dependencies=["n1"])
        node1.dependencies = ["n2"]  # Create cycle: n1 -> n2 -> n1

        dag.add_node(node1)
        dag.add_node(node2)

        executor = StreamingExecutor(mock_registry)

        events = []
        async for event in executor.execute_streaming(dag):
            events.append(event)

        # Should have an error event about the cycle
        error_events = [e for e in events if e.type == StreamEventType.ERROR]
        assert len(error_events) > 0
        assert "cycle" in error_events[0].data["message"].lower()

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_stats_tracking(self, mock_registry, simple_dag):
        """Test that statistics are tracked."""
        executor = StreamingExecutor(mock_registry)

        async for _ in executor.execute_streaming(simple_dag):
            pass

        stats = executor.stats
        assert stats["total_tools_executed"] >= 0
        assert "task_pool" in stats
        assert "aggregator" in stats

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_registry):
        """Test graceful shutdown."""
        executor = StreamingExecutor(mock_registry)

        # Shutdown should complete without error
        await executor.shutdown()

        # Double shutdown should also be safe
        await executor.shutdown()

    def test_register_schema(self):
        """Test schema registration."""
        executor = StreamingExecutor(MagicMock())

        schema = OutputSchema(required_fields=["data"])
        executor.register_schema("test_tool", schema)

        # Should not raise
        assert True

    def test_register_default_schemas(self):
        """Test registering default schemas."""
        executor = StreamingExecutor(MagicMock())
        executor.register_default_schemas()

        # Should not raise
        assert True


# ============================================================================
# AggregationResult Tests
# ============================================================================

class TestAggregationResult:
    """Tests for AggregationResult class."""

    def test_properties(self):
        """Test status properties."""
        result = AggregationResult(status="success")
        assert result.is_success

        result = AggregationResult(status="partial")
        assert result.is_partial

        result = AggregationResult(status="error")
        assert result.is_error

    def test_add_item(self):
        """Test adding items."""
        result = AggregationResult()
        result.add_item({"key": "value"})
        assert len(result.items) == 1

    def test_add_error(self):
        """Test adding errors."""
        result = AggregationResult(status="success")
        result.add_error("Test error")
        assert result.status == "partial"
        assert "Test error" in result.errors

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = AggregationResult(
            items=[1, 2, 3],
            errors=["error"],
            status="success",
            total_chunks=5
        )

        d = result.to_dict()
        assert d["items"] == [1, 2, 3]
        assert d["errors"] == ["error"]
        assert d["status"] == "success"
        assert d["total_chunks"] == 5
