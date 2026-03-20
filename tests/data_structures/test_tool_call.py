"""Tests for ToolCall data class - Red Phase (should fail initially)."""

import pytest
from petals.data_structures import ToolCall, CallStatus


class TestToolCallCreation:
    """Test ToolCall creation with valid arguments."""

    def test_tool_call_creation_with_valid_args(self):
        """Test creating ToolCall with name and arguments."""
        tool_call = ToolCall(name="search", arguments={"query": "test"})
        assert tool_call.name == "search"
        assert tool_call.arguments == {"query": "test"}

    def test_tool_call_id_generation(self):
        """Test that ToolCall generates unique IDs automatically."""
        tool_call1 = ToolCall(name="test1")
        tool_call2 = ToolCall(name="test2")
        assert tool_call1.id != tool_call2.id
        assert tool_call1.id.startswith("tool_")

    def test_tool_call_default_status(self):
        """Test that default status is PENDING."""
        tool_call = ToolCall(name="test")
        assert tool_call.status == CallStatus.PENDING


class TestToolCallStatusTransitions:
    """Test ToolCall status transitions."""

    def test_tool_call_status_pending_to_running(self):
        """Test transition from PENDING to RUNNING."""
        tool_call = ToolCall(name="test")
        assert tool_call.status == CallStatus.PENDING
        tool_call.status = CallStatus.RUNNING
        assert tool_call.status == CallStatus.RUNNING

    def test_tool_call_status_running_to_done(self):
        """Test transition from RUNNING to DONE."""
        tool_call = ToolCall(name="test", status=CallStatus.RUNNING)
        tool_call.status = CallStatus.DONE
        assert tool_call.status == CallStatus.DONE

    def test_tool_call_status_transitions_full_flow(self):
        """Test full status flow: PENDING -> RUNNING -> DONE."""
        tool_call = ToolCall(name="test")
        assert tool_call.status == CallStatus.PENDING
        tool_call.status = CallStatus.RUNNING
        assert tool_call.status == CallStatus.RUNNING
        tool_call.status = CallStatus.DONE
        assert tool_call.status == CallStatus.DONE


class TestToolCallDependencies:
    """Test ToolCall dependency tracking."""

    def test_tool_call_dependencies_tracking(self):
        """Test that dependencies can be tracked as list of IDs."""
        dep1_id = "tool_abc12345"
        dep2_id = "tool_def67890"
        tool_call = ToolCall(name="test", dependencies=[dep1_id, dep2_id])
        assert len(tool_call.dependencies) == 2
        assert dep1_id in tool_call.dependencies
        assert dep2_id in tool_call.dependencies

    def test_tool_call_empty_dependencies_by_default(self):
        """Test that dependencies default to empty list."""
        tool_call = ToolCall(name="test")
        assert tool_call.dependencies == []


class TestToolCallResult:
    """Test ToolCall result assignment."""

    def test_tool_call_result_assignment(self):
        """Test storing execution results."""
        tool_call = ToolCall(name="search", arguments={"query": "test"})
        result = {"results": ["item1", "item2"]}
        tool_call.result = result
        assert tool_call.result == result

    def test_tool_call_result_none_by_default(self):
        """Test that result defaults to None."""
        tool_call = ToolCall(name="test")
        assert tool_call.result is None

    def test_tool_call_result_with_different_types(self):
        """Test result can store various types."""
        tool_call = ToolCall(name="test")

        # String result
        tool_call.result = "string result"
        assert tool_call.result == "string result"

        # Dict result
        tool_call.result = {"key": "value"}
        assert tool_call.result == {"key": "value"}

        # List result
        tool_call.result = [1, 2, 3]
        assert tool_call.result == [1, 2, 3]


class TestToolCallFailedStatus:
    """Test ToolCall failure handling."""

    def test_tool_call_failed_status(self):
        """Test marking a tool call as failed."""
        tool_call = ToolCall(name="test", status=CallStatus.RUNNING)
        tool_call.status = CallStatus.FAILED
        assert tool_call.status == CallStatus.FAILED

    def test_tool_call_failed_with_error_info(self):
        """Test storing error information when failed."""
        tool_call = ToolCall(name="test", status=CallStatus.RUNNING)
        tool_call.status = CallStatus.FAILED
        tool_call.result = {"error": "Connection timeout"}
        assert tool_call.status == CallStatus.FAILED
        assert tool_call.result == {"error": "Connection timeout"}
