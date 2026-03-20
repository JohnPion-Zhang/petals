"""Tests for AgentState data class - Red Phase (should fail initially)."""

import pytest
from petals.data_structures import AgentState, ToolCall, CallStatus


class TestAgentStateBasics:
    """Test AgentState basic functionality."""

    def test_agent_state_creation_defaults(self):
        """Test creating AgentState with default values."""
        state = AgentState()
        assert state.current_iteration == 0
        assert state.max_iterations == 10
        assert state.tool_history == []

    def test_agent_state_custom_iterations(self):
        """Test creating AgentState with custom iteration limits."""
        state = AgentState(max_iterations=5)
        assert state.max_iterations == 5


class TestAgentStateIterations:
    """Test iteration tracking in AgentState."""

    def test_agent_state_iteration_increment(self):
        """Test incrementing current iteration."""
        state = AgentState()
        assert state.current_iteration == 0
        state.current_iteration += 1
        assert state.current_iteration == 1

    def test_agent_state_multiple_iterations(self):
        """Test multiple iteration increments."""
        state = AgentState()
        for i in range(5):
            state.current_iteration += 1
        assert state.current_iteration == 5


class TestAgentStateToolHistory:
    """Test tool history management in AgentState."""

    def test_agent_state_tool_history_empty_initially(self):
        """Test that tool_history is empty by default."""
        state = AgentState()
        assert state.tool_history == []

    def test_agent_state_add_tool_call(self):
        """Test adding a tool call to history."""
        state = AgentState()
        tool_call = ToolCall(name="search", arguments={"query": "test"})
        state.tool_history.append(tool_call)
        assert len(state.tool_history) == 1
        assert state.tool_history[0].name == "search"

    def test_agent_state_multiple_tool_calls(self):
        """Test adding multiple tool calls to history."""
        state = AgentState()
        tool1 = ToolCall(name="search")
        tool2 = ToolCall(name="calculate")
        tool3 = ToolCall(name="format")
        state.tool_history.extend([tool1, tool2, tool3])
        assert len(state.tool_history) == 3

    def test_agent_state_tool_history_tracking(self):
        """Test that tool calls in history maintain their state."""
        state = AgentState()
        tool_call = ToolCall(name="search", arguments={"query": "test"})
        tool_call.status = CallStatus.RUNNING
        tool_call.result = {"results": ["item1", "item2"]}
        state.tool_history.append(tool_call)

        assert state.tool_history[0].status == CallStatus.RUNNING
        assert state.tool_history[0].result == {"results": ["item1", "item2"]}


class TestAgentStateIterationLimits:
    """Test iteration limit enforcement in AgentState."""

    def test_agent_state_max_iterations_default(self):
        """Test default max iterations."""
        state = AgentState()
        assert state.max_iterations == 10

    def test_agent_state_custom_max_iterations(self):
        """Test setting custom max iterations."""
        state = AgentState(max_iterations=20)
        assert state.max_iterations == 20

    def test_agent_state_iteration_limit_check(self):
        """Test checking if iteration limit is reached."""
        state = AgentState(max_iterations=3)
        state.current_iteration = 3
        assert state.current_iteration >= state.max_iterations
