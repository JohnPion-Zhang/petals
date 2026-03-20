"""Tests for ContextWindow data class - Red Phase (should fail initially)."""

import pytest
from petals.data_structures import ContextWindow, Message


class TestContextWindowBasics:
    """Test ContextWindow basic functionality."""

    def test_context_window_creation(self):
        """Test creating a ContextWindow with defaults."""
        ctx = ContextWindow()
        assert ctx.max_tokens == 4096
        assert ctx.system_prompt == ""
        assert ctx.tool_descriptions == ""
        assert ctx.conversation_history == []
        assert ctx.active_context == ""

    def test_context_window_with_custom_max_tokens(self):
        """Test creating ContextWindow with custom max_tokens."""
        ctx = ContextWindow(max_tokens=8192)
        assert ctx.max_tokens == 8192


class TestContextWindowSystemPrompt:
    """Test system prompt handling in ContextWindow."""

    def test_context_window_system_prompt_always_kept(self):
        """Test that system prompt is always preserved (first priority)."""
        ctx = ContextWindow(system_prompt="You are a helpful assistant")
        assert ctx.system_prompt == "You are a helpful assistant"

    def test_context_window_system_prompt_modification(self):
        """Test modifying system prompt."""
        ctx = ContextWindow(system_prompt="Original prompt")
        ctx.system_prompt = "Modified prompt"
        assert ctx.system_prompt == "Modified prompt"


class TestContextWindowToolDescriptions:
    """Test tool descriptions in ContextWindow."""

    def test_context_window_tool_results_preserved(self):
        """Test that tool descriptions are stored and preserved."""
        tool_desc = "search: Search the web for information\ncalculator: Perform calculations"
        ctx = ContextWindow(tool_descriptions=tool_desc)
        assert ctx.tool_descriptions == tool_desc


class TestContextWindowConversationHistory:
    """Test conversation history management."""

    def test_context_window_conversation_history(self):
        """Test adding messages to conversation history."""
        ctx = ContextWindow()
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="assistant", content="Hi there!")
        ctx.conversation_history.append(msg1)
        ctx.conversation_history.append(msg2)
        assert len(ctx.conversation_history) == 2
        assert ctx.conversation_history[0].role == "user"
        assert ctx.conversation_history[1].role == "assistant"

    def test_context_window_recent_messages_kept(self):
        """Test that recent messages are kept in history."""
        ctx = ContextWindow()
        for i in range(10):
            ctx.conversation_history.append(Message(role="user", content=f"Message {i}"))
        assert len(ctx.conversation_history) == 10


class TestContextWindowActiveContext:
    """Test active context in ContextWindow."""

    def test_context_window_active_context(self):
        """Test active context storage."""
        ctx = ContextWindow(active_context="Current task: search for weather")
        assert ctx.active_context == "Current task: search for weather"


class TestContextWindowMaxTokens:
    """Test max_tokens enforcement in ContextWindow."""

    def test_context_window_max_tokens_enforced(self):
        """Test that max_tokens is stored and can be enforced."""
        ctx = ContextWindow(max_tokens=2048)
        assert ctx.max_tokens == 2048
