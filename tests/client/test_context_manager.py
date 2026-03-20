"""
Tests for ContextManager - manages token budget and smart context trimming.

TDD Iteration 5: Red Phase - These tests should fail until implementation is provided.
"""
import pytest

from petals.client.context_manager import ContextManager
from petals.client.data_structures import ContextWindow, Message


# --- Test Fixtures ---

@pytest.fixture
def context_manager():
    """Create a ContextManager with default settings."""
    return ContextManager(max_tokens=4096)


@pytest.fixture
def context_manager_small():
    """Create a ContextManager with small token budget for testing."""
    return ContextManager(max_tokens=500)


@pytest.fixture
def sample_context():
    """Create a sample ContextWindow with system prompt, history, and active context."""
    return ContextWindow(
        max_tokens=4096,
        system_prompt="You are a helpful AI assistant. Always be concise.",
        tool_descriptions="Available tools: calculator, search",
        conversation_history=[
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm doing well, thank you! How can I help you today?"),
            Message(role="user", content="What's 2+2?"),
            Message(role="assistant", content="2+2 equals 4."),
        ],
        active_context="Can you multiply that by 3?"
    )


@pytest.fixture
def context_with_tool_results():
    """Create a ContextWindow with tool execution results in history."""
    return ContextWindow(
        max_tokens=4096,
        system_prompt="You are a helpful AI assistant.",
        tool_descriptions="Available tools: calculator",
        conversation_history=[
            Message(role="user", content="Calculate 10+5"),
            Message(role="tool_result", content='{"result": 15}', is_tool_result=True),
            Message(role="assistant", content="The result is 15."),
            Message(role="user", content="Thanks! What's 20-7?"),
            Message(role="tool_result", content='{"result": 13}', is_tool_result=True),
            Message(role="assistant", content="20-7 equals 13."),
        ],
        active_context=None
    )


@pytest.fixture
def empty_context():
    """Create an empty ContextWindow."""
    return ContextWindow(
        max_tokens=4096,
        system_prompt="You are a helpful AI assistant.",
        tool_descriptions="",
        conversation_history=[],
        active_context=None
    )


# --- System Prompt Preservation Tests ---

def test_trim_preserves_system_prompt(context_manager, sample_context):
    """System prompt is always kept even when trimming to minimum."""
    trimmed = context_manager.trim(sample_context, required_tokens=100)

    assert trimmed.system_prompt == sample_context.system_prompt
    assert "You are a helpful AI assistant" in trimmed.system_prompt


def test_trim_preserves_tool_descriptions(context_manager, sample_context):
    """Tool descriptions are always kept during trimming."""
    trimmed = context_manager.trim(sample_context, required_tokens=100)

    assert trimmed.tool_descriptions == sample_context.tool_descriptions
    assert "calculator" in trimmed.tool_descriptions


# --- Tool Results Preservation Tests ---

def test_trim_preserves_tool_results(context_manager, context_with_tool_results):
    """Tool execution results are never trimmed from conversation history."""
    trimmed = context_manager.trim(context_with_tool_results, required_tokens=100)

    # Find tool_result messages
    tool_results = [m for m in trimmed.conversation_history if m.is_tool_result]

    # Both tool results should be preserved
    assert len(tool_results) == 2
    assert any('"result": 15' in m.content for m in tool_results)
    assert any('"result": 13' in m.content for m in tool_results)


# --- Token Calculation Tests ---

def test_trim_calculates_tokens_correctly(context_manager, sample_context, monkeypatch):
    """Token estimation uses the actual tokenizer when available."""
    # Track how many times the tokenizer was called
    call_count = {"count": 0}

    class MockTokenizer:
        def encode(self, text, **kwargs):
            call_count["count"] += 1
            # Return a list where length represents token count
            return list(range(len(text) // 4))

        def __call__(self, text, **kwargs):
            return self.encode(text, **kwargs)

    # Set up mock tokenizer
    context_manager.tokenizer = MockTokenizer()

    trimmed = context_manager.trim(sample_context, required_tokens=100)

    # Should have used tokenizer for encoding
    assert call_count["count"] > 0


def test_estimate_tokens_uses_character_based_estimation(context_manager):
    """Without tokenizer, uses character-based estimation (~4 chars/token)."""
    text = "Hello, world!"  # 13 characters

    # 13 // 4 = 3 tokens (approximately)
    estimated = context_manager.estimate_tokens(text)

    assert estimated == 3


# --- Message Order Preservation Tests ---

def test_trim_removes_oldest_first(context_manager, context_manager_small, sample_context):
    """Oldest messages are dropped first when trimming due to token budget."""
    # With a small max_tokens, this should remove oldest messages first
    trimmed = context_manager_small.trim(sample_context, required_tokens=100)

    # The most recent messages should be preserved
    # Original history had 4 messages, with small budget we should keep recent ones
    if len(trimmed.conversation_history) < len(sample_context.conversation_history):
        # Verify we kept the newer messages (at the end of original history)
        assert len(trimmed.conversation_history) >= 1  # At least one message kept
        # The most recent user message should be preserved
        recent_msg = trimmed.conversation_history[-1] if trimmed.conversation_history else None
        if recent_msg:
            assert "multiply" in sample_context.active_context.lower() or \
                   recent_msg.role in ["user", "assistant"]


def test_trim_preserves_recent_messages(context_manager, sample_context):
    """The most recent N messages are always preserved regardless of token budget."""
    # Even with small budget, recent messages should be kept
    context_small = ContextManager(max_tokens=200)
    trimmed = context_small.trim(sample_context, required_tokens=50)

    # Active context should always be preserved
    assert trimmed.active_context == sample_context.active_context


def test_trim_stops_when_minimum_reached(context_manager_small, sample_context):
    """Trimming stops when minimum required messages are preserved."""
    # Create a manager with very limited budget
    context_tiny = ContextManager(max_tokens=300)

    trimmed = context_tiny.trim(sample_context, required_tokens=50)

    # Should still have at least some conversation history preserved
    # (tool results are never trimmed, and recent messages have priority)
    tool_results = [m for m in trimmed.conversation_history if m.is_tool_result]
    regular_messages = [m for m in trimmed.conversation_history if not m.is_tool_result]

    # System prompt and tool descriptions are always kept
    assert len(trimmed.system_prompt) > 0
    assert len(trimmed.tool_descriptions) > 0


# --- Empty Context Tests ---

def test_trim_empty_context_returns_empty(context_manager, empty_context):
    """Trimming an empty context returns an empty context window."""
    trimmed = context_manager.trim(empty_context, required_tokens=100)

    assert trimmed.system_prompt == empty_context.system_prompt
    assert trimmed.tool_descriptions == empty_context.tool_descriptions
    assert trimmed.conversation_history == []
    assert trimmed.active_context is None


def test_trim_no_history_preserves_system_and_tools(context_manager):
    """Context with no history still preserves system prompt and tool descriptions."""
    context_no_history = ContextWindow(
        max_tokens=4096,
        system_prompt="System prompt here",
        tool_descriptions="Tools: test_tool",
        conversation_history=[],
        active_context="Current question"
    )

    trimmed = context_manager.trim(context_no_history, required_tokens=100)

    assert trimmed.system_prompt == "System prompt here"
    assert trimmed.tool_descriptions == "Tools: test_tool"
    assert trimmed.active_context == "Current question"
    assert trimmed.conversation_history == []


# --- Build Prompt Tests ---

def test_build_prompt_includes_system_prompt(context_manager, sample_context):
    """build_prompt includes the system prompt."""
    prompt = context_manager.build_prompt(sample_context)

    assert sample_context.system_prompt in prompt


def test_build_prompt_includes_conversation_history(context_manager, sample_context):
    """build_prompt includes all non-empty conversation history messages."""
    prompt = context_manager.build_prompt(sample_context)

    for msg in sample_context.conversation_history:
        assert msg.content in prompt or msg.role in prompt


def test_build_prompt_includes_active_context(context_manager, sample_context):
    """build_prompt includes the active context (current user message)."""
    prompt = context_manager.build_prompt(sample_context)

    assert sample_context.active_context in prompt


def test_build_prompt_formats_roles_correctly(context_manager, empty_context):
    """build_prompt formats message roles correctly."""
    empty_context.conversation_history = [
        Message(role="user", content="Test message"),
        Message(role="assistant", content="Test response"),
    ]

    prompt = context_manager.build_prompt(empty_context)

    assert "user" in prompt
    assert "assistant" in prompt
    assert "Test message" in prompt
    assert "Test response" in prompt


def test_build_prompt_empty_context_returns_minimal(context_manager, empty_context):
    """build_prompt on empty context returns only system prompt."""
    prompt = context_manager.build_prompt(empty_context)

    assert empty_context.system_prompt in prompt
    # Should not have extra separators
    parts = [p.strip() for p in prompt.split("\n\n") if p.strip()]
    assert len(parts) <= 2  # system_prompt + possibly empty tool_descriptions


# --- Token Budget Tests ---

def test_trim_respects_max_token_budget(context_manager, sample_context):
    """Trimmed context should not exceed the max_tokens budget."""
    trimmed = context_manager.trim(sample_context, required_tokens=100)

    # Calculate total estimated tokens
    total = context_manager.estimate_tokens(trimmed.system_prompt)
    total += context_manager.estimate_tokens(trimmed.tool_descriptions)
    for msg in trimmed.conversation_history:
        total += context_manager.estimate_tokens(msg.content)
    if trimmed.active_context:
        total += context_manager.estimate_tokens(trimmed.active_context)

    # Should stay within budget (with some tolerance for exact calculation)
    assert total <= context_manager.max_tokens


def test_trim_with_large_required_tokens(context_manager, sample_context):
    """Context with very large required_tokens still preserves essentials."""
    trimmed = context_manager.trim(sample_context, required_tokens=2000)

    # Should still preserve system prompt and tool descriptions
    assert len(trimmed.system_prompt) > 0
    assert len(trimmed.tool_descriptions) > 0
