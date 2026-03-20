"""
Data structures for agent context management.

Provides Message and ContextWindow classes used by the ContextManager.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    """Represents a single message in a conversation.

    Attributes:
        role: The role of the message sender (e.g., "user", "assistant", "system", "tool_result").
        content: The content of the message.
        is_tool_result: Whether this message is a result from a tool execution.
    """
    role: str
    content: str
    is_tool_result: bool = False

    def __post_init__(self):
        """Validate message fields."""
        if not self.role:
            raise ValueError("Message role cannot be empty")
        if self.content is None:
            raise ValueError("Message content cannot be None")


@dataclass
class ContextWindow:
    """Represents a complete conversation context window for LLM inference.

    Attributes:
        max_tokens: Maximum token budget for this context.
        system_prompt: The system prompt that sets the assistant's behavior.
        tool_descriptions: Descriptions of available tools.
        conversation_history: List of previous messages in the conversation.
        active_context: The current active context (e.g., current user message).
    """
    max_tokens: int
    system_prompt: str
    tool_descriptions: str = ""
    conversation_history: List[Message] = field(default_factory=list)
    active_context: Optional[str] = None

    def __post_init__(self):
        """Validate context window fields."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    @property
    def total_messages(self) -> int:
        """Return the total number of messages in the context."""
        return len(self.conversation_history)

    @property
    def has_active_context(self) -> bool:
        """Return whether there is an active context."""
        return self.active_context is not None and len(self.active_context) > 0
