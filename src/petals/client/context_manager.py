"""
ContextManager - manages token budget and smart context trimming.

Provides intelligent context window management for LLM interactions,
preserving essential elements (system prompt, tool descriptions, tool results)
while trimming conversation history to fit within token budgets.
"""
from typing import Optional, List

from petals.client.data_structures import ContextWindow, Message


class ContextManager:
    """Manages token budget and smart context trimming for LLM interactions.

    The ContextManager ensures that conversation contexts stay within token
    budgets by intelligently trimming conversation history while preserving
    essential elements:

    1. System prompt (always kept)
    2. Tool descriptions (always kept)
    3. Tool results from history (never trimmed)
    4. Recent conversation messages (prioritized from newest to oldest)

    Attributes:
        max_tokens: The maximum token budget for the context window.
        tokenizer: Optional tokenizer for accurate token counting.
    """

    def __init__(self, max_tokens: int = 4096, tokenizer=None):
        """Initialize the ContextManager.

        Args:
            max_tokens: Maximum token budget for context windows. Defaults to 4096.
            tokenizer: Optional tokenizer instance for accurate token counting.
                      If not provided, uses character-based estimation.
        """
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string.

        Uses the tokenizer if available, otherwise falls back to
        character-based estimation (~4 characters per token).

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated number of tokens.
        """
        if not text:
            return 0

        if self.tokenizer is not None:
            # Use the tokenizer's encode method
            tokens = self.tokenizer.encode(text)
            return len(tokens)

        # Fallback: character-based estimation (~4 chars per token)
        return len(text) // 4

    def trim(self, context: ContextWindow, required_tokens: int = 100) -> ContextWindow:
        """Smart context trimming that preserves essential elements.

        Trims conversation history to fit within the token budget while
        always preserving:
        - System prompt
        - Tool descriptions
        - Tool results (never trimmed)
        - Recent messages (prioritized from newest to oldest)

        Args:
            context: The context window to trim.
            required_tokens: Tokens to reserve for the response. Defaults to 100.

        Returns:
            A new ContextWindow with trimmed conversation history.
        """
        # Calculate tokens used by essential elements
        system_tokens = self.estimate_tokens(context.system_prompt)
        tool_desc_tokens = self.estimate_tokens(context.tool_descriptions)
        active_context_tokens = self.estimate_tokens(context.active_context or "")

        # Calculate available tokens for conversation history
        available = self.max_tokens - system_tokens - tool_desc_tokens - active_context_tokens - required_tokens
        available = max(available, 0)  # Ensure non-negative

        # Build result context, preserving essentials
        result = ContextWindow(
            max_tokens=context.max_tokens,
            system_prompt=context.system_prompt,
            tool_descriptions=context.tool_descriptions,
            conversation_history=[],
            active_context=context.active_context
        )

        # Separate tool results from regular messages
        # Tool results are never trimmed
        tool_results = [msg for msg in context.conversation_history if msg.is_tool_result]
        regular_messages = [msg for msg in context.conversation_history if not msg.is_tool_result]

        # Add tool results first (they're never trimmed)
        tool_result_tokens = sum(self.estimate_tokens(msg.content) for msg in tool_results)
        available -= tool_result_tokens
        result.conversation_history.extend(tool_results)

        # Add regular messages from newest to oldest until budget exhausted
        # This preserves recent context while dropping oldest messages first
        reversed_history = list(reversed(regular_messages))
        current_tokens = 0

        for msg in reversed_history:
            msg_tokens = self.estimate_tokens(msg.content)
            if current_tokens + msg_tokens <= available:
                # Insert at position after tool results to maintain chronological order
                insert_pos = len(tool_results)
                result.conversation_history.insert(insert_pos, msg)
                current_tokens += msg_tokens
            else:
                # Stop trimming - we can't fit more messages
                break

        return result

    def build_prompt(self, context: ContextWindow) -> str:
        """Build the full prompt string from a context window.

        Constructs a formatted prompt by combining:
        - System prompt
        - Tool descriptions (if present)
        - Conversation history
        - Active context (if present)

        Args:
            context: The context window to build prompt from.

        Returns:
            A formatted prompt string.
        """
        parts = []

        # Add system prompt
        if context.system_prompt:
            parts.append(context.system_prompt)

        # Add tool descriptions
        if context.tool_descriptions:
            parts.append(f"Available tools:\n{context.tool_descriptions}")

        # Add conversation history
        for msg in context.conversation_history:
            if msg.is_tool_result:
                # Tool results are formatted differently
                parts.append(f"[TOOL RESULT]: {msg.content}")
            else:
                parts.append(f"{msg.role}: {msg.content}")

        # Add active context
        if context.active_context:
            parts.append(f"user: {context.active_context}")

        return "\n\n".join(parts)

    def calculate_total_tokens(self, context: ContextWindow) -> int:
        """Calculate the total estimated tokens in a context window.

        Args:
            context: The context window to calculate tokens for.

        Returns:
            Total estimated token count.
        """
        total = self.estimate_tokens(context.system_prompt)
        total += self.estimate_tokens(context.tool_descriptions)
        total += self.estimate_tokens(context.active_context or "")

        for msg in context.conversation_history:
            total += self.estimate_tokens(msg.content)

        return total

    def fits_in_budget(self, context: ContextWindow, required_tokens: int = 100) -> bool:
        """Check if a context fits within the token budget.

        Args:
            context: The context window to check.
            required_tokens: Tokens to reserve for the response.

        Returns:
            True if the context fits, False otherwise.
        """
        return self.calculate_total_tokens(context) + required_tokens <= self.max_tokens
