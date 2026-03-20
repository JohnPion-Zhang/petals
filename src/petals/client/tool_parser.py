"""Tool parser for extracting tool calls from LLM output."""

import json
import re
from typing import List

from petals.data_structures import ToolCall


class ToolParser:
    """Parses tool calls from LLM text output.

    Supports the syntax: <tool_call>tool_name({...})</tool_call>

    Args:
        strict: If True, raises ValueError for malformed tool calls.
                If False, ignores malformed calls silently.
    """

    def __init__(self, strict: bool = True):
        """Initialize the tool parser.

        Args:
            strict: Whether to raise errors on malformed tool calls.
        """
        self.strict = strict

    def parse(self, text: str) -> List[ToolCall]:
        """Parse tool calls from text.

        Args:
            text: The raw text output from the LLM.

        Returns:
            List of ToolCall objects parsed from the output.

        Raises:
            ValueError: If strict mode is enabled and a tool call
                       has invalid JSON arguments.
        """
        if not text:
            return []

        return list(self._parse_iter(text))

    def _parse_iter(self, text: str) -> List[ToolCall]:
        """Internal iterator-based parser."""
        # Find all tool_call tags
        start_pattern = re.compile(r'<tool_call>')
        end_tag = '</tool_call>'

        pos = 0
        while True:
            start_match = start_pattern.search(text, pos)
            if not start_match:
                break

            start_pos = start_match.end()

            # Find the matching end tag
            end_pos = text.find(end_tag, start_pos)
            if end_pos == -1:
                break

            # Extract content between tags
            content = text[start_pos:end_pos].strip()
            end_pos += len(end_tag)

            # Parse the tool call
            tool_call = self._parse_single(content)
            if tool_call is not None:
                yield tool_call

            pos = end_pos

    def _parse_single(self, content: str) -> ToolCall | None:
        """Parse a single tool call content.

        Args:
            content: Content between <tool_call> and </tool_call>

        Returns:
            ToolCall object or None if parsing failed in non-strict mode.
        """
        # Match: tool_name(args) or tool_name()
        match = re.match(r'^(\w+)\s*\((.*)\)\s*$', content, re.DOTALL)
        if not match:
            # No parentheses - treat as tool with no arguments
            tool_name = content.strip()
            if tool_name:
                return ToolCall(name=tool_name, arguments={})
            return None

        tool_name = match.group(1)
        args_str = match.group(2).strip()

        # Parse arguments
        if not args_str:
            return ToolCall(name=tool_name, arguments={})

        # Handle JSON object format
        if args_str.startswith('{'):
            return self._parse_json_args(tool_name, args_str)
        else:
            # Simple key=value format
            return self._parse_simple_args(tool_name, args_str)

    def _parse_json_args(self, tool_name: str, args_str: str) -> ToolCall | None:
        """Parse JSON arguments.

        Args:
            tool_name: Name of the tool.
            args_str: JSON string of arguments.

        Returns:
            ToolCall with parsed arguments, or None if parsing failed in non-strict mode.

        Raises:
            ValueError: If JSON parsing fails in strict mode.
        """
        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError as e:
            if self.strict:
                raise ValueError(f"Invalid arguments for {tool_name}: {args_str}") from e
            return None  # Skip this tool call in lenient mode

        return ToolCall(name=tool_name, arguments=arguments)

    def _parse_simple_args(self, tool_name: str, args_str: str) -> ToolCall:
        """Parse simple key=value arguments.

        Args:
            tool_name: Name of the tool.
            args_str: String of key=value pairs.

        Returns:
            ToolCall with parsed arguments.
        """
        result = {}
        for pair in args_str.split(','):
            if '=' in pair:
                key, value = pair.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Remove surrounding quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                result[key] = value

        return ToolCall(name=tool_name, arguments=result)

    def extract_tool_names(self, text: str) -> List[str]:
        """Extract just the tool names from text.

        Args:
            text: The raw text output from the LLM.

        Returns:
            List of tool names.
        """
        tool_calls = self.parse(text)
        return [tc.name for tc in tool_calls]

    def has_tool_calls(self, text: str) -> bool:
        """Check if text contains any tool calls.

        Args:
            text: The raw text output from the LLM.

        Returns:
            True if tool calls are present, False otherwise.
        """
        return len(self.parse(text)) > 0
