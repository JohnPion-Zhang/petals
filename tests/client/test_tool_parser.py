"""Tests for ToolParser - parses LLM output for tool call syntax."""

import pytest

from petals.client.tool_parser import ToolParser


class TestToolParser:
    """Test suite for ToolParser class."""

    def test_parse_valid_tool_call(self):
        """Test extracting a valid tool call with JSON arguments."""
        parser = ToolParser()
        text = '<tool_call>search({"query": "weather", "location": "NYC"})</tool_call>'

        result = parser.parse(text)

        assert len(result) == 1
        assert result[0].name == "search"
        assert result[0].arguments == {"query": "weather", "location": "NYC"}

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls from text."""
        parser = ToolParser()
        text = '''
        Let me search for that information.
        <tool_call>search({"query": "weather"})</tool_call>
        Then I'll calculate the result.
        <tool_call>calculator({"expression": "2+2"})</tool_call>
        '''

        result = parser.parse(text)

        assert len(result) == 2
        assert result[0].name == "search"
        assert result[0].arguments == {"query": "weather"}
        assert result[1].name == "calculator"
        assert result[1].arguments == {"expression": "2+2"}

    def test_parse_tool_with_json_arguments(self):
        """Test parsing JSON-like arguments correctly."""
        parser = ToolParser()
        text = '<tool_call>fetch({"url": "https://example.com", "method": "GET", "headers": {"Content-Type": "application/json"}})</tool_call>'

        result = parser.parse(text)

        assert len(result) == 1
        assert result[0].name == "fetch"
        assert result[0].arguments["url"] == "https://example.com"
        assert result[0].arguments["method"] == "GET"
        assert result[0].arguments["headers"] == {"Content-Type": "application/json"}

    def test_parse_tool_with_nested_objects(self):
        """Test parsing deeply nested dict arguments."""
        parser = ToolParser()
        text = '<tool_call>process({"config": {"nested": {"deep": {"value": 42}}, "list": [1, 2, 3]}})</tool_call>'

        result = parser.parse(text)

        assert len(result) == 1
        assert result[0].name == "process"
        assert result[0].arguments == {
            "config": {
                "nested": {
                    "deep": {
                        "value": 42
                    }
                },
                "list": [1, 2, 3]
            }
        }

    def test_parse_empty_string_returns_empty_list(self):
        """Test that empty string returns empty list."""
        parser = ToolParser()

        result = parser.parse("")

        assert result == []

    def test_parse_no_tool_calls_returns_empty_list(self):
        """Test that text without tool calls returns empty list."""
        parser = ToolParser()
        text = "This is just regular text without any tool calls."

        result = parser.parse(text)

        assert result == []

    def test_parse_malformed_call_raises_in_strict_mode(self):
        """Test that malformed tool call raises ValueError in strict mode."""
        parser = ToolParser(strict=True)
        text = '<tool_call>search({invalid json})</tool_call>'

        with pytest.raises(ValueError, match="Invalid arguments for search"):
            parser.parse(text)

    def test_parse_malformed_call_ignores_in_lenient_mode(self):
        """Test that malformed tool call is ignored in lenient mode."""
        parser = ToolParser(strict=False)
        text = '<tool_call>search({invalid json})</tool_call>'

        result = parser.parse(text)

        assert result == []

    def test_parse_tool_with_special_characters(self):
        """Test parsing tool calls with quotes and escaping."""
        parser = ToolParser()
        text = '<tool_call>message({"content": "Hello \\"World\\"!", "special": "\\\\"})</tool_call>'

        result = parser.parse(text)

        assert len(result) == 1
        assert result[0].name == "message"
        assert result[0].arguments["content"] == 'Hello "World"!'
        assert result[0].arguments["special"] == "\\"

    def test_parse_tool_with_empty_arguments(self):
        """Test parsing tool call with empty argument dict."""
        parser = ToolParser()
        text = '<tool_call>ping({})</tool_call>'

        result = parser.parse(text)

        assert len(result) == 1
        assert result[0].name == "ping"
        assert result[0].arguments == {}

    def test_parse_tool_with_no_arguments(self):
        """Test parsing tool call with no arguments."""
        parser = ToolParser()
        text = '<tool_call>ping()</tool_call>'

        result = parser.parse(text)

        assert len(result) == 1
        assert result[0].name == "ping"
        assert result[0].arguments == {}

    def test_parse_tool_with_numeric_arguments(self):
        """Test parsing tool call with numeric arguments."""
        parser = ToolParser()
        text = '<tool_call>compute({"x": 10, "y": 20.5, "z": -5})</tool_call>'

        result = parser.parse(text)

        assert len(result) == 1
        assert result[0].arguments == {"x": 10, "y": 20.5, "z": -5}

    def test_parse_tool_with_boolean_arguments(self):
        """Test parsing tool call with boolean arguments."""
        parser = ToolParser()
        text = '<tool_call>configure({"enabled": true, "debug": false, "optional": null})</tool_call>'

        result = parser.parse(text)

        assert len(result) == 1
        assert result[0].arguments["enabled"] is True
        assert result[0].arguments["debug"] is False
        assert result[0].arguments["optional"] is None

    def test_parse_multiple_same_tool_calls(self):
        """Test parsing multiple calls to the same tool."""
        parser = ToolParser()
        text = '''
        <tool_call>search({"query": "first"})</tool_call>
        <tool_call>search({"query": "second"})</tool_call>
        <tool_call>search({"query": "third"})</tool_call>
        '''

        result = parser.parse(text)

        assert len(result) == 3
        assert all(call.name == "search" for call in result)
        assert result[0].arguments == {"query": "first"}
        assert result[1].arguments == {"query": "second"}
        assert result[2].arguments == {"query": "third"}

    def test_parse_tool_with_whitespace_variations(self):
        """Test parsing tool calls with varying whitespace."""
        parser = ToolParser()
        text = '''
        <tool_call>
            search(  {"query": "test"}  )
        </tool_call>
        '''

        result = parser.parse(text)

        assert len(result) == 1
        assert result[0].name == "search"
        assert result[0].arguments == {"query": "test"}

    def test_parse_tool_name_with_underscores_and_numbers(self):
        """Test parsing tool names that contain underscores and numbers."""
        parser = ToolParser()
        text = '<tool_call>get_user_data_2({"user_id": 123})</tool_call>'

        result = parser.parse(text)

        assert len(result) == 1
        assert result[0].name == "get_user_data_2"
        assert result[0].arguments == {"user_id": 123}
