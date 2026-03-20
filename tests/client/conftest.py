"""Pytest fixtures for agent orchestrator tests."""

import pytest
from typing import List, Dict, Any, Optional


class MockLLM:
    """Mock LLM client for testing agent behavior."""

    def __init__(self, responses: Optional[List[str]] = None):
        """
        Initialize the mock LLM.

        Args:
            responses: List of responses to return in order.
        """
        self.responses = responses or []
        self.call_count = 0
        self.call_history = []

    async def generate(self, prompt: str) -> str:
        """Return the next response in the list."""
        self.call_history.append(prompt)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "No more responses configured."


@pytest.fixture
def mock_llm():
    """Create a mock LLM client with no responses."""
    return MockLLM()


@pytest.fixture
def mock_llm_with_responses():
    """Factory to create a mock LLM with specific responses."""
    def _create(responses: List[str]) -> MockLLM:
        return MockLLM(responses)
    return _create


@pytest.fixture
async def search_tool():
    """Create a mock search tool."""

    async def search(query: str) -> str:
        return f"Search results for: {query}"

    return search


@pytest.fixture
async def calculator_tool():
    """Create a mock calculator tool."""

    async def calculate(expression: str) -> str:
        try:
            # Safe in test context - only basic arithmetic
            result = eval(expression)  # noqa: PGH001 - Safe in test
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    return calculate


@pytest.fixture
async def failing_tool():
    """Create a mock tool that always fails."""

    async def fail(*args, **kwargs):
        raise ConnectionError("Network timeout")

    return fail


@pytest.fixture
async def orchestrator(mock_llm, search_tool, calculator_tool):
    """Create an AgentOrchestrator with common tools."""
    from petals.client.agent import AgentOrchestrator

    tools = [
        {
            "name": "search",
            "func": search_tool,
            "schema": {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "name": "calculate",
            "func": calculator_tool,
            "schema": {
                "name": "calculate",
                "description": "Perform a calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    return AgentOrchestrator(mock_llm, tools)


@pytest.fixture
async def orchestrator_with_failing_tool(mock_llm, failing_tool):
    """Create an AgentOrchestrator with a failing tool."""
    from petals.client.agent import AgentOrchestrator

    tools = [
        {
            "name": "failing",
            "func": failing_tool,
            "schema": {
                "name": "failing",
                "description": "A tool that always fails",
                "parameters": {}
            }
        }
    ]
    return AgentOrchestrator(mock_llm, tools)


@pytest.fixture
async def orchestrator_no_tools(mock_llm):
    """Create an AgentOrchestrator with no tools."""
    from petals.client.agent import AgentOrchestrator
    return AgentOrchestrator(mock_llm, [])
